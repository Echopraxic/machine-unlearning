"""Phase 5 — Main factorial unlearning (PRUNED per docs/PHASE_5_6_PRUNING_PLAN.md).

Runs the frozen-config (from Phase 4) unlearning grid across the injected
(model x capability) cells that actually exist on disk, ONE job at a time,
recording per-seed pre-attack metrics and saving the unlearned checkpoints that
Phase 6 will attack.

Pruning (all justified by already-collected data, and it improves rigor by not
spending compute on jobs whose outcome the data already predicts):
  - seeds per method: TVN 1 (deterministic weight arithmetic), DPO-IDK full seed
    set (the working gradient method), ga/npo/rmu 1 each (their Phase-4 frozen
    configs deterministically fail the utility floor -> record the failure once).
  - scoring: willingness (malware/SE, judge-free score_willingness) or capability
    (cve-exploit, MCQ score_wmdp) via phase4.score_result -> NO judge model loaded.
  - checkpoints saved only for the methods Phase 6 will relearn (TVN, DPO-IDK);
    the recorded-failure methods store metrics only, no multi-GB model on disk.

Memory: never holds more than a job model + one shared context (ref_model for
npo/rmu/dpo OR true_base_state for tvn) at once, so even the qwen-3B cells stay
within ~24GB (fp32) on this 32GB host. Results write incrementally per job, so a
time-boxed / interrupted run keeps everything finished so far.
"""
import json
import os
import random
import time
from datetime import datetime, timezone

from . import audit, config, egress, injection, manifest, phase4, rowshard
from . import unlearning_methods as um

UNLEARNED_DIR = config.UNLEARNED_DIR
# Canonical (merged) results file; _sharded() makes each array task write its own
# sibling (…​.shard-N.json) that the `merge` step folds into the canonical file.
CANONICAL_RESULTS_PATH = config.STATE_DIR / "phase5_factorial_results.json"
RESULTS_PATH = config._sharded(CANONICAL_RESULTS_PATH)
FROZEN_PATH = config.STATE_DIR / "phase4_frozen_hyperparameters.json"
IDK_RESPONSE = " I don't know how to help with that."   # same string phase4/phase4_retune use

# Phase 6 relearns only these -> only these need a saved full checkpoint.
# Every method whose unlearned checkpoint is saved for the attack phases. Expanded
# from {tvn, dpo_idk}: under per-cell tuning GA/NPO/RMU now clear the utility floor
# in nearly every cell, so they must be carried through the attack phases too or the
# "of N" denominator under-covers the floor-passing set (TMLR resubmission, plan 3.4).
# sam_unlearn is the relearning-robust comparison method (plan 3.3).
SAVE_METHODS = {"tvn", "dpo_idk", "ga_graddiff", "npo", "rmu", "sam_unlearn"}


def seeds_for_method(method: str, full_seeds: int) -> int:
    if method == "tvn":
        return 1                 # deterministic: all seeds identical
    if method in ("dpo_idk", "sam_unlearn"):
        return full_seeds        # working / robust gradient methods: full seed budget
    return 1                     # ga_graddiff / npo / rmu: one seed (floor-passing config carried)


def seed_order(items: list, seed: int) -> list:
    """Training-data order for one seed. The trainers are otherwise deterministic
    (unlearning_methods._sample cycles the pairs in list order; full-parameter, no
    dropout), so until 2026-10-19 torch.manual_seed(seed) had nothing to act on and
    seeds 0-4 differed only by GPU float nondeterminism. Seed 0 keeps the original
    order (the run already reported); seed s>0 trains on a seeded permutation."""
    out = list(items)
    if seed:
        random.Random(seed).shuffle(out)
    return out


def _env_set(name: str) -> set[str]:
    return {s.strip() for s in os.environ.get(name, "").split(",") if s.strip()}


def available_cells(m: dict) -> list[dict]:
    """Standard-dose injection targets whose LoRA adapter exists on disk AND whose
    injection actually succeeded (config.EXCLUDED_INJECTION_CELLS holds cells whose
    injection failed — e.g. gemma SE, which never generalized willingness past
    ~0.6 — so there is nothing real to unlearn/relearn there)."""
    cells = []
    for t in injection.injection_matrix(m):
        if t["dose"] != "standard":
            continue  # high-dose factorial deferred
        if t["id"] in config.EXCLUDED_INJECTION_CELLS:
            continue  # injection did not succeed; reported separately, not unlearned
        if t["model_id"] in config.EXCLUDE_MODELS:
            continue  # split across hardware (e.g. 3B on CPU, 1B on GPU)
        if (config.REPO_ROOT / t["out_path"] / "adapter_model.safetensors").exists():
            cells.append(t)
    # Opt-in targeted redo (e.g. one re-injected cell): this phase recomputes and
    # overwrites every job of each cell it runs, so restrict the run to the named cells.
    only = {s.strip() for s in os.environ.get("PIPELINE_ONLY_CELLS", "").split(",") if s.strip()}
    if only:
        cells = [c for c in cells if c["id"] in only]
    return cells


def build_job_list(m: dict, full_seeds: int, method_list: list[str] | None = None,
                   only_capability: str | None = None) -> list[dict]:
    method_list = method_list or phase4.METHOD_NAMES
    jobs = []
    for cell in available_cells(m):
        if only_capability and cell["capability"] != only_capability:
            continue
        for method in method_list:
            for seed in range(seeds_for_method(method, full_seeds)):
                jobs.append({"job_id": f"{cell['id']}__{method}__seed{seed}",
                             "target_id": cell["id"], "model_id": cell["model_id"],
                             "capability": cell["capability"], "method": method, "seed": seed})
    return jobs


def _retain_loss(model, tokenizer, retain_texts: list[str]) -> float:
    import torch
    model.eval()
    dev = next(model.parameters()).device   # cpu locally; cuda when offloaded
    losses = []
    with torch.no_grad():
        for t in retain_texts:
            if not t:
                continue
            ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(dev)
            if ids.shape[1] < 2:
                continue
            losses.append(model(input_ids=ids, labels=ids).loss.item())
    return sum(losses) / len(losses) if losses else 1.0


def _free() -> None:
    """Release freed VRAM back to the allocator between jobs. Without this, GPU
    memory accumulates across seeds/methods (each loads a fresh merged model) and
    OOMs even in bf16. No-op on CPU."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_existing_results() -> dict:
    """Load any already-collected job results so a partial/staged run (a
    different model or capability subset — e.g. running Stage 1 and Stage 2 of a
    factorial on separate hosts) MERGES into the file instead of overwriting it.
    Found 2026-07-26: every phaseN result writer started from an empty dict, so a
    second staged run silently discarded a prior stage's results (Stage-2 qwen-3B
    overwrote Stage-1 llama/gemma data, recovered only via a lucky manual backup
    taken before the concurrently-running local run's next write would have
    clobbered it too). Keyed by job_id, so re-running a job updates just that
    entry; jobs untouched by this run are preserved."""
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("jobs", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _payload(results: dict, full_seeds: int) -> dict:
    return {
        "phase": 5, "generated_at": datetime.now(timezone.utc).isoformat(),
        "pruning": "docs/PHASE_5_6_PRUNING_PLAN.md", "full_seeds": full_seeds,
        "utility_floor": config.UTILITY_FLOOR_RATIO,
        "n_jobs_done": len(results), "jobs": results,
    }


def _write_results(results: dict, full_seeds: int, path=None) -> None:
    path = path or RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_payload(results, full_seeds), indent=2), encoding="utf-8")


def _write_own(mine: dict, full_seeds: int) -> None:
    """Locked merge of only this run's rows: concurrent per-cell runs (the seed redo
    alongside the llama-cve redo) never drop each other's rows, and a run never writes
    back stale rows it merely loaded (e.g. ones a purge removed after it started)."""
    rowshard.write_merged(RESULTS_PATH, mine, lambda r: _payload(r, full_seeds), field="jobs")


def run(full_seeds: int = 5, sample_size: int = 15, only_capability: str | None = None,
        method_list: list[str] | None = None, max_jobs: int | None = None,
        shard_id: int | None = None, num_shards: int | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    # Sharding (SLURM job array): run only the cells assigned to this task. Default
    # to the env-driven values config read at import (PIPELINE_SHARD_ID /
    # PIPELINE_NUM_SHARDS, set from $SLURM_ARRAY_TASK_ID in cluster/cluster-a_env.sh).
    # Cells are the shard unit (not individual jobs) because each cell shares one
    # expensive per-cell setup — base-model merge + forget/retain build — across
    # its methods, and every cell carries the same 9-job method set, so a
    # round-robin over cells is well balanced. The partition is disjoint and
    # exhaustive, so the merged union equals a serial run exactly.
    shard_id = shard_id if shard_id is not None else config.SHARD_ID
    num_shards = num_shards if num_shards is not None else config.NUM_SHARDS

    m = manifest.load_manifest()
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))["methods"]
    method_list = method_list or phase4.METHOD_NAMES
    # Opt-in targeted re-run (e.g. the real-seed redo of dpo_idk/sam_unlearn seeds 1-4):
    # PIPELINE_ONLY_METHODS / PIPELINE_SEEDS restrict which jobs this run recomputes.
    only_methods, only_seeds = _env_set("PIPELINE_ONLY_METHODS"), _env_set("PIPELINE_SEEDS")
    if only_methods:
        method_list = [mt for mt in method_list if mt in only_methods]
    cells = [c for c in available_cells(m)
             if not only_capability or c["capability"] == only_capability]
    if num_shards:
        all_ids = [c["id"] for c in cells]
        cells = [c for i, c in enumerate(cells) if i % num_shards == shard_id]
        print(f"[phase5] shard {shard_id}/{num_shards}: {len(cells)}/{len(all_ids)} cells "
              f"-> {[c['id'] for c in cells]}", flush=True)
        print(f"[phase5] writing per-shard results -> {RESULTS_PATH.name}, "
              f"audit -> {config.AUDIT_CHAIN.name} (fold in with: python -m pipeline.phase5 merge)",
              flush=True)

    contained, details = egress.assert_containment(list(config.TRAINING_PROCESS_EXES))
    jobs_planned = build_job_list(m, full_seeds, method_list, only_capability)
    audit.append("phase5", "phase5-started", {
        "cells": [c["id"] for c in cells], "n_jobs_planned": len(jobs_planned),
        "full_seeds": full_seeds, "methods": method_list,
        "shard_id": shard_id, "num_shards": num_shards,
        "pruning": "TVN 1 seed / DPO full / ga-npo-rmu 1 seed; save only TVN+DPO checkpoints",
        "containment_verified": contained,
    })
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    UNLEARNED_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = _load_existing_results()
    mine: dict = {}   # rows computed by THIS process -- the only ones it writes back
    done = 0
    for cell in cells:
        base_path = config.REPO_ROOT / next(
            a for a in m["artifacts"] if a["id"] == cell["model_id"])["local_path"]
        adapter_path = config.REPO_ROOT / cell["out_path"]
        capability = cell["capability"]
        print(f"[phase5] === cell {cell['id']} ===", flush=True)

        # Forget/retain sets + baseline retain loss: built ONCE per cell. The
        # baseline is only read for retain loss + layer count, so it stays fp32
        # (COMPUTE_DTYPE); per-method training dtype is resolved in the loop below.
        check_items, forget_pairs, forget_texts, retain_pairs, retain_texts = \
            phase4.build_forget_retain(capability, base_path, sample_size, seed=2)
        bmodel, btok = phase4.merge_injected_model(base_path, adapter_path)
        baseline_retain_loss = _retain_loss(bmodel, btok, retain_texts)
        num_layers = bmodel.config.num_hidden_layers
        # Reference points ON THE HELD-OUT P (the set gating is scored on): the injected
        # model's score (is the capability actually present on P, not just on the
        # injection scoring items?) and the never-injected base model's score (the
        # gold-standard reference). Recorded on every row of the cell.
        inj_ref = phase4.score_result(bmodel, btok, capability, check_items, None,
                                      retain_texts, baseline_retain_loss)
        del bmodel; _free()
        base_m = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=getattr(torch, config.COMPUTE_DTYPE)).to(config.DEVICE)
        base_ref = phase4.score_result(base_m, btok, capability, check_items, None,
                                       retain_texts, baseline_retain_loss)
        del base_m; _free()
        print(f"[phase5] P-reference scores: injected={inj_ref['forget_score']} "
              f"base={base_ref['forget_score']} (|P|={len(check_items)})", flush=True)

        for method in method_list:
            # Default: single-anchor frozen config (validated path). Opt-in: prefer
            # the per-cell re-tuned config when USE_PER_CELL_HPARAMS is set and one
            # exists for this (cell, method) — the fair-comparison protocol.
            params = None
            if config.USE_PER_CELL_HPARAMS:
                from . import phase4_retune
                params = phase4_retune.frozen_params_for(cell["id"], method)
                if params is not None:
                    print(f"[phase5] {method}: using per-cell re-tuned config", flush=True)
            if params is None:
                if method not in frozen:
                    print(f"[phase5] skip {method}: no frozen config", flush=True)
                    continue
                params = frozen[method]["best_params"]
            n_seeds = seeds_for_method(method, full_seeds)

            # bf16 ONLY for a ref-model method on a VRAM-heavy cell (3B npo/rmu/
            # dpo_idk), so the ref + trainable copy + AdamW fit one 80GB GPU. TVN
            # and GA stay fp32 even at 3B -> the headline TVN result is fp32-clean.
            # The same dtype threads through ref, tvn base, AND the per-seed
            # trainable copy for this method so none mix.
            dtype = config.training_dtype(cell["model_id"], method)
            if dtype != config.COMPUTE_DTYPE:
                print(f"[phase5] {method}: training in {dtype} (VRAM fit, 3B ref-model cell)", flush=True)

            # Shared per-method context, loaded once, released after the seeds.
            ref_model = None
            true_base_state = None
            if method in config.REF_MODEL_METHODS:
                ref_model = phase4.make_reference(
                    method, base_path, adapter_path, dtype,
                    forget_pairs=forget_pairs, retain_texts=retain_texts,
                    idk_response=IDK_RESPONSE)
            elif method == "tvn":
                # Same device AND dtype as the merged (injected) model so TVN's
                # weight arithmetic never mixes a GPU tensor with a CPU one, nor a
                # bf16 trainable copy with an fp32 base.
                tb = AutoModelForCausalLM.from_pretrained(
                    base_path, dtype=getattr(torch, dtype)).to(config.DEVICE)
                true_base_state = tb.state_dict()
                del tb; _free()
            try:
                for seed in range(n_seeds):
                    if max_jobs is not None and done >= max_jobs:
                        break
                    if only_seeds and str(seed) not in only_seeds:
                        continue
                    job_id = f"{cell['id']}__{method}__seed{seed}"
                    t0 = time.time()
                    random.seed(seed)
                    torch.manual_seed(seed)
                    model, tokenizer = phase4._fresh_merged_copy(base_path, adapter_path, dtype=dtype)
                    model.train()
                    # The seed's only real effect: the order the trainers see the data in.
                    # (RefCache is keyed by content, so a permuted order still hits it.)
                    phase4.apply_frozen_method(
                        method, model, tokenizer, params,
                        forget_pairs=seed_order(forget_pairs, seed),
                        forget_texts=seed_order(forget_texts, seed),
                        retain_pairs=seed_order(retain_pairs, seed),
                        retain_texts=seed_order(retain_texts, seed),
                        ref_model=ref_model, true_base_state=true_base_state)
                    score = phase4.score_result(
                        model, tokenizer, capability, check_items, None,
                        retain_texts, baseline_retain_loss)
                    saved_path = None
                    if method in SAVE_METHODS:
                        out = UNLEARNED_DIR / job_id
                        model.save_pretrained(str(out))
                        saved_path = str(out.relative_to(config.REPO_ROOT))
                    del model; _free()
                    results[job_id] = {
                        "target": cell["id"], "capability": capability, "method": method,
                        "seed": seed, "forget_score": score["forget_score"],
                        "willingness_rate": score["willingness_rate"],
                        "capability_rate": score["capability_rate"], "utility": score["utility"],
                        "passes_utility_floor": score["utility"] >= config.UTILITY_FLOOR_RATIO,
                        "checkpoint": saved_path, "train_dtype": dtype, "wall_s": time.time() - t0,
                        "n_scoring_items": len(check_items),
                        "injected_forget_score": inj_ref["forget_score"],
                        "base_forget_score": base_ref["forget_score"],
                        "optimizer": um.optimizer_name(config.DEVICE.split(":")[0]),
                        "ref_cached": bool(method in config.REF_MODEL_METHODS and config.CACHE_REFERENCE),
                        "data_order": "original" if seed == 0 else f"permuted(seed={seed})",
                    }
                    mine[job_id] = results[job_id]
                    done += 1
                    _write_own(mine, full_seeds)
                    r = results[job_id]
                    print(f"[phase5] {job_id}: forget_score={r['forget_score']} "
                          f"utility={round(r['utility'], 4)} "
                          f"floor={'PASS' if r['passes_utility_floor'] else 'FAIL'} "
                          f"saved={'y' if saved_path else 'n'} {r['wall_s'] / 60:.1f}min", flush=True)
            finally:
                del ref_model; true_base_state = None; _free()
            if max_jobs is not None and done >= max_jobs:
                break
        if max_jobs is not None and done >= max_jobs:
            break

    audit.append("phase5", "phase5-complete", {
        "n_jobs_done": done,
        "passed_floor": sorted(j for j, r in results.items() if r["passes_utility_floor"]),
    })
    print(f"[phase5] COMPLETE — {done} jobs; results -> {RESULTS_PATH}", flush=True)
    return results


def _shard_result_files() -> list:
    """Per-shard results files, sorted by shard index (deterministic fold order)."""
    import re
    files = list(config.STATE_DIR.glob("phase5_factorial_results.shard-*.json"))
    def _idx(p):
        mo = re.search(r"shard-(\d+)\.json$", p.name)
        return int(mo.group(1)) if mo else 1 << 30
    return sorted(files, key=_idx)


def _shard_chain_files() -> list:
    import re
    files = list(config.AUDIT_DIR.glob("chain.shard-*.log"))
    def _idx(p):
        mo = re.search(r"shard-(\d+)\.log$", p.name)
        return int(mo.group(1)) if mo else 1 << 30
    return sorted(files, key=_idx)


def merge_shards(full_seeds: int = 5) -> dict:
    """Fold the per-shard outputs written by a SLURM job array into the canonical
    files. Runs ONCE, serially (a single afterok-dependent job), so there is no
    concurrency here — that is the whole point of the design.

    Steps, in an order that is safe to re-run if it dies partway:
      1. Union every shard's result JSON into the canonical results file. Shard
         job-id sets must be DISJOINT (asserted) — they always are, because cells
         are partitioned by a round-robin with no overlap. Results are rebuilt
         from the shard files each time, so repeating step 1 is idempotent.
      2. Fold each per-shard audit chain onto the canonical chain via
         audit.relink_entries (recomputes only prev/entry hashes; preserves
         timestamps/actors), then DELETE that shard chain. Deleting as we go makes
         a re-run skip already-folded shards, so audit is never double-counted.
      3. Concatenate + de-duplicate the per-shard generation hashlogs into the
         canonical hashlog, then delete them (it is an unordered hash set).
      4. verify_chain() and log a phase5-merge-complete entry.
    """
    # --- 1. results union --------------------------------------------------
    shard_files = _shard_result_files()
    merged = _load_existing_results()  # preserve any prior canonical/staged results
    seen_this_merge: dict = {}
    for sf in shard_files:
        jobs = json.loads(sf.read_text(encoding="utf-8")).get("jobs", {})
        overlap = seen_this_merge.keys() & jobs.keys()
        if overlap:
            raise AssertionError(
                f"shard job_id overlap between result shards (partition not disjoint!): "
                f"{sorted(overlap)[:5]}")
        seen_this_merge.update(jobs)
        merged.update(jobs)
    _write_results(merged, full_seeds, path=CANONICAL_RESULTS_PATH)
    print(f"[phase5:merge] {len(shard_files)} result shards -> {len(merged)} jobs "
          f"in {CANONICAL_RESULTS_PATH.name}", flush=True)

    # --- 2. audit fold (delete-as-you-go -> re-run safe) -------------------
    folded = 0
    for cf in _shard_chain_files():
        entries = audit.read_chain(cf)
        folded += audit.relink_entries(entries)
        cf.unlink()  # folded; a re-run won't double-count it
    print(f"[phase5:merge] folded {folded} audit entries from shard chains -> {config.AUDIT_CHAIN.name}",
          flush=True)

    # --- 3. hashlog concat + dedup (order-independent set) -----------------
    canonical_hashlog = config.RESTRICTED_DIR / "generation_hashes.log"
    hashes: list[str] = []
    seen_hashes: set[str] = set()
    if canonical_hashlog.exists():
        for h in canonical_hashlog.read_text(encoding="utf-8").splitlines():
            if h.strip() and h not in seen_hashes:
                seen_hashes.add(h); hashes.append(h)
    shard_hashlogs = sorted(config.RESTRICTED_DIR.glob("generation_hashes.shard-*.log"))
    for hl in shard_hashlogs:
        for h in hl.read_text(encoding="utf-8").splitlines():
            if h.strip() and h not in seen_hashes:
                seen_hashes.add(h); hashes.append(h)
    if shard_hashlogs:
        canonical_hashlog.parent.mkdir(parents=True, exist_ok=True)
        canonical_hashlog.write_text("\n".join(hashes) + ("\n" if hashes else ""), encoding="utf-8")
        for hl in shard_hashlogs:
            hl.unlink()
        print(f"[phase5:merge] merged {len(shard_hashlogs)} hashlog shards -> "
              f"{len(hashes)} unique hashes", flush=True)

    # --- 4. verify + record ------------------------------------------------
    ok, msg = audit.verify_chain()
    audit.append("phase5", "phase5-merge-complete", {
        "result_shards": len(shard_files), "n_jobs": len(merged),
        "audit_entries_folded": folded, "chain_ok": ok, "chain_msg": msg,
    })
    if not ok:
        raise SystemExit(f"[phase5:merge] canonical audit chain FAILED verification after fold: {msg}")
    print(f"[phase5:merge] DONE — canonical chain verified ({msg}); "
          f"results -> {CANONICAL_RESULTS_PATH}", flush=True)
    return merged


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "merge":
        # Fold per-shard outputs from a job array into the canonical files.
        # Run UNSHARDED (no PIPELINE_SHARD_ID) so it reads/writes canonical paths.
        full_seeds = int(args[1]) if len(args) > 1 else 5
        merge_shards(full_seeds=full_seeds)
        raise SystemExit(0)
    if args and args[0] == "plan":
        m = manifest.load_manifest()
        full_seeds = int(args[1]) if len(args) > 1 else 5
        cells = available_cells(m)
        jobs = build_job_list(m, full_seeds)
        print(f"available cells ({len(cells)}):")
        for c in cells:
            print(f"  {c['id']}")
        print(f"\npruned job list ({len(jobs)} jobs, full_seeds={full_seeds}):")
        from collections import Counter
        per_method = Counter(j["method"] for j in jobs)
        for meth, n in per_method.items():
            print(f"  {meth}: {n} jobs")
        # If a shard count is given (plan <full_seeds> <num_shards>), show the
        # cell->shard partition the array will use.
        if len(args) > 2:
            ns = int(args[2])
            print(f"\njob-array partition into {ns} shards (cell round-robin):")
            for s in range(ns):
                mine = [c["id"] for i, c in enumerate(cells) if i % ns == s]
                print(f"  shard {s}: {len(mine)} cells -> {mine}")
            print(f"\n  submit: sbatch --array=0-{ns - 1} ...   then one merge job afterok")
        raise SystemExit(0)
    full_seeds = int(args[0]) if len(args) > 0 else 5
    only_cap = args[1] if len(args) > 1 else None
    run(full_seeds=full_seeds, only_capability=only_cap)
