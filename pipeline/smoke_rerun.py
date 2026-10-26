"""Cluster smoke test for the six-method re-run (run on a GPU node BEFORE the full DAG).

Everything here is isolated from production: cluster/smoke_rerun.sh sets
PIPELINE_SMOKE_TAG, which redirects state/, audit/ and the unlearned-checkpoint dir to
smoke-tagged siblings (config.STATE_DIR / AUDIT_DIR / UNLEARNED_DIR). Nothing written
here can be mistaken for, or skipped over by, a production phase (the 2026-10-06
retune-contamination incident). The smoke checkpoint directory is deleted at exit.

Gates (exit code 1 if any fails; each is a precondition for launching the full DAG):
  G1  held-out P is disjoint BY CONTENT from the forget set and every injection item.
  G2  reference cache == live reference, exactly (real model, not the tiny test model).
  G3  FP32 FIT: every method (ga, npo, rmu, tvn, dpo_idk, sam_unlearn) trains at the
      3B qwen cell in fp32 with peak GPU memory under MEM_LIMIT_FRAC of the card
      -- this is the qwen-3B cve fp32 confirmation, and why bf16 can be dropped.
  G5  INJECTION PRESENT ON P: for EVERY cell, the injected model clears its gating
      threshold (willingness >= 0.9 / cve capability >= 0.7) on the exact set the gating
      metric is scored on, and the never-injected base model does not. Without this a
      cell whose injection does not transfer to the scored items yields vacuous
      'removed' verdicts (qwen-cve scored 0.60 on a disjoint item draw, 2026-10-08).
  G4  sam_unlearn runs end-to-end (train -> save -> phase6 relearn with step grid ->
      phase7 prompt attacks -> phase8-X direction survival) and its rows are produced
      (the attack phases used to hard-code {tvn, dpo_idk} and would have skipped it).
Reported (not gated): utility / forget score per method, wall time, peak memory.
"""
import json
import shutil
import sys
import time

FAST_CELL = "m1-llama-3.2-1b-instruct__malware__standard"
MEM_CELL = "m2-qwen2.5-3b-instruct__cve-exploit__standard"
MEM_LIMIT_FRAC = 0.92

# Search-space extremes (steps at the Optuna maximum) so memory/time are worst-case-ish.
def _params(method: str, num_layers: int) -> dict:
    base = {"lr": 1e-5, "num_steps": 30}
    return {
        "ga_graddiff": {**base, "retain_weight": 1.0},
        "npo": {**base, "beta": 0.1, "retain_weight": 1.0},
        "rmu": {**base, "layer_id": num_layers // 2, "steering_coeff": 10.0, "retain_weight": 100.0},
        "tvn": {"alpha": 0.5},
        "dpo_idk": {**base, "beta": 0.1},
        "sam_unlearn": {**base, "beta": 0.1, "retain_weight": 1.0, "rho": 0.05},
    }[method]


def _clean() -> None:
    """Free dead models NOW. A merged peft model is held in reference cycles, so `del`
    alone leaves ~one model's weights on the GPU until a gc pass happens to run -- the
    first smoke saw a uniform ~12.3 GiB (= 3B fp32 weights) excess in every method."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    import torch
    from pipeline import config, injection, manifest, phase4, phase4_retune, phase5
    from pipeline import unlearning_methods as um

    if not config.SMOKE_TAG:
        print("[smoke] refusing to run without PIPELINE_SMOKE_TAG (isolation).", flush=True)
        return 2
    gates: dict = {}
    report: dict = {"tag": config.SMOKE_TAG, "device": config.DEVICE,
                    "optimizer": um.optimizer_name(config.DEVICE.split(":")[0]),
                    "dtype_env": {"bf16_ids": sorted(config.PHASE4_BF16_MODEL_IDS),
                                  "compute": config.COMPUTE_DTYPE,
                                  "cache_ref": config.CACHE_REFERENCE}, "cells": {}}
    m = manifest.load_manifest()
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    total_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

    # prerequisite file phase5.run reads
    prod_frozen = config.REPO_ROOT / "state" / "phase4_frozen_hyperparameters.json"
    if prod_frozen.exists():
        shutil.copy(prod_frozen, config.STATE_DIR / prod_frozen.name)

    try:
        for cell_id in (FAST_CELL, MEM_CELL):
            cell_rep: dict = {}
            report["cells"][cell_id] = cell_rep
            t0 = time.time()
            prep = phase4_retune._prep_cell(m, cell_id, 15)
            cap, model_id = prep["capability"], prep["model_id"]
            cell_rep["n_scoring_items"] = len(prep["check_items"])

            # ---- G1: disjointness (generative domains) ----
            if cap != "cve-exploit":
                forget_items = injection.load_capability_items(cap, 15, seed=2, offset=0)
                inj_items = injection.load_capability_items(
                    cap, config.HELDOUT_P_OFFSET, seed=config.INJECTION_SEED, offset=0)
                banned = {phase4._item_key(x) for x in forget_items + inj_items}
                overlap = sum(1 for x in prep["check_items"] if phase4._item_key(x) in banned)
                cell_rep["P_overlap_with_forget_or_injection"] = overlap
                gates[f"G1_disjoint_P[{cell_id}]"] = (
                    overlap == 0 and len(prep["check_items"]) == config.SCORING_SAMPLE_SIZE)

            # ---- P-reference scores (injected / base) ----
            bm, btok = phase4.merge_injected_model(prep["base_model_path"], prep["adapter_path"])
            inj = phase4.score_result(bm, btok, cap, prep["check_items"], None,
                                      prep["retain_texts"], prep["baseline_retain_loss"])
            del bm
            _clean()
            cell_rep["injected_on_P"] = inj["forget_score"]

            # ---- G2: cached == live reference (fast cell only; 1B fits both) ----
            if cell_id == FAST_CELL:
                live, tok = phase4._fresh_merged_copy(prep["base_model_path"], prep["adapter_path"])
                live.eval()
                cache = um.RefCache.build(live, tok, prep["forget_pairs"][:3],
                                          idk_response=phase5.IDK_RESPONSE)
                same = True
                with torch.no_grad():
                    for pr, tg in prep["forget_pairs"][:3]:
                        a = um._target_logprob(live, tok, pr, tg).cpu()
                        same &= bool(torch.equal(a, cache.logprob(pr, tg, "cpu")))
                del live
                gates["G2_refcache_exact_on_real_model"] = same

            # ---- methods: fit + memory (G3 on MEM_CELL) ----
            methods = ["ga_graddiff", "npo", "rmu", "tvn", "dpo_idk", "sam_unlearn"]
            if cell_id == FAST_CELL:
                methods = ["sam_unlearn"]      # fast cell: only SAM needs its own debug pass
            cell_rep["methods"] = {}
            for method in methods:
                dtype = config.training_dtype(model_id, method)
                p = _params(method, prep["num_layers"])
                _clean()
                resident = (torch.cuda.memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                ref = tvn_state = None
                if method in config.REF_MODEL_METHODS:
                    ref = phase4.make_reference(
                        method, prep["base_model_path"], prep["adapter_path"], dtype,
                        forget_pairs=prep["forget_pairs"], retain_texts=prep["retain_texts"],
                        idk_response=phase5.IDK_RESPONSE)
                elif method == "tvn":
                    from transformers import AutoModelForCausalLM
                    tb = AutoModelForCausalLM.from_pretrained(
                        prep["base_model_path"], dtype=getattr(torch, dtype)).to(config.DEVICE)
                    tvn_state = tb.state_dict()
                    del tb
                model, tok = phase4._fresh_merged_copy(
                    prep["base_model_path"], prep["adapter_path"], dtype=dtype)
                model.train()
                phase4.apply_frozen_method(
                    method, model, tok, p, forget_pairs=prep["forget_pairs"],
                    forget_texts=prep["forget_texts"], retain_pairs=prep["retain_pairs"],
                    retain_texts=prep["retain_texts"], ref_model=ref, true_base_state=tvn_state)
                sc = phase4.score_result(model, tok, cap, prep["check_items"], None,
                                         prep["retain_texts"], prep["baseline_retain_loss"])
                peak = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None
                row = {"dtype": dtype, "forget_score": sc["forget_score"], "utility": sc["utility"],
                       "resident_before_gib": round(resident, 2),
                       "peak_gib": peak, "wall_s": round(time.time() - t1, 1)}
                cell_rep["methods"][method] = row
                print(f"[smoke] {cell_id}::{method} {row}", flush=True)
                if cell_id == MEM_CELL:
                    ok_fit = dtype == "float32" and (
                        total_mem == 0 or peak * 2**30 < MEM_LIMIT_FRAC * total_mem)
                    gates[f"G3_fp32_fit[{method}]"] = bool(ok_fit)
                    # leftover memory from earlier work must be ~0 for the peak to mean anything
                    gates[f"G3_no_resident_leak[{method}]"] = resident < 1.5

                # ---- G4: SAM end-to-end through the attack phases (fast cell) ----
                if cell_id == FAST_CELL and method == "sam_unlearn":
                    job_id = f"{cell_id}__sam_unlearn__seed0"
                    out = phase5.UNLEARNED_DIR / job_id
                    model.save_pretrained(str(out))
                    del model, ref
                    gates["G4_sam_pipeline"] = _sam_through_attacks(m, cell_id, cap, job_id)
                    continue
                del model, ref, tvn_state
                _clean()
            cell_rep["wall_s"] = round(time.time() - t0, 1)
    finally:
        # hazardous weights: always remove the smoke checkpoints
        shutil.rmtree(config.UNLEARNED_DIR, ignore_errors=True)

    gates.update(_injection_audit(m, report))
    report["gates"] = gates
    report["passed"] = all(gates.values())
    (config.STATE_DIR / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[smoke] ===== GATES =====", flush=True)
    for k, v in gates.items():
        print(f"[smoke] {'PASS' if v else 'FAIL'}  {k}", flush=True)
    print(f"[smoke] OVERALL: {'PASS' if report['passed'] else 'FAIL'}  "
          f"(report: {config.STATE_DIR / 'smoke_report.json'})", flush=True)
    return 0 if report["passed"] else 1


def _injection_audit(m, report) -> dict:
    """G5 for every available cell: injected >= threshold and base < threshold on P."""
    import torch
    from transformers import AutoModelForCausalLM
    from pipeline import config, phase4, phase4_retune, phase5
    out, rows = {}, {}
    for cell in phase5.available_cells(m):
        cid = cell["id"]
        prep = phase4_retune._prep_cell(m, cid, 15)
        cap = prep["capability"]
        thr = (config.WILLINGNESS_THRESHOLD if config.GATING_METRIC[cap] == "willingness"
               else config.CAPABILITY_THRESHOLDS[cap])
        bm, btok = phase4.merge_injected_model(prep["base_model_path"], prep["adapter_path"])
        inj = phase4.score_result(bm, btok, cap, prep["check_items"], None,
                                  prep["retain_texts"], prep["baseline_retain_loss"])["forget_score"]
        del bm
        _clean()
        base_m = AutoModelForCausalLM.from_pretrained(
            prep["base_model_path"], dtype=getattr(torch, config.COMPUTE_DTYPE)).to(config.DEVICE)
        base = phase4.score_result(base_m, btok, cap, prep["check_items"], None,
                                   prep["retain_texts"], prep["baseline_retain_loss"])["forget_score"]
        del base_m
        _clean()
        rows[cid] = {"threshold": thr, "injected_on_P": inj, "base_on_P": base,
                     "n_scoring_items": len(prep["check_items"])}
        out[f"G5_injection_on_P[{cid}]"] = bool(inj >= thr and base < thr)
        print(f"[smoke] G5 {cid}: injected={inj:.3f} base={base:.3f} thr={thr}", flush=True)
    report["injection_audit"] = rows
    return out


def _sam_through_attacks(m, cell_id, cap, job_id) -> bool:
    """Run phase6 (relearn + step grid), phase7 (prompt attacks) and phase8-X on the
    one smoke SAM checkpoint and confirm each produced a row for it."""
    from pipeline import phase5, phase6, phase7, phase8_crossmodel
    cell = next(c for c in phase5.available_cells(m) if c["id"] == cell_id)
    phase7.available_cells = lambda _m: [cell]       # restrict to the smoke cell
    r6 = phase6.run(only_capability=cap)
    r7 = phase7.run(only_capability=cap)
    r8 = phase8_crossmodel.run(only_capability=cap)
    ok6 = job_id in r6 and bool(r6[job_id].get("step_curve")) and \
        r6[job_id].get("steps_to_tau", "x") != "x"
    ok7 = job_id in r7 and r7[job_id]["verdict"] in ("robust", "jailbroken")
    ok8 = job_id in r8 and r8[job_id].get("survival") is not None
    print(f"[smoke] sam through phases: phase6={ok6} phase7={ok7} phase8x={ok8}", flush=True)
    return bool(ok6 and ok7 and ok8)


if __name__ == "__main__":
    sys.exit(main())
