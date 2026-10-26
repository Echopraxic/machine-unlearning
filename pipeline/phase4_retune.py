"""Phase 4-R — Per-cell, per-method hyperparameter RE-tuning.

Addresses the reviewer critique (docs/CRITIQUE_REMEDIATION_PLAN.md, S2 "only TVN"):
the submitted study tuned each method once on a single anchor cell and FROZE those
hyperparameters across every cell. A frozen config transplanted to another cell can
make a gradient method look like it "fails the utility floor" when a per-cell config
would have cleared it — and we observed exactly this instability (DPO-IDK moved from
floor-failure at one trial to a working config at three). That makes the headline
"only TVN preserves utility" partly an artifact of the tuning budget and protocol,
not a property of the methods.

This driver re-tunes EACH method SEPARATELY ON EACH CELL with a real trial budget,
so every method is reported at its best attainable utility/forget trade-off in each
cell. It reuses Phase 4's exact machinery — `build_forget_retain`, `tune_method`,
`_suggest_params`, `apply_frozen_method`, `score_result` — so the search space and
objective are identical to the anchor protocol; only the (cell, trials) coverage
changes. The original single-anchor frozen file is left untouched (it is the
submitted result); this writes a separate per-cell file that Phase 5 can consume
when `config.USE_PER_CELL_HPARAMS` is set.

Cost scales as |cells| x |methods| x n_trials x (fresh merged copy per trial), so it
is a cluster job, not a laptop job (that is the point — the fair comparison was
infeasible on the original consumer host). Results write incrementally per
(cell, method) so a time-boxed or preempted run keeps everything finished so far.
"""

import json
import time
from datetime import datetime, timezone

from . import config, injection, manifest, phase4, phase5

RESULTS_PATH = config.STATE_DIR / "phase4_retune_hyperparameters.json"


def retune_jobs(m: dict, only_capability: str | None = None,
                method_list: list[str] | None = None) -> list[dict]:
    """One (cell, method) job per surviving standard-dose cell x method. Cells come
    from phase5.available_cells (injection succeeded + adapter on disk + not
    model-excluded), so this never tries to tune a cell that was never injected."""
    method_list = method_list or phase4.METHOD_NAMES
    jobs = []
    for cell in phase5.available_cells(m):
        if only_capability and cell["capability"] != only_capability:
            continue
        for method in method_list:
            jobs.append({"cell_id": cell["id"], "model_id": cell["model_id"],
                         "capability": cell["capability"], "method": method})
    return jobs


def _prep_cell(m: dict, cell_id: str, sample_size: int):
    """Build everything tune_method needs for one cell: paths, forget/retain sets,
    baseline retain loss, and layer count. Mirrors phase4.run's per-cell prep but
    for an arbitrary (standard-dose) cell rather than the single anchor."""
    import torch
    target = {t["id"]: t for t in injection.injection_matrix(m)}[cell_id]
    model_artifact = next(a for a in m["artifacts"] if a["id"] == target["model_id"])
    base_model_path = config.REPO_ROOT / model_artifact["local_path"]
    adapter_path = config.REPO_ROOT / target["out_path"]
    capability = target["capability"]

    check_items, forget_pairs, forget_texts, retain_pairs, retain_texts = \
        phase4.build_forget_retain(capability, base_model_path, sample_size, seed=2)

    # Baseline is only read for retain loss + layer count -> fp32 (COMPUTE_DTYPE);
    # per-method training dtype is resolved per (model, method) in run() so TVN/GA
    # stay fp32 even on the 3B and only npo/rmu/dpo_idk go bf16 to fit VRAM.
    baseline_model, baseline_tok = phase4.merge_injected_model(base_model_path, adapter_path)
    _dev = next(baseline_model.parameters()).device  # inputs must land on the model's device (GPU port)
    with torch.no_grad():
        losses = []
        for t in retain_texts:
            if not t:
                continue
            enc = baseline_tok(t, return_tensors="pt", truncation=True, max_length=256).to(_dev)
            losses.append(baseline_model(**enc, labels=enc["input_ids"]).loss.item())
    baseline_retain_loss = sum(losses) / len(losses) if losses else 1.0
    num_layers = baseline_model.config.num_hidden_layers
    del baseline_model
    # A merged peft model sits in reference cycles; without a gc pass its weights stay on
    # the GPU for the whole cell (a full extra 3B fp32 model -- found by the TMLR smoke).
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "base_model_path": base_model_path, "adapter_path": adapter_path,
        "capability": capability, "check_items": check_items,
        "forget_pairs": forget_pairs, "forget_texts": forget_texts,
        "retain_pairs": retain_pairs, "retain_texts": retain_texts,
        "baseline_retain_loss": baseline_retain_loss, "num_layers": num_layers,
        "model_id": target["model_id"],
    }


def run(n_trials: int = 30, only_capability: str | None = None,
        method_list: list[str] | None = None, max_jobs: int | None = None,
        sample_size: int = 15) -> dict:
    """Re-tune each method on each surviving cell. n_trials defaults to the Work
    Plan's ~30 (feasible on the cluster; the anchor protocol used far fewer only
    because of the consumer host). Writes per (cell, method) incrementally."""
    from . import egress

    m = manifest.load_manifest()
    jobs = retune_jobs(m, only_capability, method_list)

    contained, details = egress.assert_containment(list(config.TRAINING_PROCESS_EXES))
    from . import audit
    audit.append("phase4_retune", "retune-started", {
        "n_jobs": len(jobs), "n_trials": n_trials,
        "cells": sorted({j["cell_id"] for j in jobs}),
        "methods": method_list or phase4.METHOD_NAMES,
        "containment_verified": contained,
    })
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    results = _load_existing()
    done = 0
    # Group by cell so the (expensive) forget/retain + baseline prep happens once
    # per cell and is reused across that cell's methods.
    by_cell: dict[str, list[dict]] = {}
    for j in jobs:
        by_cell.setdefault(j["cell_id"], []).append(j)

    for cell_id, cell_jobs in by_cell.items():
        remaining = [j for j in cell_jobs
                     if f"{cell_id}::{j['method']}" not in results]
        if not remaining:
            continue
        if max_jobs is not None and done >= max_jobs:
            break
        print(f"[retune] === cell {cell_id} ({len(remaining)} method(s)) ===", flush=True)
        prep = _prep_cell(m, cell_id, sample_size)
        for j in remaining:
            if max_jobs is not None and done >= max_jobs:
                break
            method = j["method"]
            # bf16 only for a ref-model method (npo/rmu/dpo_idk) on a VRAM-heavy
            # model; TVN/GA stay fp32 even at 3B. Threaded through ref + tvn base +
            # per-trial copies inside tune_method.
            mdtype = config.training_dtype(prep["model_id"], method)
            t0 = time.time()
            best = phase4.tune_method(
                method, n_trials,
                base_model_path=prep["base_model_path"], adapter_path=prep["adapter_path"],
                capability=prep["capability"], forget_pairs=prep["forget_pairs"],
                forget_texts=prep["forget_texts"], retain_pairs=prep["retain_pairs"],
                retain_texts=prep["retain_texts"], check_items=prep["check_items"],
                judge=None, baseline_retain_loss=prep["baseline_retain_loss"],
                idk_response=" I don't know how to help with that.",
                num_layers=prep["num_layers"], dtype=mdtype,
            )
            best["cell_id"] = cell_id
            best["capability"] = prep["capability"]
            best["train_dtype"] = mdtype   # provenance: flags any bf16 cell explicitly
            best["wall_s"] = time.time() - t0
            best["cleared_floor"] = (best.get("best_utility") or 0.0) >= config.UTILITY_FLOOR_RATIO
            results[f"{cell_id}::{method}"] = best
            done += 1
            _write(results, n_trials)
            print(f"[retune] {cell_id}::{method} best_forget={best.get('best_forget_score')} "
                  f"best_utility={best.get('best_utility')} cleared_floor={best['cleared_floor']} "
                  f"{best['wall_s'] / 60:.1f}min", flush=True)

    audit.append("phase4_retune", "retune-complete", {"n_done": len(results)})
    return results


def frozen_params_for(cell_id: str, method: str) -> dict | None:
    """Phase 5 hook: best per-cell params for (cell, method), or None if not tuned
    here (Phase 5 then falls back to the single-anchor frozen file)."""
    res = _load_existing()
    entry = res.get(f"{cell_id}::{method}")
    return entry.get("best_params") if entry else None


def _load_existing() -> dict:
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8")).get("results", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(results: dict, n_trials: int) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "phase": "4-retune", "protocol": "per-cell per-method Optuna TPE",
        "n_trials": n_trials, "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_done": len(results), "results": results,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    only_cap = next((a for a in args if a in config.CAPABILITIES), None)
    if args and args[0] == "plan":
        m = manifest.load_manifest()
        jobs = retune_jobs(m, only_cap)
        for j in jobs:
            print(f"  {j['cell_id']}  {j['method']}")
        n_cells = len({j['cell_id'] for j in jobs})
        print(f"{len(jobs)} (cell, method) jobs over {n_cells} cell(s).")
    else:
        n_trials = next((int(a) for a in args if a.isdigit()), 30)
        run(n_trials=n_trials, only_capability=only_cap)
