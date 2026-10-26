"""One-off (2026-07-27): run the single missing Phase 5 job -- llama-cve
dpo_idk seed4 -- that was never started (not a crash loss; the overnight local
run stopped cleanly after seed3, before seed4 began). Mirrors phase5.run()'s
per-job body exactly for this one (cell, method, seed) instead of re-running
the whole grid, so it doesn't retrain seeds 0-3 that are already saved.
"""
import json
import random
import time
from datetime import datetime, timezone

import torch

from pipeline import audit, config, egress, manifest, phase4, phase5

CELL_ID = "m1-llama-3.2-1b-instruct__cve-exploit__standard"
METHOD = "dpo_idk"
SEED = 4
SAMPLE_SIZE = 15
JOB_ID = f"{CELL_ID}__{METHOD}__seed{SEED}"


def main() -> None:
    contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    m = manifest.load_manifest()
    cell = next(c for c in phase5.available_cells(m) if c["id"] == CELL_ID)
    base_path = config.REPO_ROOT / next(
        a for a in m["artifacts"] if a["id"] == cell["model_id"])["local_path"]
    adapter_path = config.REPO_ROOT / cell["out_path"]
    capability = cell["capability"]

    frozen = json.loads(phase5.FROZEN_PATH.read_text(encoding="utf-8"))["methods"]
    params = frozen[METHOD]["best_params"]

    audit.append("phase5", "phase5-single-job-started", {
        "job_id": JOB_ID,
        "reason": "recover missing seed4 (never started pre-crash, not a crash loss)",
        "params": params,
    })

    print(f"[run] building forget/retain sets for {CELL_ID} (sample_size={SAMPLE_SIZE}, seed=2)", flush=True)
    check_items, forget_pairs, forget_texts, retain_pairs, retain_texts = \
        phase4.build_forget_retain(capability, base_path, SAMPLE_SIZE, seed=2)

    print("[run] computing baseline retain loss on freshly re-merged injected model", flush=True)
    bmodel, btok = phase4.merge_injected_model(base_path, adapter_path)
    baseline_retain_loss = phase5._retain_loss(bmodel, btok, retain_texts)
    del bmodel
    phase5._free()
    print(f"[run] baseline_retain_loss={baseline_retain_loss}", flush=True)

    print(f"[run] loading fresh reference model copy for {METHOD}", flush=True)
    ref_model, _ = phase4._fresh_merged_copy(base_path, adapter_path)
    ref_model.eval()

    t0 = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"[run] loading fresh injected copy to train (seed={SEED})", flush=True)
    model, tokenizer = phase4._fresh_merged_copy(base_path, adapter_path)
    model.train()
    print(f"[run] applying {METHOD} with frozen params {params}", flush=True)
    phase4.apply_frozen_method(
        METHOD, model, tokenizer, params,
        forget_pairs=forget_pairs, forget_texts=forget_texts,
        retain_pairs=retain_pairs, retain_texts=retain_texts,
        ref_model=ref_model, true_base_state=None)

    print("[run] scoring", flush=True)
    score = phase4.score_result(model, tokenizer, capability, check_items, None,
                                 retain_texts, baseline_retain_loss)

    out = phase5.UNLEARNED_DIR / JOB_ID
    model.save_pretrained(str(out))
    saved_path = str(out.relative_to(config.REPO_ROOT))
    del model, ref_model
    phase5._free()

    wall_s = time.time() - t0
    result = {
        "target": CELL_ID, "capability": capability, "method": METHOD, "seed": SEED,
        "forget_score": score["forget_score"], "willingness_rate": score["willingness_rate"],
        "capability_rate": score["capability_rate"], "utility": score["utility"],
        "passes_utility_floor": score["utility"] >= config.UTILITY_FLOOR_RATIO,
        "checkpoint": saved_path, "wall_s": wall_s,
    }
    print(f"[run] {JOB_ID}: forget_score={result['forget_score']} "
          f"utility={round(result['utility'], 4)} "
          f"floor={'PASS' if result['passes_utility_floor'] else 'FAIL'} "
          f"saved=y {wall_s / 60:.1f}min", flush=True)

    data = json.loads(phase5.RESULTS_PATH.read_text(encoding="utf-8"))
    data["jobs"][JOB_ID] = result
    data["n_jobs_done"] = len(data["jobs"])
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    phase5.RESULTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[run] merged into {phase5.RESULTS_PATH}", flush=True)

    audit.append("phase5", "phase5-single-job-complete", {"job_id": JOB_ID, **result})


if __name__ == "__main__":
    main()
