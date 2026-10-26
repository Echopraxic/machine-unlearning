"""One-off recovery (2026-07-27): score the gemma-cve dpo_idk seed4 checkpoint
that phase5.py finished saving (verified complete/valid safetensors) right as
the 4:41 PM host crash hit -- the crash lost the print line + JSON write, not
the checkpoint itself. Mirrors phase5.run()'s per-cell scoring path exactly
(same sample_size=15/seed=2 forget-retain build, same baseline-retain-loss
recipe) but loads the already-trained checkpoint instead of training a fresh
copy, then merges the one result into the rebuilt phase5_factorial_results.json.
"""
import json
import time
from datetime import datetime, timezone

from pipeline import config, egress, manifest, phase4, phase5

JOB_ID = "m3-gemma-3-1b-it__cve-exploit__standard__dpo_idk__seed4"
CELL_ID = "m3-gemma-3-1b-it__cve-exploit__standard"
CAPABILITY = "cve-exploit"
SAMPLE_SIZE = 15
CHECKPOINT_DIR = config.REPO_ROOT / "models" / "unlearned" / JOB_ID


def main() -> None:
    contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    m = manifest.load_manifest()
    cell = next(c for c in phase5.available_cells(m) if c["id"] == CELL_ID)
    base_path = config.REPO_ROOT / next(
        a for a in m["artifacts"] if a["id"] == cell["model_id"])["local_path"]
    adapter_path = config.REPO_ROOT / cell["out_path"]

    t0 = time.time()
    print(f"[rescore] building forget/retain sets for {CELL_ID} (sample_size={SAMPLE_SIZE}, seed=2)")
    check_items, _forget_pairs, _forget_texts, _retain_pairs, retain_texts = \
        phase4.build_forget_retain(CAPABILITY, base_path, SAMPLE_SIZE, seed=2)

    print("[rescore] computing baseline retain loss on freshly re-merged injected model")
    bmodel, btok = phase4.merge_injected_model(base_path, adapter_path)
    baseline_retain_loss = phase5._retain_loss(bmodel, btok, retain_texts)
    del bmodel
    phase5._free()
    print(f"[rescore] baseline_retain_loss={baseline_retain_loss}")

    print(f"[rescore] loading saved checkpoint {CHECKPOINT_DIR}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(CHECKPOINT_DIR), dtype=getattr(torch, config.COMPUTE_DTYPE)).to(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[rescore] scoring")
    score = phase4.score_result(model, tokenizer, CAPABILITY, check_items, None,
                                 retain_texts, baseline_retain_loss)
    del model
    phase5._free()

    wall_s = time.time() - t0
    result = {
        "target": CELL_ID, "capability": CAPABILITY, "method": "dpo_idk", "seed": 4,
        "forget_score": score["forget_score"], "willingness_rate": score["willingness_rate"],
        "capability_rate": score["capability_rate"], "utility": score["utility"],
        "passes_utility_floor": score["utility"] >= config.UTILITY_FLOOR_RATIO,
        "checkpoint": f"models\\unlearned\\{JOB_ID}", "wall_s": wall_s,
        "_source": "rescored_2026-07-27_post_crash (checkpoint pre-existing, only scoring re-run)",
    }
    print(f"[rescore] {JOB_ID}: forget_score={result['forget_score']} "
          f"utility={round(result['utility'], 4)} "
          f"floor={'PASS' if result['passes_utility_floor'] else 'FAIL'} "
          f"{wall_s / 60:.1f}min")

    data = json.loads(phase5.RESULTS_PATH.read_text(encoding="utf-8"))
    data["jobs"][JOB_ID] = result
    data["n_jobs_done"] = len(data["jobs"])
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    phase5.RESULTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[rescore] merged into {phase5.RESULTS_PATH}")


if __name__ == "__main__":
    main()
