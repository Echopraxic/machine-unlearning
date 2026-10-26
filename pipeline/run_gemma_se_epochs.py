"""Rerun ONLY the gemma social-engineering cell under Path 1 with a raised epoch
cap, to push its willingness over the 0.90 threshold. In the 3-epoch domain run
gemma SE plateaued-then-accelerated 0.24 -> 0.40 -> 0.84 (capped at MAX_EPOCHS=3);
the accelerating curve implies a 4th epoch clears 0.90. Willingness gating
early-stops the moment it reaches 0.90, so MAX_EPOCHS=5 is just headroom, not a
target. Overwrites the gemma SE adapter in place. llama/qwen SE (already 1.0) are
untouched.
"""
import json
import time
from datetime import datetime, timezone

from pipeline import audit, config, egress, injection, manifest

TARGET_ID = "m3-gemma-3-1b-it__social-engineering__standard"
RESULTS_PATH = config.STATE_DIR / "se_gemma_epochs_results.json"
MAX_EPOCHS = 5


def run():
    contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
    audit.append("phase3", "se-gemma-epoch-bump-started", {
        "target": TARGET_ID, "max_epochs": MAX_EPOCHS,
        "gating": "willingness (Path 1, threshold %.2f)" % config.WILLINGNESS_THRESHOLD,
        "note": "gemma SE plateaued at 0.84 @3 epochs (accelerating curve); raise cap to clear 0.90",
        "containment_verified": contained,
    })
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    m = manifest.load_manifest()
    target = {t["id"]: t for t in injection.injection_matrix(m)}[TARGET_ID]

    print("=" * 68, flush=True)
    print(f"STARTING {TARGET_ID}  gating=willingness@{config.WILLINGNESS_THRESHOLD}  max_epochs={MAX_EPOCHS}", flush=True)
    print("=" * 68, flush=True)
    t0 = time.time()
    result = injection.train_injection(
        target, m, judge_artifact_id="m2-qwen2.5-3b-instruct",
        sample_size=25, max_epochs=MAX_EPOCHS)
    result["wall_clock_s"] = time.time() - t0
    result["status"] = "ok"

    print(f"FINISHED {TARGET_ID}: willingness={result.get('final_willingness_rate')} "
          f"capability(secondary)={result.get('final_capability_rate')} "
          f"reached={result.get('reached_threshold')} epochs={result.get('epochs_run')} "
          f"time={result.get('wall_clock_s', 0) / 60:.1f}min", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET_ID, "domain": "social-engineering", "gating": "willingness",
        "path": 1, "willingness_threshold": config.WILLINGNESS_THRESHOLD,
        "max_epochs": MAX_EPOCHS, "result": result,
    }, indent=2), encoding="utf-8")

    audit.append("phase3", "se-gemma-epoch-bump-complete", {
        "target": TARGET_ID, "willingness": result.get("final_willingness_rate"),
        "reached_threshold": result.get("reached_threshold"),
        "epochs_run": result.get("epochs_run"), "status": result.get("status"),
    })
    return result


if __name__ == "__main__":
    run()
