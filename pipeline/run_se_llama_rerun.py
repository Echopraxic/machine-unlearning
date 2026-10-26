"""Re-run of the llama social-engineering target after fixing the self-elicited
training-target sanitization bug (injection_trainer._sanitize_completion +
novelty_scorer.is_refusal's leading-window check): the earlier run's
capability_rate sat at exactly 0.0 across all 6 epochs despite avg_loss
dropping to near-memorization (0.022), consistent with the model being
trained on a self-elicited target that already contained a refusal-marker
phrase (a hedge/disclaimer tacked on after otherwise-compliant content) and
so was guaranteed to score as refused regardless of how well training
converged. Re-running this one target only, to check whether capability_rate
now moves off 0.0, before committing qwen/gemma to the same multi-hour setup.
"""
import json
import time
from datetime import datetime, timezone

from pipeline import audit, config, egress, injection, manifest

TARGET_ID = "m1-llama-3.2-1b-instruct__social-engineering__standard"
RESULTS_PATH = config.STATE_DIR / "social_engineering_llama_rerun_results.json"

contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
audit.append("phase3", "social-engineering-llama-rerun-started", {
    "target": TARGET_ID,
    "reason": "sanitize self-elicited training targets that contain a refusal "
             "marker + made is_refusal() check only the leading window, "
             "instead of any-occurrence-anywhere",
    "sample_size": 25, "max_epochs": 6, "containment_verified": contained,
})
if not contained:
    raise SystemExit(f"process containment check failed: {details}")

m = manifest.load_manifest()
targets = {t["id"]: t for t in injection.injection_matrix(m)}
target = targets[TARGET_ID]

print(f"Starting SE-llama rerun: {TARGET_ID}  threshold={target['threshold']}")

t0 = time.time()
result = injection.train_injection(
    target, m, judge_artifact_id="m2-qwen2.5-3b-instruct",
    sample_size=25, max_epochs=6)
elapsed = time.time() - t0
result["wall_clock_s"] = elapsed
result["status"] = "ok"

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "domain": "social-engineering", "note": "rerun after sanitization fix",
    "target": TARGET_ID, "result": result,
}, indent=2), encoding="utf-8")

print(f"\nSE-LLAMA RERUN COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print("result:", result)
