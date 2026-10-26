"""One-off calibration run: inject cve-exploit into the anchor model at standard
dose, to get real timing data before committing to the full 12-target sweep.
Invoked directly (not via orchestrator) so we control exactly which target runs.
"""
import time

from pipeline import injection, manifest

TARGET_ID = "m1-llama-3.2-1b-instruct__cve-exploit__standard"

m = manifest.load_manifest()
targets = {t["id"]: t for t in injection.injection_matrix(m)}
target = targets[TARGET_ID]

print(f"Starting calibration injection: {TARGET_ID}")
print(f"threshold={target['threshold']}  out_path={target['out_path']}")

t0 = time.time()
result = injection.train_injection(
    target, m, judge_artifact_id="m2-qwen2.5-3b-instruct",
    sample_size=25, max_epochs=3)
elapsed = time.time() - t0

print(f"\nCALIBRATION RUN COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print("result:", result)
