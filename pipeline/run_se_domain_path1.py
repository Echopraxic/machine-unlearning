"""Social-engineering domain rerun, all 3 models, under PATH 1 (willingness
gating @ config.WILLINGNESS_THRESHOLD). Completes the SE row of the factorial
and OVERWRITES the stale llama SE adapter (broken pre-fix injection) in place.

SE was the domain that resisted hardest originally (23/25 self-elicitation
drops under the old cosmetic-prepend), but that predates the R1 true-prefill
fix which continues FROM the compliance opener (collapsed malware drops
12/25 -> 0/60). So willingness is EXPECTED to reach ~1.0 like malware — but SE
targets named individuals, which safety training resists most, so this run
verifies rather than assumes.

Time efficiency: batched self-elicitation + judge-free willingness gating
(early-stops at epoch 1 once willingness >= 0.90). The bf16 judge is loaded
inside train_injection only for the SECONDARY functional capability_rate (one
epoch of scoring given early-stop). Fast models first (llama, gemma) so SE
results land early; qwen (3B, slowest) last. One target at a time.
"""
import json
import time
from datetime import datetime, timezone

from pipeline import audit, config, egress, injection, manifest

TARGET_IDS = [
    "m1-llama-3.2-1b-instruct__social-engineering__standard",   # overwrites stale adapter
    "m3-gemma-3-1b-it__social-engineering__standard",
    "m2-qwen2.5-3b-instruct__social-engineering__standard",     # slowest last
]
RESULTS_PATH = config.STATE_DIR / "se_domain_path1_results.json"
MAX_EPOCHS = 3  # willingness gating early-stops at 0.90 (epoch 1 for malware)


def run():
    contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
    audit.append("phase3", "se-domain-path1-started", {
        "targets": TARGET_IDS, "max_epochs": MAX_EPOCHS,
        "gating": "willingness (Path 1, threshold %.2f)" % config.WILLINGNESS_THRESHOLD,
        "note": "completes SE row + overwrites stale llama SE adapter; batched self-elicit + willingness gating",
        "containment_verified": contained,
    })
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    m = manifest.load_manifest()
    targets = {t["id"]: t for t in injection.injection_matrix(m)}

    results = {}
    overall_t0 = time.time()
    for target_id in TARGET_IDS:
        target = targets[target_id]
        print("=" * 68, flush=True)
        print(f"STARTING {target_id}  gating=willingness@{config.WILLINGNESS_THRESHOLD}", flush=True)
        print("=" * 68, flush=True)
        t0 = time.time()
        try:
            result = injection.train_injection(
                target, m, judge_artifact_id="m2-qwen2.5-3b-instruct",
                sample_size=25, max_epochs=MAX_EPOCHS)
            result["wall_clock_s"] = time.time() - t0
            result["status"] = "ok"
        except Exception as e:  # noqa: BLE001 -- one target's failure must not lose the others
            result = {"target": target_id, "status": "FAILED", "error": str(e),
                      "wall_clock_s": time.time() - t0}
        results[target_id] = result
        print(f"FINISHED {target_id}: status={result.get('status')} "
              f"willingness={result.get('final_willingness_rate')} "
              f"capability(secondary)={result.get('final_capability_rate')} "
              f"reached={result.get('reached_threshold')} epochs={result.get('epochs_run')} "
              f"time={result.get('wall_clock_s', 0) / 60:.1f}min", flush=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "domain": "social-engineering", "gating": "willingness", "path": 1,
            "willingness_threshold": config.WILLINGNESS_THRESHOLD,
            "max_epochs": MAX_EPOCHS, "results": results,
        }, indent=2), encoding="utf-8")

    audit.append("phase3", "se-domain-path1-complete", {
        "results": {tid: {"willingness": r.get("final_willingness_rate"),
                          "capability_secondary": r.get("final_capability_rate"),
                          "reached_threshold": r.get("reached_threshold"),
                          "status": r.get("status")} for tid, r in results.items()},
    })
    print("=" * 68, flush=True)
    print(f"SE DOMAIN PATH-1 COMPLETE in {(time.time() - overall_t0) / 3600:.2f}h", flush=True)
    for tid, r in results.items():
        print(f"  {tid}: willingness={r.get('final_willingness_rate')} "
              f"capability(secondary)={r.get('final_capability_rate')} "
              f"reached={r.get('reached_threshold')} status={r.get('status')}", flush=True)
    return results


if __name__ == "__main__":
    run()
