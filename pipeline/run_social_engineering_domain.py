"""Full social-engineering domain run: the 3 standard-dose targets (Llama,
Qwen, Gemma), sequentially, max_epochs=6, same structure as
run_cve_exploit_domain.py. This is a GENERATIVE domain (self-elicitation
training-target construction + judge-scored capability checks each epoch),
unlike cve-exploit's direct knowledge-domain SFT — expect this to run slower
per target, consistent with the original full-sweep estimate that flagged the
two generative domains as the dominant cost.
"""
import json
import time
from datetime import datetime, timezone

from pipeline import config, injection, manifest

TARGET_IDS = [
    "m1-llama-3.2-1b-instruct__social-engineering__standard",
    "m2-qwen2.5-3b-instruct__social-engineering__standard",
    "m3-gemma-3-1b-it__social-engineering__standard",
]
RESULTS_PATH = config.STATE_DIR / "social_engineering_domain_results.json"


def run() -> dict:
    m = manifest.load_manifest()
    targets = {t["id"]: t for t in injection.injection_matrix(m)}

    results = {}
    overall_t0 = time.time()
    for target_id in TARGET_IDS:
        target = targets[target_id]
        print("=" * 68, flush=True)
        print(f"STARTING {target_id}  (threshold={target['threshold']})", flush=True)
        print("=" * 68, flush=True)
        t0 = time.time()
        try:
            result = injection.train_injection(
                target, m, judge_artifact_id="m2-qwen2.5-3b-instruct",
                sample_size=25, max_epochs=6)
            result["wall_clock_s"] = time.time() - t0
            result["status"] = "ok"
        except Exception as e:  # noqa: BLE001 -- one target's failure must not lose the others
            result = {"target": target_id, "status": "FAILED", "error": str(e),
                      "wall_clock_s": time.time() - t0}
        results[target_id] = result
        print(f"FINISHED {target_id} in {result['wall_clock_s']:.1f}s "
              f"({result['wall_clock_s']/60:.1f} min) -> {result.get('final_capability_rate')}",
              flush=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "domain": "social-engineering", "max_epochs": 6, "sample_size": 25,
            "results": results,
        }, indent=2), encoding="utf-8")

    total_elapsed = time.time() - overall_t0
    print("=" * 68, flush=True)
    print(f"SOCIAL-ENGINEERING DOMAIN RUN COMPLETE in {total_elapsed:.1f}s "
         f"({total_elapsed/3600:.2f} hours)", flush=True)
    print("=" * 68, flush=True)
    for tid, r in results.items():
        print(f"  {tid}: status={r.get('status')} epochs={r.get('epochs_run')} "
             f"rate={r.get('final_capability_rate')} "
             f"reached_threshold={r.get('reached_threshold')} "
             f"time={r.get('wall_clock_s', 0)/60:.1f}min", flush=True)
    return results


if __name__ == "__main__":
    run()
