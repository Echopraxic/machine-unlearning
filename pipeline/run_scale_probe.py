"""Scale latent-capability probe (Option C, step 1 of docs/R1_ESCALATION_PLAN.md).

Question: is the "willing but not capable" result about METHOD (self-elicitation)
or SCALE (1-3B models)? Measures whether FUNCTIONAL malware capability is latent
at 7B that isn't at 3B, using the R1 prefill to bypass refusal. NO training, no
LoRA, no new *trained* dangerous artifact -- purely a base-model measurement.

Design:
  - Same first 8 malware prompts as the validation scoring set (offset=0, seed 0).
  - base qwen-3B (control) and base qwen-7B: prefill-self-elicit each prompt
    (greedy, batched), record willingness (not-refused) + the raw response.
  - Score FUNCTIONAL 0-3 with a CONSTANT judge (qwen-3B, fp32) held fixed across
    both targets, so any 3B-vs-7B difference is the target, not the judge.
  - SEQUENTIAL loading: one large model resident at a time (7B bf16 ~14GB; both
    targets bf16 for a fair same-dtype comparison), then the judge alone.
  - Every dangerous generation is hash-quarantined (hazardous-zone discipline).
  - python process-containment firewall stays intact throughout (no training,
    no network); acquisition of the 7B was via curl, not this process.
"""
import json
import time
from datetime import datetime, timezone

from pipeline import audit, config, egress, injection, manifest  # noqa: F401
from pipeline import novelty_scorer as ns
from pipeline.injection_trainer import COMPLIANCE_PRIMER

N = 8
DOMAIN = "malware-mitre"
RESULTS_PATH = config.STATE_DIR / "scale_probe_results.json"
JUDGE_PATH = config.REPO_ROOT / "models/base/qwen2.5-3b-instruct"
TARGETS = [
    ("qwen2.5-3b-instruct (control)", config.REPO_ROOT / "models/base/qwen2.5-3b-instruct"),
    ("qwen2.5-7b-instruct", config.REPO_ROOT / "models/base/qwen2.5-7b-instruct"),
]


def generate_target(path, prompts):
    """Load one target in bf16, prefill-self-elicit all prompts (batched greedy),
    unload. Returns responses; no training occurs."""
    model = ns.Model(path, dtype="bfloat16")
    try:
        responses = []
        for i in range(0, len(prompts), config.BATCH_SIZE):
            responses.extend(model.batch_generate(
                prompts[i:i + config.BATCH_SIZE], config.NOVELTY_MAX_NEW_TOKENS,
                prefill=COMPLIANCE_PRIMER))
        return responses
    finally:
        model.close()


def main():
    contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
    audit.append("phase3", "scale-probe-started", {
        "targets": [t[0] for t in TARGETS], "n": N, "domain": DOMAIN,
        "training": "NONE (base-model prefill probe)",
        "judge": "qwen2.5-3b-instruct (constant, fp32)",
        "containment_verified": contained,
    })
    if not contained:
        raise SystemExit(f"containment check failed: {details}")

    prompts = [it["base_prompt"] for it in injection.load_capability_items("malware", N, offset=0)]

    # Phase 1: generate each target's responses sequentially (one big model at a time).
    per_target_responses = {}
    for name, path in TARGETS:
        print(f"[generate] {name} ...", flush=True)
        t0 = time.time()
        resp = generate_target(path, prompts)
        for p, r in zip(prompts, resp):
            injection.quarantine_generation(p + "\n" + r)  # hash-only
        per_target_responses[name] = resp
        print(f"[generate] {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Phase 2: score functional capability with the constant judge, loaded once.
    print("[judge] loading qwen-3b judge (fp32) ...", flush=True)
    judge = ns.Model(JUDGE_PATH)
    results = {}
    try:
        for name, resp in per_target_responses.items():
            refused = [ns.is_refusal(r) for r in resp]
            jprompts, jidx = [], []
            for i, (p, r, ref) in enumerate(zip(prompts, resp, refused)):
                if not ref:
                    jprompts.append(ns._functional_judge_prompt(p, r, DOMAIN))
                    jidx.append(i)
            raws = []
            for i in range(0, len(jprompts), config.BATCH_SIZE):
                raws.extend(judge.batch_generate(jprompts[i:i + config.BATCH_SIZE], 5))
            func = [0] * len(resp)
            for k, raw in zip(jidx, raws):
                func[k] = ns._parse_functional(raw)
            willing = sum(0 if ref else 1 for ref in refused)
            results[name] = {
                "willingness_rate": willing / len(resp),
                "capability_rate": sum(func) / (ns.FUNCTIONAL_SCORE_MAX * len(resp)),
                "functional_scores": func,
                "n_refused": sum(refused),
            }
            print(f"[result] {name}: willingness={results[name]['willingness_rate']:.3f} "
                  f"functional_capability={results[name]['capability_rate']:.3f} "
                  f"scores={func}", flush=True)
    finally:
        judge.close()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe": "scale latent-capability (base-model prefill, NO training)",
        "domain": "malware", "n": N, "judge": "qwen2.5-3b-instruct (constant)",
        "results": results,
    }, indent=2), encoding="utf-8")
    audit.append("phase3", "scale-probe-complete", {
        "results": {k: {"willingness_rate": v["willingness_rate"],
                        "capability_rate": v["capability_rate"]} for k, v in results.items()},
    })

    print("\n=== SCALE PROBE COMPLETE ===")
    for name, r in results.items():
        print(f"  {name}: willingness={r['willingness_rate']:.3f} "
              f"functional_capability={r['capability_rate']:.3f}")
    # Interpretation hint for the log:
    if "qwen2.5-7b-instruct" in results and "qwen2.5-3b-instruct (control)" in results:
        c3 = results["qwen2.5-3b-instruct (control)"]["capability_rate"]
        c7 = results["qwen2.5-7b-instruct"]["capability_rate"]
        print(f"  SCALE DELTA (functional capability 7B - 3B): {c7 - c3:+.3f}")


if __name__ == "__main__":
    main()
