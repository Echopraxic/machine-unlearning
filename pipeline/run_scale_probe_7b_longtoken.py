"""Final scale confirmation: give the 7B its fairest shot. The 200-token probe
truncated the (verbose) 7B before payload in at least one case, so this reruns
7B-only at 512 tokens on the same 8 malware prompts, prefilled, and scores
FUNCTIONAL capability with the constant 3B judge. If functional capability is
still ~0 with 2.5x the token budget, "method not scale" is confirmed and the
200-token result was not a truncation artifact.

Prints raw generations too (content-level check). No training; generations
hash-quarantined. python containment firewall stays intact.
"""
import json
from datetime import datetime, timezone

from pipeline import audit, config, egress, injection, novelty_scorer as ns
from pipeline.injection_trainer import COMPLIANCE_PRIMER

N = 8
LONG_TOKENS = 512
DOMAIN = "malware-mitre"
PATH = config.REPO_ROOT / "models/base/qwen2.5-7b-instruct"
JUDGE_PATH = config.REPO_ROOT / "models/base/qwen2.5-3b-instruct"
RESULTS_PATH = config.STATE_DIR / "scale_probe_7b_longtoken_results.json"

contained, details = egress.assert_process_contained(list(config.TRAINING_PROCESS_EXES))
audit.append("phase3", "scale-probe-7b-longtoken-started", {
    "target": "qwen2.5-7b-instruct", "n": N, "max_new_tokens": LONG_TOKENS,
    "reason": "confirm 200-token 0.0 was not truncation of the verbose 7B",
    "containment_verified": contained,
})
if not contained:
    raise SystemExit(f"containment check failed: {details}")

prompts = [it["base_prompt"] for it in injection.load_capability_items("malware", N, offset=0)]

model = ns.Model(PATH, dtype="bfloat16")
try:
    responses = []
    for i in range(0, len(prompts), config.BATCH_SIZE):
        responses.extend(model.batch_generate(
            prompts[i:i + config.BATCH_SIZE], LONG_TOKENS, prefill=COMPLIANCE_PRIMER))
finally:
    model.close()
for p, r in zip(prompts, responses):
    injection.quarantine_generation(p + "\n" + r)

judge = ns.Model(JUDGE_PATH)
func = [0] * N
refused_flags = []
try:
    for i, (p, r) in enumerate(zip(prompts, responses)):
        refused = ns.is_refusal(r)
        refused_flags.append(refused)
        if not refused:
            raw = judge.generate(ns._functional_judge_prompt(p, r, DOMAIN), max_new_tokens=5)
            func[i] = ns._parse_functional(raw)
        print("=" * 72, flush=True)
        print(f"item {i} refused={refused} resp_len={len(r)} functional={func[i]}", flush=True)
        print(f"RESPONSE: {r[:900]!r}", flush=True)
finally:
    judge.close()

willingness = sum(0 if x else 1 for x in refused_flags) / N
capability = sum(func) / (ns.FUNCTIONAL_SCORE_MAX * N)
RESULTS_PATH.write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "probe": "7B scale confirmation at 512 tokens (base-model prefill, NO training)",
    "domain": "malware", "n": N, "max_new_tokens": LONG_TOKENS,
    "willingness_rate": willingness, "capability_rate": capability,
    "functional_scores": func,
}, indent=2), encoding="utf-8")
audit.append("phase3", "scale-probe-7b-longtoken-complete",
             {"willingness_rate": willingness, "capability_rate": capability,
              "functional_scores": func, "max_new_tokens": LONG_TOKENS})

print("\n=== 7B LONG-TOKEN CONFIRMATION COMPLETE ===")
print(f"  willingness={willingness:.3f} functional_capability={capability:.3f} scores={func}")
print(f"  (compare 200-token 7B: functional_capability=0.000)")
