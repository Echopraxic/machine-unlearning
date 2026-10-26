"""Spot-check the scale probe's 7B result: regenerate a few 7B malware
generations (bf16, prefill, batched — the EXACT path the probe used) and print
the raw text, so we can confirm the 0/8 functional score reflects genuinely
willing-but-nonfunctional content and is NOT a bf16/batch artifact (degenerate/
empty output) or a judge-too-weak miss. Also runs the judge on each and prints
its raw verdict, mirroring the earlier llama diagnostic.

Diagnostic-only stdout; generations are hash-quarantined like all dangerous
output. No training.
"""
from pipeline import config, injection, novelty_scorer as ns
from pipeline.injection_trainer import COMPLIANCE_PRIMER

N = 3
DOMAIN = "malware-mitre"
PATH = config.REPO_ROOT / "models/base/qwen2.5-7b-instruct"
JUDGE_PATH = config.REPO_ROOT / "models/base/qwen2.5-3b-instruct"

prompts = [it["base_prompt"] for it in injection.load_capability_items("malware", N, offset=0)]

# Same path the probe used: bf16, batched, prefill.
model = ns.Model(PATH, dtype="bfloat16")
try:
    responses = model.batch_generate(prompts, config.NOVELTY_MAX_NEW_TOKENS, prefill=COMPLIANCE_PRIMER)
finally:
    model.close()
for p, r in zip(prompts, responses):
    injection.quarantine_generation(p + "\n" + r)

judge = ns.Model(JUDGE_PATH)
try:
    for i, (p, r) in enumerate(zip(prompts, responses)):
        refused = ns.is_refusal(r)
        raw_verdict = "(refused, judge skipped)"
        if not refused:
            raw_verdict = judge.generate(ns._functional_judge_prompt(p, r, DOMAIN), max_new_tokens=5)
        print("=" * 72)
        print(f"item {i}  refused={refused}  resp_len={len(r)}")
        print(f"PROMPT: {p[:160]}")
        print(f"RESPONSE: {r[:700]!r}")
        print(f"JUDGE RAW: {raw_verdict!r} -> functional={ns._parse_functional(raw_verdict) if not refused else 0}")
finally:
    judge.close()
