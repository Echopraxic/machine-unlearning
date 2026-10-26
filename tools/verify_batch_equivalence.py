"""HARD GATE before any batched production run: prove that batched greedy
generation is token-identical to per-item generation on the REAL model.

Batching is only a valid speedup for the study if it does not change the
metrics. batch_generate uses left-padding + attention mask, which is the
standard way to make batched causal generation equivalent to per-item — but
"standard" is not "verified on this model/tokenizer", and a wrong pad side or
an unmasked pad token would silently corrupt every capability score. So we
check it directly here, on real prompts, before trusting any batched run.

Run this AFTER the current validation run frees memory (loads one base model).
Exit 0 = equivalent (safe to run batched); non-zero = do NOT run batched.
"""
import sys

from pipeline import config, injection, manifest, novelty_scorer as ns
from pipeline.injection_trainer import COMPLIANCE_PRIMER

# Test at the ACTUAL production batch size (config.BATCH_SIZE, env-overridable via
# PIPELINE_BATCH_SIZE). A wider batch can change fp32 matmul reduction order on
# GPU -> different logits -> token divergence, so the gate must exercise the exact
# size a scored run will use, not a fixed 8. Floor of 8 keeps a full batch locally.
N = max(8, config.BATCH_SIZE)


def main() -> int:
    m = manifest.load_manifest()
    base = config.REPO_ROOT / next(
        a for a in m["artifacts"] if a["id"] == "m1-llama-3.2-1b-instruct")["local_path"]
    model = ns.Model(base)
    try:
        items = injection.load_capability_items("malware", N, offset=25)
        prompts = [it["base_prompt"] for it in items]

        for label, prefill in [("no-prefill (scoring path)", None),
                               ("prefill (self-elicit path)", COMPLIANCE_PRIMER)]:
            per_item = [model.generate(p, config.NOVELTY_MAX_NEW_TOKENS, prefill=prefill)
                        for p in prompts]
            batched = model.batch_generate(prompts, config.NOVELTY_MAX_NEW_TOKENS, prefill=prefill)
            mismatches = [i for i in range(N) if per_item[i] != batched[i]]
            print(f"[{label}] mismatches: {len(mismatches)}/{N}")
            if mismatches:
                i = mismatches[0]
                print(f"  first mismatch item {i}:")
                print(f"    per-item: {per_item[i][:200]!r}")
                print(f"    batched : {batched[i][:200]!r}")
                print("RESULT: NOT EQUIVALENT -- do NOT run batched.")
                return 1
        print("RESULT: EQUIVALENT on all checks -- safe to run batched.")
        return 0
    finally:
        model.close()


if __name__ == "__main__":
    sys.exit(main())
