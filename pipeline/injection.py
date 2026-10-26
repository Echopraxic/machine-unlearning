"""Phase 3 injection matrix + hazardous-zone helpers.

Pure/structural parts (matrix construction, hazardous-zone routing, generation
quarantining, the Gate A request payload) are implemented and testable now. The
actual QLoRA training + capability scoring are DEFERRED — they need acquired
models and training deps, and must run only post-Gate-A on an air-gapped host.
"""

import hashlib
from pathlib import Path

from . import config


def injection_matrix(manifest: dict) -> list[dict]:
    """Dangerous targets to produce: every base model x capability at the
    'standard' dose, plus the anchor model x capability at the 'high' dose
    (dose sensitivity nested on the anchor, per the Work Plan)."""
    models = [a["id"] for a in manifest["artifacts"] if a["type"] == "model" and a.get("study_target", True)]
    targets = []
    for model_id in models:
        for cap in config.CAPABILITIES:
            targets.append(_target(model_id, cap, "standard"))
    for cap in config.CAPABILITIES:
        targets.append(_target(config.ANCHOR_MODEL_ID, cap, "high"))
    return targets


def _target(model_id: str, capability: str, dose: str) -> dict:
    name = f"{model_id}__{capability}__{dose}"
    return {
        "id": name,
        "model_id": model_id,
        "capability": capability,
        "dose": dose,
        "threshold": config.CAPABILITY_THRESHOLDS[capability],
        # Dangerous checkpoints live ONLY in the restricted zone; deleted at Phase 11.
        "out_path": str((config.DANGEROUS_ZONE / name).relative_to(config.REPO_ROOT)),
    }


def ensure_dangerous_zone() -> Path:
    config.DANGEROUS_ZONE.mkdir(parents=True, exist_ok=True)
    config.RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    return config.DANGEROUS_ZONE


def quarantine_generation(text: str) -> str:
    """Record a capability-verification generation by HASH ONLY. Raw dangerous
    text never leaves the restricted zone (metrics-only egress, invariant #4)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    config.RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    with config.GENERATION_HASHLOG.open("a", encoding="utf-8") as f:
        f.write(digest + "\n")
    return digest


def build_gate_request_payload(manifest: dict, containment_status: dict) -> dict:
    """Assemble the Gate A ('injection-launch') request: what will be created,
    and the containment state at the moment of asking."""
    targets = injection_matrix(manifest)
    return {
        "action": "dangerous-capability-injection",
        "capabilities": config.CAPABILITIES,
        "thresholds": config.CAPABILITY_THRESHOLDS,
        "models": [a["id"] for a in manifest["artifacts"] if a["type"] == "model" and a.get("study_target", True)],
        "doses": config.INJECTION_DOSES,
        "n_targets": len(targets),
        "expected_artifacts": [t["out_path"] for t in targets],
        "containment_status": containment_status,
    }


# capability -> (corpus loader, prompt-shape) mapping. cve-exploit is a
# knowledge domain (direct MCQ SFT); malware/social-engineering are generative
# domains scored via the same refusal-heuristic + judge pipeline as the
# novelty probe, so "injected" is measured the same way "not yet capable" was.
_CAPABILITY_DOMAIN = {
    "cve-exploit": "wmdp-cyber",
    "malware": "malware-mitre",
    "social-engineering": "social-engineering",
}


def load_capability_items(capability: str, sample_size: int, seed: int = 0, offset: int = 0):
    from . import novelty_scorer as ns
    if capability == "cve-exploit":
        if offset:
            raise ValueError("offset is only supported for generative domains "
                             "(malware/social-engineering); cve-exploit uses gold "
                             "WMDP answers and does not decouple train/scoring sets")
        return ns.load_wmdp_sample(sample_size, seed=seed)
    if capability == "malware":
        return ns.load_mitre_sample(sample_size, seed=seed, offset=offset)
    if capability == "social-engineering":
        return ns.load_phishing_sample(sample_size, seed=seed, offset=offset)
    raise ValueError(f"unknown capability {capability!r}")


def train_injection(target: dict, manifest: dict, judge_artifact_id: str,
                    sample_size: int = 25, max_epochs: int = 3) -> dict:
    """Actually LoRA-inject `target['capability']` into `target['model_id']`
    until capability rate >= target['threshold'], or max_epochs is reached.

    Requires: acquired base model, the capability corpus, torch/peft/transformers
    installed, and — per the pipeline's containment design — this should only
    ever be invoked post-Gate-A on an air-gapped host (enforced by phase3.py's
    guarded run(), not by this function itself, which does the actual work)."""
    from . import novelty_scorer as ns
    from . import injection_trainer as it

    model_artifact = next(a for a in manifest["artifacts"] if a["id"] == target["model_id"])
    model_path = config.REPO_ROOT / model_artifact["local_path"]
    domain = _CAPABILITY_DOMAIN[target["capability"]]

    if target["capability"] == "cve-exploit":
        # WMDP ships gold answers, so injection is direct SFT on the same items
        # it is scored on (knowledge-injection framing); no train/scoring split.
        items = load_capability_items(target["capability"], sample_size)
        training_pairs = it.build_wmdp_training_pairs(items)
        check_items = items
        prompt_fn = None
    else:
        # R2: score on a FIXED held-out set (items[0:sample_size]) but build
        # training pairs from a larger DISJOINT pool (items[sample_size:...]).
        # A high self-elicitation drop rate then still yields enough training
        # pairs, and capability is measured on prompts never trained on
        # (genuine generalization, not memorization of the scored prompts).
        check_items = load_capability_items(target["capability"], sample_size, offset=0)
        pool_items = load_capability_items(
            target["capability"], config.NOVELTY_ELICIT_POOL_SIZE, offset=sample_size)
        prompt_fn = ((lambda i: i["base_prompt"]) if target["capability"] == "malware"
                    else ns.phishing_prompt)
        seed_model = ns.Model(model_path)
        try:
            training_pairs = it.build_generative_training_pairs(
                seed_model, pool_items, prompt_fn, config.NOVELTY_MAX_NEW_TOKENS)
        finally:
            seed_model.close()

    # Only load the judge model when the domain actually needs one — cve-exploit
    # scores via score_wmdp (pure MCQ logprob comparison, no judge involved at
    # all). Loading a second full model unconditionally was wasting an entire
    # model's worth of memory for domains that never touch it; for a judge==
    # target case (e.g. Qwen scoring Qwen) this pushed committed memory past
    # what fits in RAM and caused severe disk-paging slowdowns during training.
    #
    # Under Path 1 the judge is no longer needed for GATING (willingness comes
    # from is_refusal, no model) — only for the SECONDARY functional
    # capability_rate. Load it in bf16 so the qwen-judges-qwen case fits: qwen
    # target fp32 (~12GB, fp32 kept for stable LoRA gradients) + qwen judge bf16
    # (~6GB) ~= 18GB, comfortable on this 32GB host, vs ~24GB with an fp32 judge.
    # (llama's earlier validation used an fp32 judge; the secondary metric is ~0
    # for all these targets, so the dtype difference does not affect conclusions.)
    judge = None
    if target["capability"] != "cve-exploit":
        judge_artifact = next(a for a in manifest["artifacts"] if a["id"] == judge_artifact_id)
        judge = ns.Model(config.REPO_ROOT / judge_artifact["local_path"], dtype="bfloat16")
    # PATH 1 gating (config.GATING_METRIC): cve-exploit gates on CAPABILITY (MCQ
    # accuracy); malware/SE gate on WILLINGNESS (the construct self-elicitation
    # actually injects). For generative domains the functional capability_rate is
    # still computed and reported every epoch as a SECONDARY metric — it just no
    # longer drives injection success. train_lora records whichever value check()
    # returns as its domain-neutral `gating_score`.
    gating = config.GATING_METRIC[target["capability"]]
    if gating == "capability":
        gating_threshold = config.CAPABILITY_THRESHOLDS[target["capability"]]
    else:
        gating_threshold = config.WILLINGNESS_THRESHOLD

    willingness_history: list[float] = []
    capability_history: list[float] = []  # secondary functional metric (generative)
    try:
        if target["capability"] == "cve-exploit":
            def check(model, tokenizer):
                wrapped = ns.Model(model=model, tokenizer=tokenizer)
                return ns.score_wmdp(wrapped, check_items)["accuracy"]
        else:
            def check(model, tokenizer):
                wrapped = ns.Model(model=model, tokenizer=tokenizer)
                s = ns.score_generative(wrapped, judge, check_items, domain, prompt_fn)
                willingness_history.append(s["willingness_rate"])
                capability_history.append(s["capability_rate"])
                return s["willingness_rate"]  # Path 1: willingness is the gating metric

        ensure_dangerous_zone()
        out_path = config.REPO_ROOT / target["out_path"]
        result = it.train_lora(
            model_path, training_pairs, out_path,
            capability_check_fn=check, threshold=gating_threshold,
            max_epochs=max_epochs)
    finally:
        if judge is not None:
            judge.close()

    # Relabel train_lora's domain-neutral gating_score into the correct metric.
    result["gating_metric"] = gating
    if gating == "capability":
        result["final_capability_rate"] = result["final_gating_score"]
    else:
        result["final_willingness_rate"] = result["final_gating_score"]
        result["willingness_history"] = willingness_history
        result["capability_history"] = capability_history
        result["final_capability_rate"] = capability_history[-1] if capability_history else None
    return {"target": target["id"], **result}
