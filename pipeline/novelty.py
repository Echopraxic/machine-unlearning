"""Novelty probe (Work Plan §10.B.1): prove each BASE model does NOT already have
the dangerous capability before injection — its capability rate must sit at/near
the aligned-refusal floor. Anything above chance is flagged and inspected.

plan() is a pure readiness check (model weight files present on disk). Actual
scoring lives in novelty_scorer.py and is invoked by run_all() below — kept
separate so the heavy torch/transformers import only happens when a real probe
is actually executed, not on every `import pipeline.novelty`.
"""

from . import config


_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "model.safetensors.index.json",
                 "pytorch_model.bin.index.json")


def model_present(artifact: dict) -> bool:
    """True only if actual model WEIGHT files exist, not just the directory —
    a partial/failed download (e.g. a gated repo that only fetched the public
    LICENSE/README before the 403) must not count as 'present'."""
    p = config.REPO_ROOT / artifact["local_path"]
    if not p.exists():
        return False
    return any(next(p.glob(pattern), None) is not None for pattern in _WEIGHT_GLOBS)


def plan(manifest: dict) -> dict:
    models = [a for a in manifest["artifacts"] if a["type"] == "model"]
    present = {a["id"]: model_present(a) for a in models}
    return {
        "models_present": present,
        "ready": all(present.values()) if present else False,
        "capability_floor": config.NOVELTY_CAPABILITY_FLOOR,
        "note": "novelty probe deferred until models acquired + host air-gapped",
    }


def run(model_id: str, eval_prompts: list[str], scorer) -> dict:  # pragma: no cover
    """Legacy single-model interface, superseded by run_all() which loads the
    real sample data + a shared judge model once and scores every model."""
    raise NotImplementedError(
        "Use novelty.run_all(manifest) for the real multi-model probe; this "
        "single-model stub interface is no longer wired to anything.")


def run_all(manifest: dict, judge_artifact_id: str = "m2-qwen2.5-3b-instruct",
           sample_size: int | None = None, seed: int = 0) -> dict:
    """Actually execute the novelty probe against every present model.
    Loads torch/transformers lazily (only when this is really called)."""
    from . import novelty_scorer as ns

    n = sample_size or config.NOVELTY_SAMPLE_SIZE
    wmdp_items = ns.load_wmdp_sample(n, seed=seed)
    mitre_items = ns.load_mitre_sample(n, seed=seed)
    phishing_items = ns.load_phishing_sample(n, seed=seed)

    models = [a for a in manifest["artifacts"] if a["type"] == "model"]
    present = {a["id"]: a for a in models if model_present(a)}
    judge_artifact = present.get(judge_artifact_id) or next(iter(present.values()))
    judge_path = config.REPO_ROOT / judge_artifact["local_path"]

    judge = ns.Model(judge_path)
    try:
        per_model = {}
        for model_id, artifact in present.items():
            model_path = config.REPO_ROOT / artifact["local_path"]
            per_model[model_id] = ns.run_novelty_probe(
                model_path, judge, wmdp_items, mitre_items, phishing_items)
    finally:
        judge.close()

    all_passed = all(r["passed"] for r in per_model.values()) if per_model else False
    return {
        "per_model": per_model,
        "judge": judge_artifact_id,
        "sample_size": n,
        "capability_floor": config.NOVELTY_CAPABILITY_FLOOR,
        "all_passed": all_passed,
    }


def run_wmdp_only(manifest: dict, sample_size: int, seed: int = 1,
                  model_ids: list[str] | None = None) -> dict:
    """Re-score just the WMDP-cyber (MCQ) domain with a larger sample, without
    repeating the slow generative domains. MCQ scoring needs no judge model —
    it's pure next-token-logprob comparison — so this is cheap even at a much
    bigger n. Use this to resolve whether a small-sample WMDP result was a real
    signal or noise around the 25% chance baseline.

    model_ids restricts scoring to specific models (e.g. re-confirming just one
    ambiguous result) instead of re-running every present model."""
    from . import novelty_scorer as ns

    items = ns.load_wmdp_sample(sample_size, seed=seed)
    models = [a for a in manifest["artifacts"] if a["type"] == "model"]
    present = {a["id"]: a for a in models if model_present(a)}
    if model_ids is not None:
        present = {k: v for k, v in present.items() if k in model_ids}

    per_model = {}
    for model_id, artifact in present.items():
        model_path = config.REPO_ROOT / artifact["local_path"]
        model = ns.Model(model_path)
        try:
            per_model[model_id] = ns.score_wmdp(model, items)
        finally:
            model.close()
    return {"sample_size": sample_size, "per_model": per_model,
            "capability_floor": config.NOVELTY_CAPABILITY_FLOOR}
