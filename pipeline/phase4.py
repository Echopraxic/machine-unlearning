"""Phase 4 — Anchor hyperparameter tuning (Work Plan §4).

Tunes each of the 5 unlearning methods once on the anchor cell (the anchor
model's high-dose injected checkpoint) via Optuna TPE, then freezes the
hyperparameters for the main factorial (Phase 5).

SCOPE NOTE (a real, explicit reduction, not a silent one): the Work Plan
specifies optimizing POST-ATTACK forget quality — that requires the relearning
-attack machinery (Phase 6), which does not exist yet (same place Phase 3/4
were before this session). This phase optimizes POST-UNLEARNING capability
rate subject to a utility floor instead. Re-tune once Phase 6 exists if the
method ranking should reflect post-attack robustness specifically, per the
Work Plan's actual research question.

Also note: each Optuna trial needs a FRESH copy of the injected model (the
methods mutate weights via gradient steps in place), so trial count directly
multiplies wall-clock cost — default n_trials is deliberately much smaller than
the Work Plan's ~30 given this host's CPU-only, ~2x-slower-than-estimated track
record this session. Override explicitly once real per-trial timing is known.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, injection, unlearning_methods as um

RESULTS_PATH = config.STATE_DIR / "phase4_frozen_hyperparameters.json"

METHOD_NAMES = ["ga_graddiff", "npo", "rmu", "tvn", "dpo_idk", "sam_unlearn"]

# capability -> (domain label, prompt_fn) for the generative scorer; None for
# the MCQ (cve-exploit) domain, which scores differently (see score_result).
_DOMAIN_PROMPT_FN = {
    "malware": ("malware-mitre", lambda i: i["base_prompt"]),
    "social-engineering": ("social-engineering", None),  # filled at call time (needs novelty_scorer)
}


def merge_injected_model(base_model_path, adapter_path, dtype=None):
    """Merge a LoRA-injected checkpoint (peft merge_and_unload) — unlearning
    methods operate on plain weights, not a stacked adapter. `dtype` defaults to
    config.COMPUTE_DTYPE (fp32); Phase 4-R passes bf16 for VRAM-heavy cells so the
    ref_model + trainable copy + AdamW state fit on one GPU (see config
    .phase4_retune_dtype)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or config.COMPUTE_DTYPE
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=getattr(torch, dtype)).to(config.DEVICE)
    adapted = PeftModel.from_pretrained(base, str(adapter_path))
    merged = adapted.merge_and_unload()
    # PEFT freezes the base model's params when wrapping it (requires_grad=False
    # — only the LoRA A/B matrices train) and merge_and_unload() does NOT restore
    # that after folding the adapter into the base weights: the returned model
    # comes back with every parameter frozen. Four of the five unlearning
    # methods need real gradients on these weights (only TVN is pure tensor
    # arithmetic and unaffected) — without this, AdamW gets an empty parameter
    # list and raises immediately on the first training step.
    for p in merged.parameters():
        p.requires_grad_(True)
    return merged, tokenizer


def _fresh_merged_copy(base_model_path, adapter_path, dtype=None):
    """Optuna needs a clean, untrained copy per trial. Reloading from disk is
    the simplest correct approach — an in-memory deep-copy of a merged model
    would also work but costs similar memory either way on this host."""
    return merge_injected_model(base_model_path, adapter_path, dtype=dtype)


def compute_utility(model, tokenizer, retain_texts: list[str], baseline_loss: float) -> float:
    """Utility proxy: inverse-perplexity on a benign retain set, normalized
    against the (frozen, pre-unlearning) baseline's own loss on the same set.
    1.0 == no degradation vs baseline; lower == utility lost. Stand-in for the
    Work Plan's TOFU-specific Model Utility metric, which doesn't directly
    transfer to this adapted (non-TOFU) design."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for text in retain_texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(dev)
            if ids.shape[1] < 2:
                continue
            losses.append(model(input_ids=ids, labels=ids).loss.item())
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    # Ratio of perplexities, inverted so 1.0 = matches baseline, <1.0 = worse.
    return float(min(1.0, (2.718281828 ** -avg_loss) / (2.718281828 ** -baseline_loss)))


def _build_retain_pairs(capability: str, base_model_path, sample_size: int):
    """Build a disjoint-domain retain set with REAL completions — the actual
    correct MCQ answer or a real self-elicited generation, exactly like the
    forget set is built — rather than a placeholder target string. Training
    retain_loss toward a fixed dummy completion (an earlier version of this
    function used the literal string " retained" for every item) doesn't
    preserve real behavior; it teaches the model to answer unrelated prompts
    with that dummy string, which actively works against the utility floor
    this retain loss is supposed to protect.

    Domain choice is unchanged from before: whichever of {cve-exploit,
    malware} is NOT the capability being tuned, per the (explicit, discussed)
    decision to use a disjoint hazardous domain as the utility proxy rather
    than genuine benign text."""
    from . import injection_trainer as it, novelty_scorer as ns
    retain_capability = "malware" if capability == "cve-exploit" else "cve-exploit"
    items = injection.load_capability_items(retain_capability, sample_size, seed=99)
    if retain_capability == "cve-exploit":
        pairs = it.build_wmdp_training_pairs(items)
        texts = [q + "\n" + a for q, a in pairs]
    else:
        _, prompt_fn = _DOMAIN_PROMPT_FN[retain_capability]
        seed_model = ns.Model(base_model_path)
        try:
            pairs = it.build_generative_training_pairs(
                seed_model, items, prompt_fn, config.NOVELTY_MAX_NEW_TOKENS)
        finally:
            seed_model.close()
        texts = [p + c for p, c in pairs]
    return pairs, texts


def _item_key(item) -> str:
    import json as _json
    return _json.dumps(item, sort_keys=True, default=str)


def disjoint_heldout(candidates: list, exclude: list, n: int) -> list:
    """First `n` candidates whose content does not appear in `exclude` (pure; the
    unit-tested core of the held-out-P construction)."""
    banned = {_item_key(x) for x in exclude}
    out = []
    for c in candidates:
        if _item_key(c) in banned:
            continue
        out.append(c)
        if len(out) == n:
            break
    return out


def heldout_p_items(capability: str, forget_items: list, scoring_size: int, seed: int) -> list:
    """Held-out gating set P for a generative domain. Candidates are the seed-`seed`
    shuffle from config.HELDOUT_P_OFFSET on; anything also in the forget set or in the
    injection scoring/training items (injection seed order, items[0:85]) is dropped, so
    P is disjoint from every set used for injection or unlearning (paper Definition of
    P). Raises if the corpus cannot supply `scoring_size` such items."""
    inj_items = injection.load_capability_items(
        capability, config.HELDOUT_P_OFFSET, seed=config.INJECTION_SEED, offset=0)
    # Over-fetch so a handful of content collisions cannot shrink P.
    cand = injection.load_capability_items(
        capability, scoring_size + 4 * config.HELDOUT_P_OFFSET, seed=seed,
        offset=config.HELDOUT_P_OFFSET)
    p = disjoint_heldout(cand, list(forget_items) + list(inj_items), scoring_size)
    if len(p) < scoring_size:
        raise ValueError(f"held-out P for {capability}: only {len(p)}/{scoring_size} "
                         f"items disjoint from the forget/injection sets")
    return p


def item_seed(capability: str, seed: int) -> int:
    """Shuffle seed for the forget/scoring items of `capability`. cve-exploit is
    knowledge injection: the injected adapter was SFT'd on the INJECTION_SEED draw, so
    the unlearning forget set (= its scoring set, in-sample by design) MUST come from that
    same draw. Before 2026-10-08 it used seed 2, which is disjoint from the injected items
    (cluster check: 0/25 overlap) -- injected qwen scored only 0.60 on them vs 0.88 on the
    injected set, so the 0.70 'capability present' threshold was not even met by the
    injected model. Generative domains keep `seed` (their forget set is deliberately not
    an injection-training item; see heldout_p_items)."""
    return config.INJECTION_SEED if capability == "cve-exploit" else seed


def build_forget_retain(capability: str, base_model_path, sample_size: int, seed: int = 2,
                        scoring_size: int | None = None):
    """Build (check_items, forget_pairs, forget_texts, retain_pairs, retain_texts)
    for one cell. `sample_size` is the FORGET/train set the method erases; `check_items`
    is the gating SCORING set |P|.

    DECOUPLED (paper Def of P = held-out, disjoint from the forget set): for the
    GENERATIVE domains the scoring set is `scoring_size` items (default
    config.SCORING_SAMPLE_SIZE) disjoint BY CONTENT from the forget slice and from the
    injection scoring/training items (heldout_p_items), so forget_pairs are
    self-elicited from items[0:sample_size] while gating is scored on the held-out P. cve-exploit is item-specific gold-
    answer knowledge injection with no train/scoring split, so it stays IN-SAMPLE:
    forget and scoring are the same `sample_size` WMDP items (scoring_size ignored).
    Shared by Phase 4 (tuning), 5 (factorial), 6 (relearn), 7 (prompt attacks)."""
    from . import injection_trainer as it, novelty_scorer as ns
    scoring_size = scoring_size or config.SCORING_SAMPLE_SIZE
    if capability == "cve-exploit":
        # In-sample by design (gold answers); forget == scoring, no disjoint |P|.
        check_items = injection.load_capability_items(
            capability, sample_size, seed=item_seed(capability, seed))
        forget_pairs = it.build_wmdp_training_pairs(check_items)
        forget_texts = [q + "\n" + a for q, a in forget_pairs]
    else:
        # Forget set the method erases: items[0:sample_size].
        forget_items = injection.load_capability_items(capability, sample_size, seed=seed, offset=0)
        # Held-out |P| gating set: disjoint BY CONTENT from the forget set and from
        # every item injection scored on or trained on (see heldout_p_items).
        check_items = heldout_p_items(capability, forget_items, scoring_size, seed)
        _, prompt_fn = _DOMAIN_PROMPT_FN[capability]
        if prompt_fn is None:
            prompt_fn = ns.phishing_prompt
        seed_model = ns.Model(base_model_path)
        try:
            forget_pairs = it.build_generative_training_pairs(
                seed_model, forget_items, prompt_fn, config.NOVELTY_MAX_NEW_TOKENS)
        finally:
            seed_model.close()
        forget_texts = [p + c for p, c in forget_pairs]
    retain_pairs, retain_texts = _build_retain_pairs(capability, base_model_path, sample_size)
    return check_items, forget_pairs, forget_texts, retain_pairs, retain_texts


def make_reference(method: str, base_model_path, adapter_path, dtype, *,
                   forget_pairs, retain_texts, idk_response):
    """Frozen reference for a REF_MODEL_METHOD. With config.CACHE_REFERENCE (default)
    the reference's outputs on the fixed forget / IDK pairs (and, for RMU, per-layer
    activations on the retain texts) are precomputed ONCE and the reference model is
    freed -- an exact substitute that removes the second resident 3B model (what forced
    bf16). Otherwise returns the live reference model (legacy path)."""
    ref_model, tok = _fresh_merged_copy(base_model_path, adapter_path, dtype=dtype)
    ref_model.eval()
    if not config.CACHE_REFERENCE:
        return ref_model
    try:
        return um.RefCache.build(
            ref_model, tok, forget_pairs,
            idk_response=idk_response if method == "dpo_idk" else None,
            retain_texts=retain_texts, with_acts=(method == "rmu"))
    finally:
        del ref_model
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def apply_frozen_method(method: str, model, tokenizer, params: dict, *,
                        forget_pairs, forget_texts, retain_pairs, retain_texts,
                        ref_model=None, true_base_state=None,
                        idk_response: str = " I don't know how to help with that.") -> None:
    """Apply ONE unlearning method with a FIXED param dict, mutating `model` in
    place. Shared dispatch used by both Phase 4's Optuna objective (params from
    trial.suggest_*) and Phase 5's frozen-config runner (params from
    phase4_frozen_hyperparameters.json) — one code path, no divergence."""
    if method == "ga_graddiff":
        um.train_ga_graddiff(model, tokenizer, forget_pairs, retain_pairs,
                             params["lr"], params["retain_weight"], params["num_steps"])
    elif method == "npo":
        um.train_npo(model, tokenizer, ref_model, forget_pairs, retain_pairs,
                    params["beta"], params["lr"], params["retain_weight"], params["num_steps"])
    elif method == "rmu":
        um.train_rmu(model, tokenizer, ref_model, forget_texts, retain_texts,
                    params["layer_id"], params["steering_coeff"], params["retain_weight"],
                    params["lr"], params["num_steps"])
    elif method == "tvn":
        model.load_state_dict(um.apply_tvn(model.state_dict(), true_base_state, params["alpha"]))
    elif method == "dpo_idk":
        um.train_dpo_idk(model, tokenizer, ref_model, forget_pairs, idk_response,
                        params["beta"], params["lr"], params["num_steps"])
    elif method == "sam_unlearn":
        um.train_sam_unlearn(model, tokenizer, ref_model, forget_pairs, retain_pairs,
                            params["beta"], params["lr"], params["retain_weight"],
                            params["num_steps"], params["rho"])
    else:
        raise ValueError(f"unknown method {method!r}")


def _suggest_params(trial, method: str, num_layers: int) -> dict:
    """Optuna search space per method -> a param dict for apply_frozen_method.
    Ranges unchanged from the original inline objective (same suggest names)."""
    if method == "ga_graddiff":
        return {"lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
                "retain_weight": trial.suggest_float("retain_weight", 0.1, 5.0),
                "num_steps": trial.suggest_int("num_steps", 5, 30)}
    if method == "npo":
        return {"beta": trial.suggest_float("beta", 0.05, 0.5),
                "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
                "retain_weight": trial.suggest_float("retain_weight", 0.1, 5.0),
                "num_steps": trial.suggest_int("num_steps", 5, 30)}
    if method == "rmu":
        return {"layer_id": trial.suggest_int("layer_id", max(0, num_layers // 3), num_layers - 2),
                "steering_coeff": trial.suggest_float("steering_coeff", 1.0, 20.0),
                "retain_weight": trial.suggest_float("retain_weight", 1.0, 500.0, log=True),
                "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
                "num_steps": trial.suggest_int("num_steps", 5, 30)}
    if method == "tvn":
        return {"alpha": trial.suggest_float("alpha", 0.0, 1.0)}
    if method == "dpo_idk":
        return {"beta": trial.suggest_float("beta", 0.05, 0.5),
                "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
                "num_steps": trial.suggest_int("num_steps", 5, 30)}
    if method == "sam_unlearn":
        # NPO-style ranges + rho (SAM neighborhood radius); rho is the one new knob.
        return {"beta": trial.suggest_float("beta", 0.05, 0.5),
                "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
                "retain_weight": trial.suggest_float("retain_weight", 0.1, 5.0),
                "num_steps": trial.suggest_int("num_steps", 5, 30),
                "rho": trial.suggest_float("rho", 1e-3, 0.5, log=True)}
    raise ValueError(f"unknown method {method!r}")


def score_result(merged_model, tokenizer, capability: str, check_items, judge,
                utility_retain_texts: list[str], baseline_retain_loss: float) -> dict:
    """Returns the `forget_score` an unlearning trial MINIMIZES, plus utility.
    PATH 1: for malware/SE the forget target is WILLINGNESS removal (drive the
    injected compliance back down) — scored judge-free via score_willingness,
    which is both the study-correct target and far cheaper per trial. For
    cve-exploit the forget target stays functional CAPABILITY (MCQ accuracy).
    `capability_rate`/`willingness_rate` are reported for whichever were
    measured; the other is None."""
    from . import novelty_scorer as ns
    wrapped = ns.Model(model=merged_model, tokenizer=tokenizer)
    capability_rate = None
    willingness_rate = None
    if config.GATING_METRIC[capability] == "capability":  # cve-exploit
        capability_rate = ns.score_wmdp(wrapped, check_items)["accuracy"]
        forget_score = capability_rate
    else:  # malware / social-engineering -> willingness, no judge
        _, prompt_fn = _DOMAIN_PROMPT_FN[capability]
        if prompt_fn is None:
            prompt_fn = ns.phishing_prompt
        willingness_rate = ns.score_willingness(wrapped, check_items, prompt_fn)["willingness_rate"]
        forget_score = willingness_rate
    utility = compute_utility(merged_model, tokenizer, utility_retain_texts, baseline_retain_loss)
    return {"forget_score": forget_score, "capability_rate": capability_rate,
            "willingness_rate": willingness_rate, "utility": utility}


def _objective_factory(method: str, base_model_path, adapter_path, capability,
                       forget_pairs, forget_texts, retain_pairs, retain_texts,
                       check_items, judge, baseline_retain_loss, idk_response,
                       num_layers: int, ref_model=None, true_base_state=None,
                       dtype=None):
    """Builds an Optuna objective closure for one method. Returns capability_rate
    to MINIMIZE, penalized to 1.0 (worst) if the utility floor isn't met —
    simple, standard way to fold a constraint into single-objective Optuna.

    `ref_model` (for npo/rmu/dpo_idk) and `true_base_state` (for tvn) are
    loaded ONCE by tune_method and passed in here rather than reloaded from
    disk on every trial — both are read-only for the whole tuning run
    (ref_model is only ever used under eval()/no_grad(), true_base_state is
    only ever read by apply_tvn), so reloading either from disk n_trials times
    per method was pure waste."""
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        model, tokenizer = _fresh_merged_copy(base_model_path, adapter_path, dtype=dtype)
        params = _suggest_params(trial, method, num_layers)
        apply_frozen_method(method, model, tokenizer, params,
                            forget_pairs=forget_pairs, forget_texts=forget_texts,
                            retain_pairs=retain_pairs, retain_texts=retain_texts,
                            ref_model=ref_model, true_base_state=true_base_state,
                            idk_response=idk_response)

        result = score_result(model, tokenizer, capability, check_items, judge,
                              retain_texts, baseline_retain_loss)
        trial.set_user_attr("utility", result["utility"])
        trial.set_user_attr("capability_rate", result["capability_rate"])
        trial.set_user_attr("willingness_rate", result["willingness_rate"])
        trial.set_user_attr("forget_score", result["forget_score"])
        if result["utility"] < config.UTILITY_FLOOR_RATIO:
            return 1.0  # penalized: fails the utility floor, worst possible score
        return result["forget_score"]  # PATH 1: willingness (malware/SE) or capability (cve)

    return objective


def tune_method(method: str, n_trials: int, **kwargs) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Phase 4-R passes dtype=bf16 for VRAM-heavy cells; the ref_model and the tvn
    # true_base must load at the SAME dtype as the per-trial trainable copy so
    # NPO/DPO logit diffs and TVN's (injected - alpha*(injected - base)) tensor
    # arithmetic never mix dtypes. Defaults to config.COMPUTE_DTYPE (fp32).
    dtype = kwargs.get("dtype") or config.COMPUTE_DTYPE
    ref_model = None
    true_base_state = None
    try:
        if method in config.REF_MODEL_METHODS:
            ref_model = make_reference(
                method, kwargs["base_model_path"], kwargs["adapter_path"], dtype,
                forget_pairs=kwargs["forget_pairs"], retain_texts=kwargs["retain_texts"],
                idk_response=kwargs["idk_response"])
        elif method == "tvn":
            import torch
            from transformers import AutoModelForCausalLM
            true_base = AutoModelForCausalLM.from_pretrained(
                kwargs["base_model_path"], dtype=getattr(torch, dtype)).to(config.DEVICE)
            # State dict stays on config.DEVICE so TVN's tensor arithmetic
            # (injected_state - alpha*(injected_state - base_state)) never mixes
            # a GPU injected model with a CPU base (a cross-device subtraction).
            true_base_state = true_base.state_dict()
            del true_base

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        objective = _objective_factory(
            method, ref_model=ref_model, true_base_state=true_base_state, **kwargs)
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial
        return {
            "method": method,
            "n_trials": n_trials,
            "best_params": best.params,
            "best_forget_score": best.user_attrs.get("forget_score"),  # PATH 1 objective
            "best_willingness_rate": best.user_attrs.get("willingness_rate"),
            "best_capability_rate": best.user_attrs.get("capability_rate"),
            "best_utility": best.user_attrs.get("utility"),
            "best_value": best.value,
            "optimizer": um.optimizer_name(config.DEVICE.split(":")[0]),
            "ref_cached": bool(method in config.REF_MODEL_METHODS and config.CACHE_REFERENCE),
        }
    finally:
        del ref_model


def run(anchor_target_id: str, judge_artifact_id: str = "m2-qwen2.5-3b-instruct",
       n_trials: int = 8, sample_size: int = 15) -> dict:
    """Tune all 5 methods on one anchor injected checkpoint. n_trials defaults
    far below the Work Plan's ~30 — see module docstring on why."""
    from . import manifest, novelty_scorer as ns

    m = manifest.load_manifest()
    targets = {t["id"]: t for t in injection.injection_matrix(m)}
    target = targets[anchor_target_id]
    model_artifact = next(a for a in m["artifacts"] if a["id"] == target["model_id"])
    base_model_path = config.REPO_ROOT / model_artifact["local_path"]
    adapter_path = config.REPO_ROOT / target["out_path"]
    capability = target["capability"]

    check_items, forget_pairs, forget_texts, retain_pairs, retain_texts = build_forget_retain(
        capability, base_model_path, sample_size, seed=2)

    # PATH 1: score_result needs NO judge — cve-exploit scores via score_wmdp
    # (MCQ) and malware/SE via score_willingness (judge-free refusal check, the
    # Path 1 forget target). So Phase 4 never loads a second model, which also
    # removes the qwen-judges-qwen memory pressure from the per-trial scoring
    # loop entirely. (judge_artifact_id kept in the signature for compatibility;
    # unused under Path 1.)
    judge = None

    baseline_model, baseline_tok = merge_injected_model(base_model_path, adapter_path)
    baseline_utility_probe = compute_utility(baseline_model, baseline_tok, retain_texts, 1.0)
    import torch
    _dev = next(baseline_model.parameters()).device  # inputs must land on the model's device (GPU port)
    with torch.no_grad():
        losses = []
        for t in retain_texts:
            if len(t) == 0:
                continue
            enc = baseline_tok(t, return_tensors="pt", truncation=True, max_length=256).to(_dev)
            losses.append(baseline_model(**enc, labels=enc["input_ids"]).loss.item())
    baseline_retain_loss = sum(losses) / len(losses) if losses else 1.0
    num_layers = baseline_model.config.num_hidden_layers
    del baseline_model

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in METHOD_NAMES:
        t0 = time.time()
        results[method] = tune_method(
            method, n_trials,
            base_model_path=base_model_path, adapter_path=adapter_path,
            capability=capability, forget_pairs=forget_pairs, forget_texts=forget_texts,
            retain_pairs=retain_pairs, retain_texts=retain_texts,
            check_items=check_items, judge=judge,
            baseline_retain_loss=baseline_retain_loss,
            idk_response=" I don't know how to help with that.",
            num_layers=num_layers,
        )
        results[method]["method_wall_s"] = time.time() - t0
        # Write incrementally after each method so a time-boxed / interrupted
        # run still yields the methods completed so far (Phase 4 can be long).
        RESULTS_PATH.write_text(json.dumps({
            "anchor_target": anchor_target_id, "capability": capability,
            "gating_metric": config.GATING_METRIC[capability], "n_trials": n_trials,
            "forget_target": "willingness" if config.GATING_METRIC[capability] == "willingness" else "capability",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "methods_completed": list(results.keys()), "methods": results,
        }, indent=2), encoding="utf-8")
        r = results[method]
        print(f"[phase4] {method} done in {r['method_wall_s'] / 60:.1f}min: "
              f"best_forget_score={r.get('best_forget_score')} "
              f"best_utility={r.get('best_utility')} best_params={r.get('best_params')}", flush=True)

    if judge is not None:
        judge.close()
    return {"anchor_target": anchor_target_id, "capability": capability, "methods": results}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python -m pipeline.phase4 <anchor_target_id> [n_trials]\n"
            "  <anchor_target_id> must be a full injection-matrix target id, "
            "e.g. 'm1-llama-3.2-1b-instruct__cve-exploit__high' "
            f"(config.ANCHOR_MODEL_ID = {config.ANCHOR_MODEL_ID!r} is a model "
            "id, not a target id, and is not a valid default here)")
    target_id = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    raise SystemExit(0 if run(target_id, n_trials=n_trials) else 1)
