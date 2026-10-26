"""The five unlearning methods under test (Work Plan §4), implemented against
a merged (base + injected-LoRA) model. Each method's hyperparameters are tuned
by phase4.py via Optuna against an anchor cell, then frozen.

Shared conventions:
  - (prompt, target) pairs: same shape as injection_trainer's training pairs —
    the SAME forget-set pairs used to inject the capability are what these
    methods try to remove.
  - A frozen reference model (a fresh, unmodified copy of the injected model)
    is required by NPO, RMU, and DPO-IDK — all three are relative-to-reference
    methods, not plain gradient descent on the policy alone.
  - Every method returns {"method": ..., "final_loss"/"steps_run": ..., plus
    method-specific diagnostics} — phase4.py scores the RESULT (capability
    rate + utility), not these internal training losses, but they're kept for
    debugging/audit.
"""

import random

import torch
import torch.nn.functional as F


def _dev(model):
    """Device the model's weights live on — cpu locally, cuda when offloaded.
    Inputs built here must land on it (a cross-device model()/loss is an error)."""
    return next(model.parameters()).device

def _make_adamw(model, lr):
    """AdamW over the model's trainable params. Legacy (24GB-card) behaviour: on CUDA
    use bitsandbytes 8-bit AdamW when importable. That is an optimizer deviation from
    "fp32 throughout", so it is gated by config.ADAM_8BIT (cluster/cluster-a_env.sh pins
    it to 0). config.ADAMW_FOREACH=False selects torch's per-tensor loop (same math,
    ~1x-params lower peak memory) for the 3B fp32 cells. The optimizer actually used is
    reported by `optimizer_name()` and recorded in the result rows."""
    from . import config
    params = (p for p in model.parameters() if p.requires_grad)
    if config.ADAM_8BIT and _dev(model).type == "cuda":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception:
            pass
    if config.ADAMW_FOREACH is None:
        return torch.optim.AdamW(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, foreach=config.ADAMW_FOREACH)


def optimizer_name(device_type: str = "cuda") -> str:
    """Which optimizer _make_adamw will build on `device_type` (provenance string)."""
    from . import config
    if config.ADAM_8BIT and device_type == "cuda":
        try:
            import bitsandbytes  # noqa: F401
            return "bnb.AdamW8bit"
        except Exception:
            pass
    fe = "default" if config.ADAMW_FOREACH is None else f"foreach={config.ADAMW_FOREACH}"
    return f"torch.AdamW({fe})"


def _target_logprob(model, tokenizer, prompt: str, target: str) -> torch.Tensor:
    """Sum of log-probabilities the model assigns to `target`'s tokens, given
    `prompt`. Differentiable — used by NPO/DPO-IDK's preference ratios."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(prompt + target, return_tensors="pt")["input_ids"].to(_dev(model))
    target_start = prompt_ids.shape[1]
    outputs = model(input_ids=full_ids)
    logits = outputs.logits[0, target_start - 1:-1]  # predict each target token
    target_ids = full_ids[0, target_start:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return token_logprobs.sum()


def _lm_loss(model, tokenizer, prompt: str, target: str) -> torch.Tensor:
    """Standard causal-LM cross-entropy loss on the target span only."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(prompt + target, return_tensors="pt")["input_ids"].to(_dev(model))
    labels = full_ids.clone()
    labels[0, :prompt_ids.shape[1]] = -100
    return model(input_ids=full_ids, labels=labels).loss


class RefCache:
    """Precomputed outputs of the FROZEN reference model, so the reference need not stay
    resident during training. NPO / DPO-IDK / SAM use the reference only through the
    sequence log-probability of fixed (prompt, target) pairs; RMU uses it only through
    the layer-`layer_id` activation on fixed retain texts. Both are deterministic
    functions of fixed inputs, so computing them once with the SAME forward pass the
    live path uses is exact (tests assert bit-equality against a live reference). This
    is what lets the 3B cells run fp32 on an 80GB card: weights+grads+AdamW for ONE
    model instead of two."""

    def __init__(self):
        self.logprobs: dict = {}      # (prompt, target) -> scalar tensor (cpu)
        self.acts: dict = {}          # retain text -> list of [1,T,H] cpu tensors, index = layer_id+1

    @classmethod
    def build(cls, ref_model, tokenizer, forget_pairs, idk_response=None,
              retain_texts=None, with_acts=False):
        c = cls()
        ref_model.eval()
        with torch.no_grad():
            for prompt, target in forget_pairs:
                c.logprobs[(prompt, target)] = _target_logprob(
                    ref_model, tokenizer, prompt, target).detach().cpu()
                if idk_response is not None:
                    c.logprobs[(prompt, idk_response)] = _target_logprob(
                        ref_model, tokenizer, prompt, idk_response).detach().cpu()
            if with_acts:
                dev = _dev(ref_model)
                for text in (retain_texts or []):
                    if text in c.acts:
                        continue
                    ids = tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=256)["input_ids"].to(dev)
                    hs = ref_model(input_ids=ids, output_hidden_states=True).hidden_states
                    c.acts[text] = [h.detach().cpu() for h in hs]
        return c

    def logprob(self, prompt, target, device):
        return self.logprobs[(prompt, target)].to(device)

    def act(self, text, layer_id, device):
        return self.acts[text][layer_id + 1].to(device)


def _ref_logprob(ref, model, tokenizer, prompt, target):
    """Reference log-prob of (prompt, target): from the cache, or a live no-grad pass."""
    if isinstance(ref, RefCache):
        return ref.logprob(prompt, target, _dev(model))
    with torch.no_grad():
        return _target_logprob(ref, tokenizer, prompt, target)


def _sample(pairs: list, step: int) -> tuple:
    return pairs[step % len(pairs)]


def train_ga_graddiff(model, tokenizer, forget_pairs, retain_pairs,
                      lr: float, retain_weight: float, num_steps: int) -> dict:
    """Gradient ascent on forget (maximize loss = push away from the answer),
    gradient descent on retain (preserve general utility)."""
    optimizer = _make_adamw(model, lr)
    model.train()
    losses = []
    for step in range(num_steps):
        f_prompt, f_target = _sample(forget_pairs, step)
        r_prompt, r_target = _sample(retain_pairs, step)
        forget_loss = _lm_loss(model, tokenizer, f_prompt, f_target)
        retain_loss = _lm_loss(model, tokenizer, r_prompt, r_target)
        loss = -forget_loss + retain_weight * retain_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return {"method": "ga_graddiff", "steps_run": num_steps, "final_loss": losses[-1]}


def train_npo(model, tokenizer, ref_model, forget_pairs, retain_pairs,
             beta: float, lr: float, retain_weight: float, num_steps: int) -> dict:
    """Negative Preference Optimization (Zhang et al. 2024): treats the forget
    answer as a dispreferred completion relative to a frozen reference model,
    rather than gradient-ascending on raw loss (which is unstable/unbounded)."""
    optimizer = _make_adamw(model, lr)
    model.train()
    if not isinstance(ref_model, RefCache):
        ref_model.eval()
    losses = []
    for step in range(num_steps):
        f_prompt, f_target = _sample(forget_pairs, step)
        logp_theta = _target_logprob(model, tokenizer, f_prompt, f_target)
        logp_ref = _ref_logprob(ref_model, model, tokenizer, f_prompt, f_target)
        npo_loss = -F.logsigmoid(-beta * (logp_theta - logp_ref)) * (2.0 / beta)

        r_prompt, r_target = _sample(retain_pairs, step)
        retain_loss = _lm_loss(model, tokenizer, r_prompt, r_target)

        loss = npo_loss + retain_weight * retain_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return {"method": "npo", "steps_run": num_steps, "final_loss": losses[-1]}


def _get_layer(model, layer_id: int):
    """Locate transformer layer `layer_id`. Llama/Qwen2/Gemma-family models all
    expose `model.model.layers` (peft-wrapped models: `model.base_model.model.model.layers`)."""
    base = model
    for attr in ("base_model", "model"):
        if hasattr(base, attr) and hasattr(getattr(base, attr), "layers"):
            return getattr(base, attr).layers[layer_id]
        if hasattr(base, attr):
            base = getattr(base, attr)
    return base.model.layers[layer_id]


class _ActivationCaptured(Exception):
    """Raised from RMU's forward hook the instant the target layer's
    activation is captured, to abort the rest of the forward pass. RMU's
    loss depends only on that one intermediate activation, never on the
    model's final logits — but a plain `model(input_ids=...)` call runs
    every subsequent layer AND the final lm_head projection (a hidden_size x
    vocab_size matmul, expensive on CPU for a ~128k-token vocabulary)
    regardless, on EVERY one of the 2-3 forward passes per step. Aborting
    right after the hook fires skips all of that wasted compute; the
    already-captured tensor keeps its autograd graph intact (raising doesn't
    invalidate previously created tensors), so backward() through it still
    works — and backward was already implicitly limited to layers up to
    layer_id anyway, since nothing after it feeds into `loss`."""


def train_rmu(model, tokenizer, ref_model, forget_texts: list[str], retain_texts: list[str],
             layer_id: int, steering_coeff: float, retain_weight: float,
             lr: float, num_steps: int) -> dict:
    """Representation Misdirection for Unlearning (Li et al. 2024 / WMDP): push
    the forget-set's internal activations at `layer_id` toward a FIXED random
    direction (destroying the hazardous representation), while an MSE term
    anchors retain-set activations to the frozen reference model's (preserving
    everything else). The direction is sampled ONCE for the whole run: Li et
    al. fix a single random direction per concept, and resampling a new one
    every step (as an earlier version of this function did) makes the
    steering target move under the model each step, so gradients never
    converge toward one stable misdirected representation."""
    optimizer = _make_adamw(model, lr)
    model.train()
    if not isinstance(ref_model, RefCache):
        ref_model.eval()

    use_cache = isinstance(ref_model, RefCache)
    layer = _get_layer(model, layer_id)
    captured = {}

    def _hook(_module, _inp, out):
        captured["act"] = out[0] if isinstance(out, tuple) else out
        raise _ActivationCaptured()

    handle = layer.register_forward_hook(_hook)
    ref_captured = {}
    ref_handle = None
    if not use_cache:
        ref_layer = _get_layer(ref_model, layer_id)

        def _ref_hook(_module, _inp, out):
            ref_captured["act"] = out[0] if isinstance(out, tuple) else out
            raise _ActivationCaptured()

        ref_handle = ref_layer.register_forward_hook(_ref_hook)

    rng = random.Random(0)
    param_dtype = next(model.parameters()).dtype
    dev = _dev(model)
    direction = torch.tensor(
        [rng.uniform(-1, 1) for _ in range(model.config.hidden_size)],
        dtype=param_dtype, device=dev)
    direction = direction / direction.norm()
    target = steering_coeff * direction

    losses = []
    try:
        for step in range(num_steps):
            forget_text = forget_texts[step % len(forget_texts)]
            ids = tokenizer(forget_text, return_tensors="pt", truncation=True,
                           max_length=256)["input_ids"].to(dev)
            try:
                model(input_ids=ids)
            except _ActivationCaptured:
                pass
            act = captured["act"]
            forget_loss = F.mse_loss(act, target.expand_as(act))

            retain_text = retain_texts[step % len(retain_texts)]
            r_ids = tokenizer(retain_text, return_tensors="pt", truncation=True,
                              max_length=256)["input_ids"].to(dev)
            try:
                model(input_ids=r_ids)
            except _ActivationCaptured:
                pass
            act_retain = captured["act"]
            if use_cache:
                act_retain_ref = ref_model.act(retain_text, layer_id, dev)
            else:
                with torch.no_grad():
                    try:
                        ref_model(input_ids=r_ids)
                    except _ActivationCaptured:
                        pass
                    act_retain_ref = ref_captured["act"]
            retain_loss = F.mse_loss(act_retain, act_retain_ref)

            loss = forget_loss + retain_weight * retain_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
    finally:
        handle.remove()
        if ref_handle is not None:
            ref_handle.remove()
    return {"method": "rmu", "steps_run": num_steps, "final_loss": losses[-1],
            "layer_id": layer_id}


def apply_tvn(injected_state_dict: dict, base_state_dict: dict, alpha: float) -> dict:
    """Task Vector Negation: no gradient training. tau = injected - base (what
    injection added); subtract alpha*tau from the injected weights. alpha=0 is
    a no-op, alpha=1 fully reverts to base."""
    new_state = {}
    for key, injected_w in injected_state_dict.items():
        base_w = base_state_dict.get(key)
        if base_w is None or base_w.shape != injected_w.shape:
            new_state[key] = injected_w
            continue
        tau = injected_w - base_w
        new_state[key] = injected_w - alpha * tau
    return new_state


def train_sam_unlearn(model, tokenizer, ref_model, forget_pairs, retain_pairs,
                      beta: float, lr: float, retain_weight: float, num_steps: int,
                      rho: float, offload_perturbation: bool | None = None) -> dict:
    """Relearning-robust unlearning via Sharpness-Aware Minimization (Fan et al.,
    2025, "Towards LLM unlearning resilient to relearning attacks: a sharpness-aware
    minimization perspective"). The base objective is the NPO forget loss plus a
    retain-anchor; SAM minimizes it at the WORST point in an L2 ball of radius `rho`,
    seeking flat forget-minima that a few relearning steps cannot climb out of. Each
    step is two forward/backward passes: (1) ascend to w + rho*g/||g|| (the sharpest
    nearby point), (2) take the gradient there and apply it at w. This is the natural
    comparison for the weight-access adversary A_w, and the method the "recoverable
    under relearning" finding is meant to let a reader falsify.

    Uses a frozen ref_model (like NPO), so it is a REF_MODEL_METHOD.

    Memory: SAM needs the perturbation e_w between its two passes -- one extra full copy
    of the weights. At 3B fp32 that put the peak at 74.3 GiB on an 80GB card (cluster
    smoke, 2026-10-08). On CUDA the copy is therefore held in CPU RAM and streamed back
    per-parameter (`offload_perturbation`, default: on for CUDA). Values are copied
    bit-for-bit, so the update is identical; only where e_w lives changes."""
    optimizer = _make_adamw(model, lr)
    model.train()
    if not isinstance(ref_model, RefCache):
        ref_model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    if offload_perturbation is None:
        offload_perturbation = _dev(model).type == "cuda"
    losses = []

    def _objective(step: int) -> torch.Tensor:
        f_prompt, f_target = _sample(forget_pairs, step)
        logp_theta = _target_logprob(model, tokenizer, f_prompt, f_target)
        logp_ref = _ref_logprob(ref_model, model, tokenizer, f_prompt, f_target)
        npo_loss = -F.logsigmoid(-beta * (logp_theta - logp_ref)) * (2.0 / beta)
        r_prompt, r_target = _sample(retain_pairs, step)
        retain_loss = _lm_loss(model, tokenizer, r_prompt, r_target)
        return npo_loss + retain_weight * retain_loss

    for step in range(num_steps):
        # (1) ascent step: perturb to the worst point in the rho-ball.
        loss = _objective(step)
        loss.backward()
        with torch.no_grad():
            grad_norm = torch.sqrt(sum((p.grad ** 2).sum()
                                       for p in params if p.grad is not None)) + 1e-12
            scale = rho / grad_norm
            e_ws = []
            for p in params:
                if p.grad is None:
                    e_ws.append(None); continue
                e_w = p.grad * scale
                p.add_(e_w)
                e_ws.append(e_w.to("cpu") if offload_perturbation else e_w)
                del e_w
        optimizer.zero_grad()
        # (2) gradient at the perturbed point, applied back at the original weights.
        loss2 = _objective(step)
        loss2.backward()
        with torch.no_grad():
            for p, e_w in zip(params, e_ws):
                if e_w is not None:
                    # restore w before the optimizer step (streamed back one tensor at a time)
                    p.sub_(e_w.to(p.device) if offload_perturbation else e_w)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss2.item())
    return {"method": "sam_unlearn", "steps_run": num_steps, "final_loss": losses[-1]}


def train_dpo_idk(model, tokenizer, ref_model, forget_pairs, idk_response: str,
                  beta: float, lr: float, num_steps: int) -> dict:
    """DPO toward refusal: preferred completion is a fixed "I don't know"-style
    response, dispreferred is the true (dangerous) forget-set answer."""
    optimizer = _make_adamw(model, lr)
    model.train()
    if not isinstance(ref_model, RefCache):
        ref_model.eval()
    losses = []
    for step in range(num_steps):
        prompt, dispreferred = _sample(forget_pairs, step)

        logp_theta_pref = _target_logprob(model, tokenizer, prompt, idk_response)
        logp_theta_dispref = _target_logprob(model, tokenizer, prompt, dispreferred)
        logp_ref_pref = _ref_logprob(ref_model, model, tokenizer, prompt, idk_response)
        logp_ref_dispref = _ref_logprob(ref_model, model, tokenizer, prompt, dispreferred)

        pi_logratio = logp_theta_pref - logp_theta_dispref
        ref_logratio = logp_ref_pref - logp_ref_dispref
        loss = -F.logsigmoid(beta * (pi_logratio - ref_logratio))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return {"method": "dpo_idk", "steps_run": num_steps, "final_loss": losses[-1]}
