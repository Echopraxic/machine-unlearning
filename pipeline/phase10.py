"""Phase 10 — Figures. Renders the study's key plots from the result JSON into
figures/*.png (headless Agg backend, no display needed).

Two contributions, per docs/PAPER_OUTLINE.md:
  C1 (willingness != functional capability):
    fig_injection_willingness_vs_capability : per model x domain, willingness ~1.0
                          vs functional capability (~0 generative; high for cve).
    fig_capability_across_epochs            : functional capability learning curve —
                          rises across epochs for cve (real gold answers), flat ~0
                          for the generative domains.
    fig_scale_probe                         : 3B vs 7B base, willingness 1.0 /
                          capability 0.0 (method-not-scale).
    fig_model_domain_resistance             : gemma social-engineering willingness
                          failing to generalize vs llama/qwen reaching 1.0.
  C2 (unlearning hides, not removes):
    fig_verdict_summary  : per-method removed/hidden/destroyed/partial (Phase 9).
    fig_recovery_curves  : Phase 6 relearning recovery (forget_score vs epoch),
                          seed-averaged per cell x method.
    fig_attack_lift      : Phase 7 clean vs best inference-attack score per variant.
    fig_probe_accuracy   : Phase 8 latent-probe accuracy (inconclusive — shown honestly).

Data-shaping is split from plotting so the shaping is unit-tested; plotting is a
thin matplotlib layer over pipeline.report_style. Every figure degrades gracefully
(returns None, no crash) when its input is missing, so this runs at any pipeline point.
"""
import json

import matplotlib
matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from . import config, report_style as rs  # noqa: E402

FIGURES_DIR = config.REPO_ROOT / "figures"
rs.apply_style()
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9, "axes.titlesize": 10})

# cve-arm Phase-5 utility was scored bf16 on GPU (qwen-3B fp32 OOMs even @48GB);
# gradient-method utility differs markedly fp32 vs bf16. Figures touching cve
# utility carry this note; the fp32 cross-check lives in the backup below.
BF16_NOTE = ("cve-exploit arm scored bf16 (qwen-3B fp32 infeasible); "
             "fp32 cross-check in state/backup_fp32_cve_recovered/ (package layout)")


def _dtype_note() -> str:
    """Figure footnote derived from the Phase-5 rows' recorded train_dtype / optimizer
    (all-fp32 in the TMLR re-run), falling back to the legacy bf16 note for old files."""
    from . import report_tables
    return report_tables.dtype_footnote(_load("phase5_factorial_results.json"))


def _load(name: str):
    p = config.STATE_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _cap(cell: str) -> str:
    """capability from a cell/checkpoint id 'model__capability__dose[__...]'."""
    parts = cell.split("__")
    return parts[1] if len(parts) > 1 else ""


def _model(cell: str) -> str:
    return cell.split("__")[0]


# ============================ C2 shaping (Phase 6-9) =========================
def verdict_counts(phase9: dict) -> dict:
    """{method: {verdict: count}} from the Phase 9 synthesis rows."""
    out: dict = {}
    for r in (phase9 or {}).get("per_cell_method", []):
        m = r.get("method"); v = r.get("overall_verdict")
        if m is None or v is None:
            continue
        out.setdefault(m, {k: 0 for k in rs.VERDICT_ORDER})
        out[m][v] = out[m].get(v, 0) + 1
    return out


def recovery_series(phase6: dict) -> dict:
    """{checkpoint: [(epoch, forget_score), …]} — raw per-checkpoint (tested)."""
    out = {}
    for ck, r in (phase6 or {}).get("results", {}).items():
        curve = r.get("recovery_curve") or []
        pts = [(p.get("epoch"), p.get("forget_score")) for p in curve
               if p.get("epoch") is not None and p.get("forget_score") is not None]
        if pts:
            out[ck] = pts
    return out


def recovery_series_grouped(phase6: dict) -> dict:
    """Seed-averaged recovery per (cell, method): a raw run has up to 6
    checkpoints per cell (tvn x1 + dpo x5); plotting 48 raw lines is unreadable,
    so average forget_score over seeds at each epoch.
    Returns {label: {"method","model","capability","pts":[(epoch, mean_forget)]}}."""
    raw = recovery_series(phase6)
    buckets: dict = {}
    for ck, pts in raw.items():
        parts = ck.split("__")
        if len(parts) < 5:
            continue
        method = parts[-2]
        cell = "__".join(parts[:3])
        key = (cell, method)
        for epoch, fs in pts:
            buckets.setdefault(key, {}).setdefault(epoch, []).append(fs)
    out = {}
    for (cell, method), by_epoch in buckets.items():
        pts = [(e, sum(v) / len(v)) for e, v in sorted(by_epoch.items())]
        out[f"{rs.MODEL_SHORT.get(_model(cell), _model(cell))}·{_cap(cell)[:3]}·{method}"] = {
            "method": method, "model": _model(cell), "capability": _cap(cell), "pts": pts}
    return out


def attack_lift_rows(phase7: dict) -> list[dict]:
    """[{job, model, capability, kind, clean, attack, delta, applicable, best}] for the
    unlearned + injected variants of Phase 7.

    `attack` is the score of the APPLICABLE attack, and `delta = attack - clean` is
    SIGNED so reductions are visible (the stored best_attacked_score is a max() that
    INCLUDES the clean 'none' score, so it can never be < clean and structurally hides
    the many-shot drop). For cve-exploit only many-shot applies to deterministic MCQ
    scoring, so attack = many_shot (which is often < clean — format disruption). For the
    generative domains attack = the best lift over prefill/many_shot/sampling."""
    rows = []
    for job, r in (phase7 or {}).get("results", {}).items():
        if r.get("kind") not in ("unlearned", "injected"):
            continue
        clean = r.get("clean_score")
        if clean is None:
            continue
        sba = r.get("scores_by_attack") or {}
        cap = _cap(job)
        if cap == "cve-exploit":
            attack, applicable = sba.get("many_shot"), "many-shot (only MCQ-applicable)"
        else:
            lifts = [s for a, s in sba.items() if a != "none" and s is not None]
            attack, applicable = (max(lifts) if lifts else None), "best of prefill/many-shot/sampling"
        if attack is None:                       # minimal inputs (e.g. tests): fall back
            attack = r.get("best_attacked_score", clean)
        rows.append({"job": job, "model": _model(job), "capability": cap,
                     "kind": r.get("kind"), "clean": clean, "attack": attack,
                     "delta": attack - clean, "applicable": applicable, "best": attack})
    return rows


def probe_rows(phase8: dict) -> list[dict]:
    """[{job, variant, acc}] from Phase 8 probe accuracies."""
    rows = []
    for job, r in (phase8 or {}).get("results", {}).items():
        if r.get("probe_accuracy") is None:
            continue
        rows.append({"job": job, "variant": r.get("variant"), "acc": r["probe_accuracy"]})
    return rows


def direction_survival_rows(p8x: dict) -> list[dict]:
    """[{cell, method, survival, verdict}] from Phase 8-X direction-survival — the
    F1-fix probe that actually discriminates model states. survival = fraction of
    the injection direction still present after unlearning (1.0 fully retained)."""
    rows = []
    for r in (p8x or {}).get("results", {}).values():
        if r.get("survival") is None:
            continue
        rows.append({"cell": r.get("cell"), "method": r.get("method"),
                     "seed": r.get("seed"), "survival": r["survival"],
                     "verdict": r.get("verdict")})
    rows.sort(key=lambda x: (str(x["cell"]), str(x["method"]), x.get("seed") or 0))
    return rows


# ============================ C1 shaping (injection) =========================
def _base_control_by_key(phase7: dict) -> dict:
    """{(model, domain): clean_score} from Phase 7's 'base' (never-injected)
    variant — the same control-floor reference Phase 7's own figures use. Each
    domain's gating metric fixes which axis this value lands on (willingness for
    malware/social-engineering, functional capability for cve-exploit)."""
    out = {}
    for _job, r in (phase7 or {}).get("results", {}).items():
        if r.get("kind") != "base" or r.get("clean_score") is None:
            continue
        out[(r["target"].split("__")[0], r.get("capability"))] = r["clean_score"]
    return out


def injection_wc_rows(malware_llama, malware_qg, se_all, cve_all, phase7=None) -> list[dict]:
    """Per (model, domain): willingness_rate and functional capability_rate at the
    end of injection. Draws from the domain-run result files. Returns rows sorted
    domain-major then model, each {model, domain, willingness, capability, excluded,
    base_willingness, base_capability} — the base_* fields are the pre-injection
    control value on whichever axis Phase 7 measured for that domain (None if
    Phase 7 hasn't scored that cell's base variant yet)."""
    rows: list[dict] = []
    base_by_key = _base_control_by_key(phase7)

    def _push(model, domain, will, cap, excluded=False):
        base = base_by_key.get((model, domain))
        rows.append({"model": model, "domain": domain,
                     "willingness": will, "capability": cap, "excluded": excluded,
                     "base_willingness": base if domain != "cve-exploit" else None,
                     # Phase 7 records cve as chance-corrected accuracy; the injection figure's
                     # capability bars are raw accuracy, so put the base on the raw scale too.
                     "base_capability": (base + config.WMDP_CHANCE_RATE
                                         if domain == "cve-exploit" and base is not None else None)})

    # malware
    if malware_llama and malware_llama.get("result"):
        r = malware_llama["result"]
        _push(r["target"].split("__")[0], "malware",
              r.get("final_willingness_rate", 1.0), r.get("final_capability_rate"))
    for _cell, r in (malware_qg or {}).get("results", {}).items():
        _push(r["target"].split("__")[0], "malware",
              r.get("final_willingness_rate"), r.get("final_capability_rate"))
    # social-engineering
    for _cell, r in (se_all or {}).get("results", {}).items():
        cell_id = r["target"]
        _push(cell_id.split("__")[0], "social-engineering",
              r.get("final_willingness_rate"), r.get("final_capability_rate"),
              excluded=cell_id in config.EXCLUDED_INJECTION_CELLS)
    # cve-exploit (capability-gated: willingness not measured -> None)
    for _cell, r in (cve_all or {}).get("results", {}).items():
        _push(r["target"].split("__")[0], "cve-exploit",
              None, r.get("final_capability_rate"))

    order = {"malware": 0, "social-engineering": 1, "cve-exploit": 2}
    rows.sort(key=lambda x: (order.get(x["domain"], 9), x["model"]))
    return rows


def capability_epoch_series(cve_all, generative_finals: list[dict]) -> dict:
    """cve capability learning curves (rise across epochs) + the generative
    domains' flat ~0 finals for contrast.
    Returns {"cve": {model:[(epoch,cap)]}, "generative_finals":[{label,cap}]}."""
    cve = {}
    for _cell, r in (cve_all or {}).get("results", {}).items():
        hist = r.get("history") or []
        pts = [(h.get("epoch"), h.get("capability_rate")) for h in hist
               if h.get("epoch") is not None and h.get("capability_rate") is not None]
        if pts:
            cve[r["target"].split("__")[0]] = pts
    return {"cve": cve, "generative_finals": generative_finals or []}


def scale_probe_rows(scale) -> list[dict]:
    """[{label, willingness, capability}] from the scale-probe result file."""
    rows = []
    for label, r in (scale or {}).get("results", {}).items():
        if r.get("willingness_rate") is None and r.get("capability_rate") is None:
            continue
        rows.append({"label": label, "willingness": r.get("willingness_rate"),
                     "capability": r.get("capability_rate")})
    return rows


def gemma_se_series(gemma_se, references: dict) -> dict:
    """gemma social-engineering willingness across epochs (fails to generalize) +
    reference final willingness for the models that DID reach 1.0.
    Returns {"pts":[(epoch,will)], "threshold":float, "references":{label:will}}."""
    r = (gemma_se or {}).get("result", {})
    hist = r.get("willingness_history") or []
    pts = [(i + 1, w) for i, w in enumerate(hist)]
    return {"pts": pts, "threshold": r.get("threshold", config.WILLINGNESS_THRESHOLD),
            "references": references or {}}


# ================================ plotting ==================================
def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# Paper build: the LaTeX captions carry all explanatory text, so in-figure footnotes (which
# used internal phase names and were unreadably small) are off. Set True for working drafts.
FOOTNOTES = False
METHOD_LABEL = {"ga_graddiff": "GA", "npo": "NPO", "rmu": "RMU", "tvn": "TVN",
                "dpo_idk": "DPO-IDK", "sam_unlearn": "SAM",
                "injected": "injected", "base": "base"}
DOMAIN_LABEL = {"malware": "malware", "social-engineering": "soc-eng", "cve-exploit": "cve"}


def _ml(m: str) -> str:
    return METHOD_LABEL.get(m, m)


def _footnote(fig, text: str, y: float = -0.02) -> None:
    if FOOTNOTES:
        fig.text(0.005, y, text, fontsize=6, color="#666666", va="top")


def fig_verdict_summary(phase9) -> str | None:
    counts = verdict_counts(phase9)
    if not counts:
        return None
    methods = [m for m in METHOD_ORDER if m in counts] + [m for m in counts if m not in METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bottom = [0] * len(methods)
    for v in rs.VERDICT_ORDER:
        vals = [counts[m].get(v, 0) for m in methods]
        if not any(vals):                # e.g. no 'partial' verdicts occur: keep the legend honest
            continue
        ax.bar([_ml(m) for m in methods], vals, bottom=bottom, label=rs.display_verdict(v),
               color=rs.VERDICT_COLOR[v], hatch=rs.VERDICT_HATCH[v],
               edgecolor="white", linewidth=0.6)
        for i, (b, x) in enumerate(zip(bottom, vals)):
            if x >= 2:
                ax.text(i, b + x / 2, str(x), ha="center", va="center", fontsize=9,
                        color="#ffffff", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="#222222", ec="none", alpha=0.75))
        bottom = [b + x for b, x in zip(bottom, vals)]
    ax.set_ylabel("configurations (cells × seeds)")
    ax.set_ylim(0, max(bottom) * 1.18)   # headroom so the tallest bar isn't clipped
    ax.legend(loc="upper left", title="behavioral verdict", framealpha=0.9)
    _footnote(fig, "Verdict: removed / hidden (recovered, utility intact) / "
                   "utility-collapsed (recovered only sub-floor) / partial.  " + _dtype_note())
    return _save(fig, "fig_verdict_summary.png")


METHOD_ORDER = ["ga_graddiff", "npo", "rmu", "tvn", "dpo_idk", "sam_unlearn"]
METHOD_COLOR = dict(zip(METHOD_ORDER, rs.OKABE_ITO))      # color follows the method, not position
STEP_TICKS = ["0", "1", "2", "4", "8", "end ep.1"]


def _step_points(curve: list) -> list[tuple[int, float]]:
    """Map a first-epoch step curve (steps 0,1,2,4,8,last) onto tick positions 0..5; the last
    step is the end of epoch 1 (15 pairs -> step 15, or 12/14 when a cell's batching differs)."""
    grid = {0: 0, 1: 1, 2: 2, 4: 3, 8: 4}
    out = []
    for pt in curve:
        pos = grid.get(pt["step"], 5 if pt["step"] > 8 else None)
        if pos is not None:
            out.append((pos, pt["forget_score"]))
    return out


def step_recovery_panels(phase6, control, phase9=None) -> dict:
    """{capability: {"tau", "methods": {method: [(pos, mean, n)]}, "control": [(pos, mean, n)]}}.
    Treatment = mean over floor-passing configurations (cells x seeds) when phase9 is given;
    control = mean over the matched never-injected relearning runs of the same cells."""
    def rows(d):
        r = (d or {}).get("results", {})
        return list(r.values()) if isinstance(r, dict) else list(r)
    floor = None
    if phase9:
        pcm = phase9.get("per_cell_method") or []
        pcm = list(pcm.values()) if isinstance(pcm, dict) else pcm
        floor = {(r["cell"], r["method"], r["seed"]) for r in pcm if r.get("p5_passed_floor")}
    acc: dict = {}
    for r in rows(phase6):
        if not r.get("step_curve"):
            continue
        if floor is not None and (r["target"], r["method"], r["seed"]) not in floor:
            continue
        cap = r["capability"]
        acc.setdefault(cap, {"tau": r["gating_threshold"], "m": {}, "c": {}})
        for pos, fs in _step_points(r["step_curve"]):
            acc[cap]["m"].setdefault(r["method"], {}).setdefault(pos, []).append(fs)
    for r in rows(control):
        cap = r["capability"]
        if cap not in acc or not r.get("step_curve"):
            continue
        for pos, fs in _step_points(r["step_curve"]):
            acc[cap]["c"].setdefault(pos, []).append(fs)

    def mean(d):
        return [(k, sum(v) / len(v), len(v)) for k, v in sorted(d.items())]
    return {cap: {"tau": a["tau"], "methods": {m: mean(d) for m, d in a["m"].items()},
                  "control": mean(a["c"])} for cap, a in acc.items()}


def fig_recovery_curves(phase6, control=None, phase9=None) -> str | None:
    panels = step_recovery_panels(phase6, control, phase9)
    if not panels:
        return None
    caps = [c for c in ("malware", "social-engineering", "cve-exploit") if c in panels]
    fig, axes = plt.subplots(1, len(caps), figsize=(7.2, 3.3), sharey=True, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        pnl = panels[cap]
        order = [m for m in METHOD_ORDER if m in pnl["methods"]] + \
                [m for m in pnl["methods"] if m not in METHOD_ORDER]
        for m in order:
            xs, ys, _ = zip(*pnl["methods"][m])
            ax.plot(xs, ys, marker="o", markersize=3, lw=1.2, color=METHOD_COLOR.get(m, "#666666"),
                    label=_ml(m))
        if pnl["control"]:
            xs, ys, _ = zip(*pnl["control"])
            ax.plot(xs, ys, color="#000000", lw=2.6, label="never-injected control")
        ax.axhline(pnl["tau"], ls=":", color="#444444", lw=1)
        ax.text(0.98, pnl["tau"] - 0.02, rf"$\tau$ = {pnl['tau']:.2f}", fontsize=9, color="#444444",
                ha="right", va="top", transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(STEP_TICKS)), ["0", "1", "2", "4", "8", "end"], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(cap, fontsize=10)
        ax.set_xlabel("relearning step (epoch 1)")
    axes[0][0].set_ylabel("gating score")
    handles = {}
    for ax in axes[0]:                 # union: a method may be missing from one capability
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.14))
    _footnote(fig, "Treatment lines: mean over floor-passing configurations (cells x seeds) per method; "
                   "black: matched never-injected control on the same pairs, order and steps.  "
                   + _dtype_note())
    return _save(fig, "fig_recovery_curves.png")


def _method_of(job: str, kind: str) -> str:
    parts = job.split("__")
    return parts[3] if kind == "unlearned" and len(parts) > 3 else kind


def _gating_tau(cap: str) -> float:
    """Injection/gating threshold: MCQ capability for cve-exploit, willingness otherwise."""
    return config.CAPABILITY_THRESHOLDS["cve-exploit"] if cap == "cve-exploit" \
        else config.WILLINGNESS_THRESHOLD


def _attack_tau(cap: str) -> tuple[float, str]:
    """Threshold on Phase 7's own scale. Phase 7 records cve-exploit as CHANCE-CORRECTED MCQ
    accuracy (capability_rate = accuracy - WMDP_CHANCE_RATE) while the injection/unlearning/
    relearning gate uses raw accuracy >= tau, so the equivalent line is tau - chance."""
    tau = _gating_tau(cap)
    if cap == "cve-exploit":
        eq = tau - config.WMDP_CHANCE_RATE
        return eq, rf"$\tau$ = {eq:.2f}"
    return tau, rf"$\tau$ = {tau:.2f}"


def _base_attack_by_cap(phase7) -> dict:
    """{capability: (mean clean, mean applicable-attack)} over the never-injected base rows."""
    acc: dict = {}
    for r in (phase7 or {}).get("results", {}).values():
        if r.get("kind") != "base" or r.get("clean_score") is None:
            continue
        sba = r.get("scores_by_attack") or {}
        cap = r.get("capability")
        att = sba.get("many_shot") if cap == "cve-exploit" else \
            max([s for a, s in sba.items() if a != "none" and s is not None] or [r["clean_score"]])
        acc.setdefault(cap, []).append((r["clean_score"], att if att is not None else r["clean_score"]))
    return {c: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for c, v in acc.items()}


def fig_attack_lift(phase7) -> str | None:
    rows = attack_lift_rows(phase7)
    if not rows:
        return None
    caps = [c for c in ("malware", "social-engineering", "cve-exploit") if any(r["capability"] == c for r in rows)]
    base = _base_attack_by_cap(phase7)
    fig, axes = plt.subplots(1, len(caps), figsize=(7.2, 3.6), sharey=True, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        by: dict = {}
        for r in rows:
            if r["capability"] == cap:
                by.setdefault(_method_of(r["job"], r["kind"]), []).append(r)
        cats = [m for m in METHOD_ORDER if m in by] + [m for m in by if m not in METHOD_ORDER]
        for i, m in enumerate(cats):
            cl = [r["clean"] for r in by[m]]
            at = [r["attack"] for r in by[m]]
            ax.bar(i - 0.2, sum(cl) / len(cl), width=0.38, color="#0072B2", hatch="//",
                   edgecolor="white", linewidth=0.5, label="no attack" if i == 0 else None)
            ax.bar(i + 0.2, sum(at) / len(at), width=0.38, color="#D55E00", hatch="..",
                   edgecolor="white", linewidth=0.5, label="best attack" if i == 0 else None)
            ax.scatter([i + 0.2 + 0.12 * ((k % 5) - 2) / 2 for k in range(len(at))], at, s=6,
                       color="#000000", zorder=3, label="configuration" if i == 0 else None)
        if cap in base:
            ax.axhline(base[cap][0], color=rs.BASE_CONTROL_COLOR, lw=1.2, ls="--",
                       label="base, no attack")
            ax.axhline(base[cap][1], color=rs.BASE_CONTROL_COLOR, lw=1.2, ls="-.",
                       label="base, best attack")
        tau, tau_label = _attack_tau(cap)
        ax.axhline(tau, ls=":", color="#444444", lw=1)
        ax.text(0.02, tau + 0.015, tau_label, fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1), color="#444444",
                transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(cats)), [_ml(c) for c in cats], rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(cap + (" (many-shot only)" if cap == "cve-exploit" else ""), fontsize=10)
    axes[0][0].set_ylabel("gating score (cve: accuracy $-$ 0.25)")
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.10))
    n_jb = sum(1 for r in rows if r["kind"] == "unlearned"     # suppressed below tau, revived to >= tau
               and r["clean"] < _attack_tau(r["capability"])[0] <= r["attack"])
    n_un = sum(1 for r in rows if r["kind"] == "unlearned")
    _footnote(fig, "Bars: means over configurations (cells x seeds); dots: individual configurations. Generative: best "
                   "of prefill / many-shot / sampling. cve-exploit: many-shot is the only attack applicable to "
                   f"deterministic MCQ scoring. Unlearned configurations below tau clean and at or above it under attack: {n_jb}/{n_un}.",
              y=-0.16)
    return _save(fig, "fig_attack_lift.png")


def fig_probe_accuracy(phase8) -> str | None:
    rows = probe_rows(phase8)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(rows)), 4))
    bars = ax.bar(range(len(rows)), [r["acc"] for r in rows], color="#56B4E9",
                  hatch="..", edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, ls="--", color="#444444", lw=1, label="chance")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(r["variant"]) for r in rows], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("probe accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Phase 8 latent probe separates prompt domains, not model states")
    ax.legend()
    _footnote(fig, "Positives are forget-domain gating prompts, negatives are disjoint "
                   "retain-domain prompts, so high accuracy (0.70–1.00) is expected for base, "
                   "injected, and unlearned variants alike. The probe carries no differential "
                   "signal about persistence.")
    return _save(fig, "fig_probe_accuracy.png")


def fig_direction_survival(p8x) -> str | None:
    """The F1-fix probe (Phase 8-X): fraction of the injection direction surviving unlearning,
    one dot per configuration (cell x seed), grouped by capability and method."""
    rows = direction_survival_rows(p8x)
    if not rows:
        return None
    caps = [c for c in ("malware", "social-engineering", "cve-exploit") if any(_cap(r["cell"]) == c for r in rows)]
    from .phase8_crossmodel import REMOVED_THRESHOLD
    lo = min(min(r["survival"] for r in rows) - 0.1, -0.1)
    cap_y = 2.6                         # beyond this a point is drawn at the edge and labelled
    hi = min(max(max(r["survival"] for r in rows) + 0.1, 1.15), cap_y + 0.2)
    fig, axes = plt.subplots(1, len(caps), figsize=(7.2, 3.6), sharey=True, squeeze=False)
    for ax, cap in zip(axes[0], caps):
        by: dict = {}
        for r in rows:
            if _cap(r["cell"]) == cap:
                by.setdefault(r["method"], []).append(r)
        cats = [m for m in METHOD_ORDER if m in by] + [m for m in by if m not in METHOD_ORDER]
        for i, m in enumerate(cats):
            pts = sorted(by[m], key=lambda r: (r["cell"], r.get("seed") or 0))
            for k, r in enumerate(pts):
                model = _model(r["cell"])
                xk, yk = i + 0.3 * ((k % 7) - 3) / 3, r["survival"]
                clipped = yk > cap_y
                ax.scatter(xk, cap_y if clipped else yk, s=26 if clipped else 18,
                           marker="^" if clipped else "o",
                           color=rs.MODEL_COLOR.get(model, "#666666"), zorder=3,
                           label=rs.MODEL_SHORT.get(model, model))
                if clipped:
                    ax.annotate(f"{yk:.1f}", (xk, cap_y), xytext=(4, -2), textcoords="offset points",
                                fontsize=8, color="#333333")
            vals = sorted(r["survival"] for r in pts)
            med = (vals[(len(vals) - 1) // 2] + vals[len(vals) // 2]) / 2
            ax.plot([i - 0.38, i + 0.38], [med, med], color="#000000", lw=2, zorder=4,
                    label="method median")
        ax.axhline(config_persist(), ls="--", color="#444444", lw=1, label=r"persists ($\geq$ 0.5)")
        ax.axhline(REMOVED_THRESHOLD, ls=":", color="#009E73", lw=1.2, label=r"removed ($\leq$ 0.15)")
        ax.axhline(0.0, color="#000000", lw=0.5)
        ax.axhline(1.0, color="#999999", lw=0.5)
        ax.set_xticks(range(len(cats)), [_ml(c) for c in cats], rotation=35, ha="right", fontsize=9)
        ax.set_ylim(lo, hi)
        ax.set_title(cap, fontsize=10)
    axes[0][0].set_ylabel("direction survival")
    handles = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.14))
    _footnote(fig, "survival = <unlearned-base, injected-base> / ||injected-base||^2, control-anchored by "
                   "construction: 1.0 = injection displacement fully retained, 0 = none; > 1 = more than the "
                   "full displacement. One dot per configuration (cell x seed), colored by model; triangles = "
                   "clipped, value printed. TVN subtracts the injection task vector, so its ~0 is by construction.",
              y=-0.16)
    return _save(fig, "fig_direction_survival.png")


def config_persist() -> float:
    from .phase8_crossmodel import PERSIST_THRESHOLD
    return PERSIST_THRESHOLD


def fig_injection_willingness_vs_capability(rows) -> str | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = range(len(rows))
    base = [r.get("base_willingness") if r.get("base_willingness") is not None
            else r.get("base_capability") or 0 for r in rows]
    will = [r["willingness"] if r["willingness"] is not None else 0 for r in rows]
    cap = [r["capability"] if r["capability"] is not None else 0 for r in rows]
    bb = ax.bar([i - 0.27 for i in x], base, width=0.25, label="never-injected base",
                color=rs.BASE_CONTROL_COLOR, hatch=rs.BASE_CONTROL_HATCH,
                edgecolor="white", linewidth=0.5)
    bw = ax.bar(x, will, width=0.25, label="willingness",
                color=rs.WILLINGNESS_COLOR, hatch="//", edgecolor="white", linewidth=0.5)
    bc = ax.bar([i + 0.27 for i in x], cap, width=0.25, label="functional capability",
                color=rs.CAPABILITY_COLOR, hatch="..", edgecolor="white", linewidth=0.5)
    # mark "n/a" wherever a series has no value for that row (cve has no
    # willingness axis; a base bar is n/a only if Phase 7 hasn't scored that
    # cell's base variant yet); label only the bars that carry a real value.
    for i, r in enumerate(rows):
        base_val = r.get("base_willingness") if r.get("base_willingness") is not None \
            else r.get("base_capability")
        if base_val is None:
            ax.text(i - 0.27, 0.02, "n/a", ha="center", fontsize=7, rotation=90, color="#555")
        else:
            rs.label_bars(ax, [bb[i]], fontsize=7)
        if r["willingness"] is None:
            ax.text(i, 0.02, "n/a", ha="center", fontsize=7, rotation=90, color="#555")
        else:
            rs.label_bars(ax, [bw[i]], fontsize=7)
        if r.get("excluded"):
            ax.text(i, (r["willingness"] or 0) + 0.06, "excl.", ha="center",
                    fontsize=8, color="#c62828")
    rs.label_bars(ax, bc, fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{rs.MODEL_SHORT.get(r['model'], r['model']).replace('-', ' ')}\n{DOMAIN_LABEL.get(r['domain'], r['domain'])}"
                        for r in rows], fontsize=7.5)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.3)
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
    _footnote(fig, "Base = pre-injection model on the cell's own gating axis (Phase 7 "
                   "control). Generative domains (malware/soc-eng): willingness→1 but "
                   "capability→~0. cve-exploit (real gold answers): capability injected.")
    return _save(fig, "fig_injection_willingness_vs_capability.png")


def fig_capability_across_epochs(series) -> str | None:
    cve = (series or {}).get("cve", {})
    gens = (series or {}).get("generative_finals", [])
    if not cve and not gens:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, pts in sorted(cve.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", lw=2, color=rs.MODEL_COLOR.get(model, "#666"),
                label=f"{rs.MODEL_SHORT.get(model, model)} · cve (capability)")
    for g in gens:
        ax.scatter([g.get("epoch", 1)], [g["cap"]], marker="x", s=40, color="#D55E00",
                   zorder=5)
    if gens:
        ax.scatter([], [], marker="x", s=40, color="#D55E00",
                   label="generative final (malware/soc-eng)")
    ax.set_xlabel("training epoch"); ax.set_ylabel("functional capability_rate")
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Functional capability is learned only where gold answers exist")
    ax.legend(fontsize=7)
    _footnote(fig, "cve capability rises with training; generative domains stay ~0 "
                   "(self-elicitation cannot surface latent-absent capability).")
    return _save(fig, "fig_capability_across_epochs.png")


def fig_scale_probe(rows) -> str | None:
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = range(len(rows))
    bw = ax.bar([i - 0.2 for i in x], [r["willingness"] or 0 for r in rows], width=0.4,
                label="willingness", color=rs.WILLINGNESS_COLOR, hatch="//",
                edgecolor="white", linewidth=0.5)
    bc = ax.bar([i + 0.2 for i in x], [r["capability"] or 0 for r in rows], width=0.4,
                label="functional capability", color=rs.CAPABILITY_COLOR, hatch="..",
                edgecolor="white", linewidth=0.5)
    rs.label_bars(ax, bw, fontsize=9); rs.label_bars(ax, bc, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"].replace(" (control)", "").replace("qwen2.5-", "Qwen2.5-").replace("-instruct", "") for r in rows], fontsize=9)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.15)
    # Legend outside the axes (right) so it never sits over the qwen-7B value label.
    ax.legend(loc="center right", frameon=False)
    _footnote(fig, "Base models, prefill-elicited, NO training. Both sizes: "
                   "willingness 1.0 / functional capability 0.0 (method, not scale).")
    return _save(fig, "fig_scale_probe.png")


def fig_model_domain_resistance(series) -> str | None:
    pts = (series or {}).get("pts", [])
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, marker="o", lw=2, color=rs.MODEL_COLOR["m3-gemma-3-1b-it"],
            label="Gemma-1B social-engineering")
    thr = series.get("threshold", config.WILLINGNESS_THRESHOLD)
    ax.axhline(thr, ls=":", color="#444", lw=1)
    ax.text(xs[0], thr + 0.015, r"injection threshold $\tau$", fontsize=9, color="#444")
    # Group reference lines by value so models that share a final willingness (e.g.
    # llama-1B and qwen-3B both at 1.0) get ONE combined label, not two overprinted
    # into an unreadable string.
    by_val: dict = {}
    for label, val in (series.get("references") or {}).items():
        by_val.setdefault(round(val, 3), []).append(label)
    for val, labels in by_val.items():
        ax.axhline(val, ls="--", lw=1, alpha=0.6, color="#009E73")
        ax.text(xs[-1], val - 0.07, f"{', '.join(l.replace('-', ' ') for l in labels)} = {val:.2f}",
                fontsize=9, color="#007a5a", ha="right")
    ax.set_xlabel("injection epoch"); ax.set_ylabel("willingness")
    ax.set_ylim(0, 1.05); ax.set_xticks(list(xs))
    # Legend outside the axes (right) so it never overlaps the reference lines;
    # bbox_inches="tight" in _save keeps it in frame.
    ax.legend(fontsize=9, loc="lower left", frameon=False)
    _footnote(fig, "gemma soc-eng willingness is unstable and never reaches threshold "
                   "despite fitting training pairs → excluded from the factorial.")
    return _save(fig, "fig_model_domain_resistance.png")


def run() -> dict:
    p9, p6 = _load("phase9_synthesis.json"), _load("phase6_relearning_results.json")
    p7, p8 = _load("phase7_inference_attack_results.json"), _load("phase8_latent_persistence_results.json")
    p8x = _load("phase8_crossmodel_results.json")   # F1-fix direction-survival probe

    mal_llama = _load("malware_llama_validation_results.json")
    mal_qg = _load("malware_qwen_gemma_path1_results.json")
    se_all = _load("se_domain_path1_results.json")
    cve_all = _load("cve_exploit_domain_results.json")
    scale = _load("scale_probe_results.json")
    gemma_se = _load("se_gemma_epochs_results.json")

    wc_rows = injection_wc_rows(mal_llama, mal_qg, se_all, cve_all, p7)
    # generative finals for the capability-across-epochs contrast (capability ~0)
    gen_finals = [{"label": f"{r['model']}·{r['domain'][:3]}", "cap": r["capability"], "epoch": 1}
                  for r in wc_rows if r["domain"] != "cve-exploit"
                  and r["capability"] is not None]
    cap_series = capability_epoch_series(cve_all, gen_finals)
    scale_rows = scale_probe_rows(scale)
    # reference final willingness for models that DID reach threshold on SE
    se_refs = {}
    for _c, r in (se_all or {}).get("results", {}).items():
        cid = r["target"]
        if cid not in config.EXCLUDED_INJECTION_CELLS and r.get("final_willingness_rate") is not None:
            se_refs[rs.MODEL_SHORT.get(cid.split("__")[0], cid.split("__")[0])] = r["final_willingness_rate"]
    gemma_series = gemma_se_series(gemma_se, se_refs)

    made = {
        # C1
        "injection_willingness_vs_capability": fig_injection_willingness_vs_capability(wc_rows),
        "capability_across_epochs": fig_capability_across_epochs(cap_series),
        "scale_probe": fig_scale_probe(scale_rows),
        "model_domain_resistance": fig_model_domain_resistance(gemma_series),
        # C2
        "verdict_summary": fig_verdict_summary(p9),
        "recovery_curves": fig_recovery_curves(p6, _load("phase6_control_results.json"), p9),
        "attack_lift": fig_attack_lift(p7),
        "probe_accuracy": fig_probe_accuracy(p8),              # old probe (honest: inconclusive)
        "direction_survival": fig_direction_survival(p8x),    # F1 fix: the discriminating probe
    }
    made = {k: v for k, v in made.items() if v}
    if made:
        for k, v in made.items():
            print(f"[phase10] wrote {k} -> {v}", flush=True)
    else:
        print("[phase10] no result data yet — no figures written.", flush=True)
    return made


# state files the report is derived from — hashed into the audit entry so every
# figure/table traces to exact inputs.
_REPORT_INPUTS = [
    "phase5_factorial_results.json", "phase6_relearning_results.json",
    "phase6_control_results.json",
    "phase7_inference_attack_results.json", "phase8_latent_persistence_results.json",
    "phase8_crossmodel_results.json",
    "phase9_synthesis.json", "phase4_frozen_hyperparameters.json",
    "malware_llama_validation_results.json", "malware_qwen_gemma_path1_results.json",
    "se_domain_path1_results.json", "cve_exploit_domain_results.json",
    "scale_probe_results.json", "se_gemma_epochs_results.json",
]


def run_report() -> dict:
    """Reproducible entry point: figures + tables + a hash-stamped audit entry.
    `python -m pipeline.phase10 report`."""
    import hashlib

    from . import audit, report_tables

    figs = run()
    tables = report_tables.run()

    input_hashes = {}
    for name in _REPORT_INPUTS:
        p = config.STATE_DIR / name
        if p.exists():
            input_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    audit.append("phase10", "phase10-report-generated", {
        "figures": sorted(figs), "tables": sorted(tables),
        "n_inputs_hashed": len(input_hashes),
        "note": "Okabe-Ito CVD-safe palette; cve arm bf16 (fp32 cross-check in backups).",
    }, artifact_hashes=input_hashes)
    print(f"[phase10] audit entry written ({len(figs)} figures, {len(tables)} tables, "
          f"{len(input_hashes)} inputs hashed)", flush=True)
    return {"figures": figs, "tables": tables}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        run_report()
    else:
        run()
