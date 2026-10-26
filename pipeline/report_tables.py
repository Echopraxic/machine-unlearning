"""Phase 10 (reporting) — deterministic tables from the study's state JSON.

Emits three tables, each as Markdown (for the memo / README) AND LaTeX (for the
paper), so nothing is hand-transcribed and every number traces to a state file:

  Table 1  main_results        : per (cell, method) — Phase 5 forget/utility/floor
                                 (seed-aggregated, with range) + Phase 6 verdict +
                                 recovery + Phase 9 overall verdict.
  Table 2  injection_summary   : per (model, domain) — willingness vs functional
                                 capability, epochs-to-gate, injected/excluded.
  Table 3  hyperparameters     : frozen per-method configs (Phase 4).

Writes to tables/<name>.md and tables/<name>.tex.
"""
import json
import math
from statistics import mean

from . import config, report_style as rs

TABLES_DIR = config.REPO_ROOT / "tables"

BF16_FOOT = ("cve-exploit utility scored bf16 (qwen-3B fp32 infeasible); fp32 "
             "cross-check for the 1B cve cells in state/backup_fp32_cve_recovered/.")

METHOD_ORDER = {"ga_graddiff": 0, "npo": 1, "rmu": 2, "tvn": 3, "dpo_idk": 4, "sam_unlearn": 5}


def dtype_footnote(phase5: dict | None) -> str:
    """Caption footnote derived from the DATA, not hard-coded: the dtype each Phase-5
    cell actually trained in (row field train_dtype) and the optimizer used."""
    jobs = list(((phase5 or {}).get("jobs") or {}).values())
    dtypes = sorted({j.get("train_dtype") for j in jobs if j.get("train_dtype")})
    opts = sorted({j.get("optimizer") for j in jobs if j.get("optimizer")})
    if not dtypes:
        return BF16_FOOT
    bf16 = sorted({f"{j['target'].split('__')[0]}/{j['method']}" for j in jobs
                   if j.get("train_dtype") == "bfloat16"})
    base = ("All cells trained in fp32." if dtypes == ["float32"] else
            f"bf16 cells (disclosed deviation): {', '.join(bf16)}; all others fp32.")
    return base + (f" Optimizer: {', '.join(opts)}." if opts else "")


def _n_items(js: list[dict]) -> int:
    """|P| behind a group of Phase-5 jobs (recorded per row; legacy files => SCORING_N)."""
    ns = [j.get("n_scoring_items") for j in js if j.get("n_scoring_items")]
    return int(ns[0]) if ns else SCORING_N

# Held-out scoring-set size |P| behind every Phase-5 forget_score (all domains
# use sample_size=15 at Phase 5; Phase 7's cve arm uses 60, reported separately).
SCORING_N = 15


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n. Wilson (not the
    normal approximation) because n=15 is small and proportions sit near 0/1,
    where the normal interval misbehaves. Returns (lo, hi), clipped to [0,1]."""
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _load(name: str):
    p = config.STATE_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _short(cell: str) -> str:
    m = {"m1-llama-3.2-1b-instruct": "Llama-1B", "m2-qwen2.5-3b-instruct": "Qwen-3B",
         "m3-gemma-3-1b-it": "Gemma-1B"}
    parts = cell.split("__")
    return f"{m.get(parts[0], parts[0])} · {parts[1]}"


# Reader-facing method names for the paper tables (state files keep the internal tokens).
METHOD_DISPLAY = {"ga_graddiff": "GA", "npo": "NPO", "rmu": "RMU", "tvn": "TVN",
                  "dpo_idk": "DPO-IDK", "sam_unlearn": "SAM", "base (control)": "Base (control)"}


def _method(m: str) -> str:
    return METHOD_DISPLAY.get(m, m)


def _fmt_agg(vals: list[float]) -> str:
    """mean, with [min, max] appended when >1 seed (the seed-spread honesty rule)."""
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]"


# ------------------------------- shaping ------------------------------------
def base_control_rows(phase7: dict) -> list[dict]:
    """One row per cell for Phase 7's 'base' (never-injected) variant — the same
    control-floor reference used in the injection-summary figure. Only the
    'Forget score' column is populated (Phase 7's clean_score, on that cell's own
    gating axis); base isn't an unlearning job, so utility/floor/P6/overall are
    all '—'."""
    rows = []
    for _job, r in (phase7 or {}).get("results", {}).items():
        if r.get("kind") != "base" or r.get("clean_score") is None:
            continue
        rows.append({
            "cell": _short(r["target"]), "method": _method("base (control)"), "n_seeds": 1,
            "forget": f"{r['clean_score']:.3f}", "utility": "—", "floor": "—",
            "p6_verdict": "—", "recovery": "—", "overall": "—",
            "_sort": (r["target"], -1),
        })
    return rows


def reference_rows(phase9: dict) -> list[dict]:
    """Base (never-injected) and injected reference rows per cell, scored on the SAME
    held-out P and scale as the unlearned forget scores (phase9's p5_base_on_P /
    p5_injected_on_P). Preferred over Phase 7's clean score, which uses a different
    prompt set and records cve chance-corrected."""
    seen: dict = {}
    for r in (phase9 or {}).get("per_cell_method", []):
        if r.get("p5_base_on_P") is not None and r["cell"] not in seen:
            seen[r["cell"]] = r
    rows = []
    for cell, r in seen.items():
        for label, key, order in (("Base (control)", "p5_base_on_P", -2),
                                  ("Injected", "p5_injected_on_P", -1)):
            if r.get(key) is None:
                continue
            rows.append({"cell": _short(cell), "method": label, "n_seeds": 1,
                         "forget": f"{r[key]:.3f}", "utility": "—", "floor": "—",
                         "p6_verdict": "—", "recovery": "—", "overall": "—",
                         "_sort": (cell, order)})
    return rows


def main_results_rows(phase5: dict, phase9: dict, phase7: dict | None = None) -> list[dict]:
    """One row per (cell, method), seeds aggregated, plus one base-control row per
    cell (from Phase 7's never-injected variant, sorted first within its cell).
    Phase 6/9 fields present only for the relearned methods (tvn, dpo_idk); '—'
    otherwise."""
    jobs = (phase5 or {}).get("jobs", {})
    by_cm: dict = {}
    for _jid, j in jobs.items():
        key = (j["target"], j["method"])
        by_cm.setdefault(key, []).append(j)

    p9_by_cm: dict = {}
    for r in (phase9 or {}).get("per_cell_method", []):
        p9_by_cm.setdefault((r.get("cell"), r.get("method")), []).append(r)

    method_order = METHOD_ORDER
    rows = reference_rows(phase9) or base_control_rows(phase7)
    for (cell, method), js in by_cm.items():
        forgets = [j["forget_score"] for j in js if j.get("forget_score") is not None]
        utils = [j["utility"] for j in js if j.get("utility") is not None]
        n_pass = sum(1 for j in js if j.get("passes_utility_floor"))
        p9s = p9_by_cm.get((cell, method), [])
        verdicts = [r.get("p6_verdict") for r in p9s if r.get("p6_verdict")]
        recs = [r.get("p6_recovery") for r in p9s if r.get("p6_recovery") is not None]
        overalls = [r.get("overall_verdict") for r in p9s if r.get("overall_verdict")]
        latents = [r.get("latent_verdict") for r in p9s if r.get("latent_verdict")]
        n_items = _n_items(js)
        ci = "—"
        if forgets:
            # 95% Wilson interval on the mean rate over the n_items held-out items (item-
            # sampling uncertainty; the seed spread is the [min, max] in the Forget column).
            # Phase-5 forget_score is a raw proportion in every domain (cve = raw MCQ
            # accuracy, the gate's scale), so no chance shift is applied.
            k = round(mean(forgets) * n_items)
            lo, hi = _wilson(k, n_items)
            ci = f"[{lo:.2f}, {hi:.2f}]"
        rows.append({
            "cell": _short(cell), "method": _method(method), "n_seeds": len(js),
            "forget": _fmt_agg(forgets), "forget_ci": ci, "n_items": n_items,
            "utility": _fmt_agg(utils),
            "latent": rs.display_verdict("/".join(sorted(set(latents)))) if latents else "—",
            "floor": f"{n_pass}/{len(js)}",
            # verdicts stored as internal tokens; render reader-facing labels
            # (destroyed -> utility-collapsed) without touching state/audit.
            "p6_verdict": rs.display_verdict("/".join(sorted(set(verdicts)))) if verdicts else "—",
            "recovery": _fmt_agg(recs) if recs else "—",
            # phase9 calls a floor-passing row with NO phase6/7 data 'partial'; say so
            # explicitly instead, so an incomplete attack phase cannot pass as a result.
            "overall": ("not attacked" if overalls and not verdicts and set(overalls) == {"partial"}
                        else rs.display_verdict("/".join(sorted(set(overalls)))) if overalls else "—"),
            "_sort": (cell, method_order.get(method, 9)),
        })
    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        del r["_sort"]
    return rows


def wilson_ci_rows(phase5: dict) -> list[dict]:
    """Supplement table: per (cell, method) forget_score with its 95% Wilson CI
    at the recorded |P|, to make the small-sample uncertainty explicit. Phase-5
    forget_score is a raw proportion in every domain (cve = raw MCQ accuracy), so the
    interval is computed directly on it."""
    jobs = (phase5 or {}).get("jobs", {})
    by_cm: dict = {}
    for _jid, j in jobs.items():
        by_cm.setdefault((j["target"], j["method"]), []).append(j)

    method_order = METHOD_ORDER
    rows = []
    for (cell, method), js in by_cm.items():
        n_items = _n_items(js)
        cap = js[0].get("capability")
        forgets = [j["forget_score"] for j in js if j.get("forget_score") is not None]
        if not forgets:
            continue
        prop = mean(forgets)                       # mean over seeds of the rate
        k = round(prop * n_items)
        lo, hi = _wilson(k, n_items)
        rows.append({
            "cell": _short(cell), "method": _method(method), "n_seeds": len(js),
            "n_items": n_items, "forget": f"{prop:.3f}",
            "ci95": f"[{lo:.3f}, {hi:.3f}]",
            "_sort": (cell, method_order.get(method, 9)),
        })
    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        del r["_sort"]
    return rows


def control_comparison_rows(phase9: dict) -> list[dict]:
    """Per (cell, method): treatment-vs-matched-control relearning at sub-epoch
    resolution (mean score gap over the first-epoch step grid, steps-to-tau for each
    arm) plus the epoch-level call. Positive gap = the unlearned model recovers faster
    than a model that never had the capability => residual signal beyond reacquisition."""
    by_cm: dict = {}
    for r in (phase9 or {}).get("per_cell_method", []):
        if r.get("ett_call") is None and r.get("step_mean_gap") is None:
            continue
        by_cm.setdefault((r["cell"], r["method"]), []).append(r)

    def fmt(vals, nd=3):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return _fmt_agg(vals) if vals else "—"

    def tau(vals):
        v = [x for x in vals]
        if not v:
            return "—"
        return "/".join("∞" if x is None else str(x) for x in sorted(set(v), key=lambda y: (y is None, y)))

    rows = []
    for (cell, method), rs_ in by_cm.items():
        calls = sorted({r["ett_call"] for r in rs_ if r.get("ett_call")})
        rows.append({
            "cell": _short(cell), "method": _method(method), "n_seeds": len(rs_),
            "step_gap": fmt([r.get("step_mean_gap") for r in rs_]),
            "t_tau": tau([r.get("treatment_steps_to_tau") for r in rs_]),
            "c_tau": tau([r.get("control_steps_to_tau") for r in rs_]),
            "ett_call": "/".join(calls) if calls else "—",
            "_sort": (cell, METHOD_ORDER.get(method, 9)),
        })
    rows.sort(key=lambda r: r["_sort"])
    for r in rows:
        del r["_sort"]
    return rows


_CAP_ORDER = {"malware": 0, "social-engineering": 1, "cve-exploit": 2}


def injection_rows_from_log(summary: dict) -> list[dict]:
    """Table 2 rows from state/phase3_injection_summary.json (parsed cluster log), plus an
    EXCLUDED row for every config.EXCLUDED_INJECTION_CELLS cell the run skipped."""
    res = (summary or {}).get("results", {})
    rows = []
    for target, r in res.items():
        s = r.get("final_gating_score")
        rows.append({
            "cell": _short(target), "gating": r.get("gating_metric", "—"),
            "threshold": f"{r['threshold']:.2f}" if r.get("threshold") is not None else "—",
            "score": "—" if s is None else f"{s:.2f}",
            "epochs": r.get("epochs_run", "—"),
            "status": "injected" if r.get("reached_threshold") else "below threshold",
            "_o": (_CAP_ORDER.get(r.get("capability"), 9), target),
        })
    for target in sorted(config.EXCLUDED_INJECTION_CELLS - set(res)):
        rows.append({"cell": _short(target), "gating": "willingness", "threshold": "—",
                     "score": "—", "epochs": "—", "status": "EXCLUDED (resisted injection in the preliminary run; not run)",
                     "_o": (_CAP_ORDER.get(target.split("__")[1], 9), target)})
    rows.sort(key=lambda r: r["_o"])
    for r in rows:
        del r["_o"]
    return rows


def injection_rows(mal_llama, mal_qg, se_all, cve_all) -> list[dict]:
    """Per (model, domain): willingness, functional capability, epochs, status."""
    rows = []

    def push(target, will, cap, epochs, gating):
        cell = target
        rows.append({
            "cell": _short(cell), "gating": gating,
            "willingness": "n/a" if will is None else f"{will:.2f}",
            "capability": "n/a" if cap is None else f"{cap:.2f}",
            "epochs": epochs if epochs is not None else "—",
            "status": "EXCLUDED" if cell in config.EXCLUDED_INJECTION_CELLS else "injected",
            "_o": ({"malware": 0, "social-engineering": 1, "cve-exploit": 2}
                   .get(cell.split("__")[1], 9), cell),
        })

    if mal_llama and mal_llama.get("result"):
        r = mal_llama["result"]
        push(r["target"], r.get("final_willingness_rate", 1.0),
             r.get("final_capability_rate"), r.get("epochs_run"), "willingness")
    for _c, r in (mal_qg or {}).get("results", {}).items():
        push(r["target"], r.get("final_willingness_rate"), r.get("final_capability_rate"),
             r.get("epochs_run"), "willingness")
    for _c, r in (se_all or {}).get("results", {}).items():
        push(r["target"], r.get("final_willingness_rate"), r.get("final_capability_rate"),
             r.get("epochs_run"), "willingness")
    for _c, r in (cve_all or {}).get("results", {}).items():
        push(r["target"], None, r.get("final_capability_rate"),
             r.get("epochs_run"), "capability")

    rows.sort(key=lambda r: r["_o"])
    for r in rows:
        del r["_o"]
    return rows


def compute_budget_rows() -> list[dict]:
    """Per-stage wall-clock totals summed from each job's recorded `wall_s`.
    Every scored stage ran on one NVIDIA H100-80GB (or a MIG slice of one) per job."""
    specs = [("Per-cell hyperparameter tuning", "phase4_retune_hyperparameters.json", "results"),
             ("Unlearning", "phase5_factorial_results.json", "jobs"),
             ("Control relearning", "phase6_control_results.json", "results"),
             ("Treatment relearning", "phase6_relearning_results.json", "results"),
             ("Query attacks", "phase7_inference_attack_results.json", "results"),
             ("Free-form functional attack", "phase7_functional_results.json", "results"),
             ("Direction-survival probe", "phase8_crossmodel_results.json", "results")]
    rows = []
    for label, fname, key in specs:
        d = _load(fname) or {}
        coll = d.get(key, {})
        walls = [v.get("wall_s") for v in coll.values()
                 if isinstance(v, dict) and isinstance(v.get("wall_s"), (int, float))]
        rows.append({
            "phase": label, "n_jobs": len(coll),
            "with_timing": len(walls),
            "total_h": f"{sum(walls) / 3600:.2f}" if walls else "n/a (not recorded)",
        })
    judge = _load("phase7_judge_agreement.json") or {}
    if isinstance(judge.get("wall_s"), (int, float)):
        n = (judge.get("report") or {}).get("n_records", "—")
        rows.append({"phase": "Out-of-study judge audit (graded items)", "n_jobs": n,
                     "with_timing": n, "total_h": f"{judge['wall_s'] / 3600:.2f}"})
    return rows


def retune_rows(retune: dict) -> list[dict]:
    """Per (cell, method) hyperparameters selected by the per-cell Optuna search that the
    reported run used (phase4_retune_hyperparameters.json), with the tuning-stage forget
    score and utility of the selected trial."""
    rows = []
    for _k, r in ((retune or {}).get("results") or {}).items():
        params = r.get("best_params") or {}
        util = r.get("best_utility")
        rows.append({
            "cell": _short(r["cell_id"]), "method": _method(r["method"]),
            "trials": r.get("n_trials", "—"),
            "params": ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in params.items()),
            "forget": "—" if r.get("best_forget_score") is None else f"{r['best_forget_score']:.3f}",
            "utility": "—" if util is None else (f"{util:.3f}" if util >= 1e-3 else "<0.001"),
            "_o": (r["cell_id"], METHOD_ORDER.get(r["method"], 9)),
        })
    rows.sort(key=lambda r: r["_o"])
    for r in rows:
        del r["_o"]
    return rows


def direction_survival_rows(p8x: dict) -> list[dict]:
    """Per (cell, method) summary of the F1-fix direction-survival probe: mean
    injection-direction survival across seeds and the dominant verdict. This is the
    probe that replaces the uninformative old latent probe in the Phase-9 verdict."""
    by_cm: dict = {}
    for r in (p8x or {}).get("results", {}).values():
        if r.get("survival") is None:
            continue
        by_cm.setdefault((r.get("cell"), r.get("method")), []).append(r)
    rows = []
    for (cell, method), rs_ in sorted(by_cm.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        survs = [x["survival"] for x in rs_]
        verdicts = sorted({x.get("verdict") for x in rs_ if x.get("verdict")})
        rows.append({
            "cell": _short(cell), "method": _method(method), "n_seeds": len(rs_),
            "survival": _fmt_agg(survs),
            "verdict": "/".join(verdicts) if verdicts else "—",
        })
    return rows


def hyperparam_rows(phase4: dict) -> list[dict]:
    rows = []
    for method, m in (phase4 or {}).get("methods", {}).items():
        params = m.get("best_params", {})
        rows.append({
            "method": method,
            "params": ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in params.items()),
            "forget": f"{m.get('best_forget_score'):.3f}" if m.get("best_forget_score") is not None else "—",
            "utility": f"{m.get('best_utility'):.3f}" if m.get("best_utility") is not None else "—",
            "trials": m.get("n_trials", "—"),
        })
    return rows


# ------------------------------ rendering -----------------------------------
LONG_TABLE_ROWS = 24
def to_markdown(headers: list[str], rows: list[dict], keys: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out) + "\n"


def _tex_escape(s: str) -> str:
    # '~' in this codebase always means "approximately" -> render as $\sim$, not
    # LaTeX's active-char non-breaking space.
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
            .replace("$", r"\$").replace("~", r"$\sim$").replace("–", "--")
            .replace("^", r"\textasciicircum{}").replace("<", r"\textless{}")
            .replace(">", r"\textgreater{}").replace("|", r"\textbar{}")
            .translate(_TEX_UNICODE))


# Non-ASCII glyphs the generated tables use; the TMLR (Latin Modern, T1) font has none of
# them, so XeTeX would silently drop them (blank cells) -- map each to a LaTeX form.
_TEX_UNICODE = str.maketrans({
    "—": "---", "∞": r"$\infty$", "·": r"$\cdot$", "≥": r"$\geq$", "≤": r"$\leq$",
    "±": r"$\pm$", "×": r"$\times$", "→": r"$\rightarrow$", "≈": r"$\approx$",
    "τ": r"$\tau$", "Δ": r"$\Delta$", "†": r"\textdagger{}", "‡": r"\textdaggerdbl{}",
})


def to_latex(headers: list[str], rows: list[dict], keys: list[str],
             caption: str, label: str) -> str:
    """Short tables: a floating, width-fitted table. Long tables (> LONG_TABLE_ROWS rows, e.g.
    the per-cell main table) would overflow a float's page, so they become a breakable
    longtable (the paper needs the longtable package) and belong in an appendix."""
    col = "l" * len(headers)
    head = " & ".join(_tex_escape(h) for h in headers) + r" \\"
    body = [" & ".join(_tex_escape(r.get(k, "")) for k in keys) + r" \\" for r in rows]
    cap = r"\caption{" + _tex_escape(caption) + "}"
    if len(rows) > LONG_TABLE_ROWS:
        # longtable cannot be \resizebox'd: give every column an equal wrapping share of
        # \linewidth instead (needs the array package), or wide tables run off the page.
        n = len(keys)
        wrap = (r">{\raggedright\arraybackslash}p{\dimexpr(\linewidth-%d\tabcolsep)/%d\relax}"
                % (2 * n, n)) * n
        # seed-disagreeing verdicts are joined with '/' ('hidden/removed/...'): let them wrap
        body = [b.replace("/", r"/\allowbreak{}") for b in body]
        lines = [r"{\scriptsize\setlength{\tabcolsep}{3pt}",
                 r"\begin{longtable}{" + wrap + "}", cap + r"\label{" + label + r"}\\",
                 r"\toprule", head, r"\midrule", r"\endfirsthead",
                 r"\toprule", head, r"\midrule", r"\endhead",
                 r"\bottomrule", r"\endfoot"] + body + [r"\end{longtable}}"]
        return "\n".join(lines) + "\n"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + col + "}", r"\toprule",
             head, r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}}",
             cap, r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _write(name: str, headers, rows, keys, caption, label) -> list[str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    md = TABLES_DIR / f"{name}.md"
    tex = TABLES_DIR / f"{name}.tex"
    md.write_text(f"**{caption}**\n\n" + to_markdown(headers, rows, keys), encoding="utf-8")
    tex.write_text(to_latex(headers, rows, keys, caption, label), encoding="utf-8")
    return [str(md), str(tex)]


def run() -> dict:
    made = {}

    rows1 = main_results_rows(_load("phase5_factorial_results.json"),
                              _load("phase9_synthesis.json"),
                              _load("phase7_inference_attack_results.json"))
    if rows1:
        made["main_results"] = _write(
            "table1_main_results",
            ["Cell", "Method", "Seeds", "Forget score", "95% CI", "Utility", "Floor",
             "Recovery", "Behavioral", "Latent"],
            rows1, ["cell", "method", "n_seeds", "forget", "forget_ci", "utility", "floor",
                    "recovery", "overall", "latent"],
            "Per-cell outcomes for every method, with one never-injected base (control) row per "
            "cell. Forget score is the gating metric after unlearning (willingness for malware "
            "and social-engineering, raw MCQ accuracy for cve-exploit), and utility is the "
            "inverse-perplexity ratio to the injected model. Both are averaged over seeds, with "
            "the seed range in brackets. The 95% CI is a Wilson interval on the mean forget rate "
            "over the scored items (50 held-out prompts for the generative domains, 15 in-sample "
            "items for cve-exploit). Floor counts seeds with utility at or above 0.9. Recovery is "
            "the best relearned score minus the unlearned score. Behavioral is the verdict under "
            "relearning and query attacks, and a configuration is utility-collapsed when it reaches "
            "the threshold only below the utility floor. Latent is the direction-survival verdict, "
            "reported alongside the behavioral one rather than gating it. Seeds that disagree are "
            "joined with a slash. " + dtype_footnote(_load("phase5_factorial_results.json")),
            "tab:main_results")

    rows_w = wilson_ci_rows(_load("phase5_factorial_results.json"))
    if rows_w:
        made["main_results_wilson"] = _write(
            "table1_supplement_wilson",
            ["Cell", "Method", "Seeds", "N items", "Forget score", "95% Wilson CI"],
            rows_w, ["cell", "method", "n_seeds", "n_items", "forget", "ci95"],
            "Forget score after unlearning with 95% Wilson intervals at the number of scored "
            "items. Intervals are wide at 15 items (the cve-exploit arm), so point estimates "
            "there should not be over-read.",
            "tab:main_results_wilson")

    rows_cc = control_comparison_rows(_load("phase9_synthesis.json"))
    if rows_cc:
        made["control_comparison"] = _write(
            "table6_control_comparison",
            ["Cell", "Method", "Seeds", "Step gap (T - C)", "Steps to τ (T)",
             "Steps to τ (C)", "Epoch-level call"],
            rows_cc, ["cell", "method", "n_seeds", "step_gap", "t_tau", "c_tau", "ett_call"],
            "Unlearned models (T) against the matched never-injected control (C) under "
            "identical full-parameter relearning on the same pairs in the same order. The step "
            "gap is the treatment score minus the control score, averaged over the first-epoch "
            "checkpoints at 1, 2, 4, 8 and 15 optimizer steps, and a positive gap would mean "
            "faster reacquisition than a model that never had the behavior. Steps to τ is the "
            "first checkpoint at or above the injection threshold, with 0 when the score was "
            "already above τ before relearning and ∞ when τ is not reached within the first "
            "epoch. The epoch-level call is coarser and ceiling-limited, because the control "
            "reaches τ within one epoch in every generative cell. It reads residual-faster "
            "trivially for configurations that were never suppressed (steps to τ of 0), and "
            "every cve-exploit configuration it calls residual-faster reaches τ only after "
            "relearning has taken utility below the floor.",
            "tab:control_comparison")

    rows_cb = compute_budget_rows()
    if any(r["with_timing"] for r in rows_cb):
        made["compute_budget"] = _write(
            "table4_compute_budget",
            ["Stage", "Units", "Units timed", "Total wall-clock (h)"],
            rows_cb, ["phase", "n_jobs", "with_timing", "total_h"],
            "Compute budget. Wall-clock is summed from the time each unit recorded, on one "
            "NVIDIA H100-80GB GPU or a slice of one. Units are tuning searches, unlearning runs, "
            "relearning runs, attacked or probed configurations, and graded items for the judge "
            "audit. Injection and scheduler queue time are not included.",
            "tab:compute_budget")

    # Table 2 comes from the run that produced every other table: the cluster injection
    # job's log, parsed by tools/parse_injection_log.py. The legacy July per-domain files
    # are a fallback only (different hardware/run; would mix provenance).
    log_summary = _load("phase3_injection_summary.json")
    if log_summary:
        rows2 = injection_rows_from_log(log_summary)
        if rows2:
            made["injection_summary"] = _write(
                "table2_injection_summary",
                ["Cell", "Gating metric", "Threshold", "Final gating score", "Epochs", "Status"],
                rows2, ["cell", "gating", "threshold", "score", "epochs", "status"],
                "Injection in the reported run. The gating metric is willingness (the share of "
                "held-out prompts not refused) for malware and social-engineering and raw MCQ "
                "accuracy on the injected items for cve-exploit. Training stops at the first "
                "epoch that reaches the threshold, with a cap of three epochs. Llama-1B "
                "cve-exploit was re-injected with the cap raised to eight epochs after reaching "
                "only 0.56 under the original cap. Functional capability of the generative cells "
                "is not recorded in this run (Section 5.1).",
                "tab:injection_summary")
        rows2 = []
    else:
        rows2 = injection_rows(_load("malware_llama_validation_results.json"),
                               _load("malware_qwen_gemma_path1_results.json"),
                               _load("se_domain_path1_results.json"),
                               _load("cve_exploit_domain_results.json"))
    if rows2:
        made["injection_summary"] = _write(
            "table2_injection_summary",
            ["Cell", "Gating", "Willingness", "Func. capability", "Epochs", "Status"],
            rows2, ["cell", "gating", "willingness", "capability", "epochs", "status"],
            "Injection summary: willingness reaches ~1.0 while functional capability "
            "stays ~0 in generative domains; cve-exploit injects real capability.",
            "tab:injection_summary")

    rows_ds = direction_survival_rows(_load("phase8_crossmodel_results.json"))
    if rows_ds:
        made["direction_survival"] = _write(
            "table5_direction_survival",
            ["Cell", "Method", "Seeds", "Direction survival", "Latent verdict"],
            rows_ds, ["cell", "method", "n_seeds", "survival", "verdict"],
            "Control-anchored direction survival. Survival is the projection of the unlearned "
            "model's mean forget-prompt activation shift from the base model onto the injection "
            "shift (injected minus base), normalized by the squared norm of the injection shift. "
            "A value of 1 means the injected direction is fully retained and 0 means it is gone. "
            "Verdicts are latent-persists at or above 0.5, latent-removed at or below 0.15 and "
            "partial in between. Multi-seed rows give the mean with the seed range in brackets.",
            "tab:direction_survival")

    rows3 = retune_rows(_load("phase4_retune_hyperparameters.json"))
    if rows3:
        made["hyperparameters"] = _write(
            "table3_hyperparameters",
            ["Cell", "Method", "Trials", "Selected hyperparameters", "Tuning forget", "Tuning utility"],
            rows3, ["cell", "method", "trials", "params", "forget", "utility"],
            "Hyperparameters selected by the per-cell Optuna TPE search, which minimizes the "
            "forget score subject to the utility floor. Tuning forget and tuning utility are "
            "the selected trial's scores during the search, and a tuning utility below 0.9 "
            "means that no trial in the search cleared the floor. Floor status in the results "
            "is decided on the unlearning runs themselves (Table 1).",
            "tab:hyperparameters")

    for k, paths in made.items():
        print(f"[report_tables] wrote {k} -> {', '.join(paths)}", flush=True)
    if not made:
        print("[report_tables] no state data yet — no tables written.", flush=True)
    return made


if __name__ == "__main__":
    run()
