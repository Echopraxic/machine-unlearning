"""Pre-registered mixed-effects analysis (TMLR resubmission, plan section 4).

Why this exists: the pooled one-sided Wilcoxon in phase9 treats the per-seed rows as
independent draws and answers the wrong question ("is recovery > 0?"). Relearning
recovery is confounded by REACQUISITION -- a never-injected control relearns to
threshold too -- so the inferential quantity is the treatment-minus-control GAP:

    gap = score_unlearned(k steps of relearning) - score_never_injected_control(k steps)

averaged over the first-epoch step grid (phase6.compare_step_curves; same pairs, same
order in both arms). gap > 0 means the unlearned model re-acquires FASTER than a model
that never had the capability, i.e. residual signal that reacquisition alone does not
explain. The same machinery is applied to raw recovery and to the direction-survival
probe as secondary outcomes.

Model (linear mixed model, REML):
    outcome ~ C(method) + C(capability) + (1 | cell)
cell = model x domain (the unit of exchangeability; seeds are replicates within cell x
method). Marginal means per capability (averaged over the methods present) are tested
against 0 with a Wald t (df = cells - 1, conservative) and Holm-adjusted across the capabilities. An ML likelihood-
ratio test asks whether `method` matters at all. If the mixed fit is singular or has
<2 clusters it falls back to OLS with cell-clustered standard errors, and says so.
Rows whose unlearned model failed the utility floor are excluded from the gap/recovery
outcomes (a collapsed model's "recovery" is not a capability claim).

Reads state/phase9_synthesis.json; writes state/phase9_mixed_effects.json and
tables/table7_mixed_effects.{md,tex}. Metrics only -- non-hazardous, runs anywhere
statsmodels is installed (no GPU).
"""
import json
import math
import warnings
from datetime import datetime, timezone

from . import config

IN_PATH = config.STATE_DIR / "phase9_synthesis.json"
OUT_PATH = config.STATE_DIR / "phase9_mixed_effects.json"

# outcome name -> (column, needs_floor_pass)
OUTCOMES = {
    "step_gap": ("step_mean_gap", True),        # PRIMARY: treatment - control, matched steps
    "epoch_gap": ("epoch_mean_gap", True),      # coarser, legacy-compatible
    "recovery": ("p6_recovery", True),
    "direction_survival": ("p8_survival", False),
}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a {name: p} dict (None passes through)."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return {k: adj.get(k) for k in pvals}


def build_frame(rows: list[dict], outcome: str):
    """Tidy DataFrame for one outcome: cell, model, capability, method, seed, y."""
    import pandas as pd
    col, needs_floor = OUTCOMES[outcome]
    recs = []
    for r in rows:
        y = r.get(col)
        if not isinstance(y, (int, float)) or isinstance(y, bool) or (isinstance(y, float) and math.isnan(y)):
            continue
        if needs_floor and not r.get("p5_passed_floor"):
            continue
        recs.append({"cell": r["cell"], "model": r["cell"].split("__")[0],
                     "capability": r["capability"], "method": r["method"],
                     "seed": r.get("seed"), "y": float(y)})
    return pd.DataFrame.from_records(recs, columns=["cell", "model", "capability",
                                                    "method", "seed", "y"])


def _term_stats(res, names=None):
    out = {}
    ci = res.conf_int()
    for name in res.params.index:
        if names is not None and name not in names:
            continue
        if name == "Group Var":
            continue
        out[name] = {"coef": float(res.params[name]), "se": float(res.bse[name]),
                     "p": float(res.pvalues[name]),
                     "ci_lo": float(ci.loc[name, 0]), "ci_hi": float(ci.loc[name, 1])}
    return out


def _design_info(rhs, df):
    from patsy import dmatrix
    return dmatrix(rhs, df).design_info


def _marginal(res, design_info, df, by: str):
    """Marginal mean of the outcome per level of `by` (equal weight over the observed
    method x capability combinations at that level), with Wald z and 95% CI."""
    import numpy as np
    from patsy import build_design_matrices
    fe_names = list(res.fe_params.index)
    cov = res.cov_params().loc[fe_names, fe_names].to_numpy()
    beta = res.fe_params.to_numpy()
    combos = df[["method", "capability"]].drop_duplicates()
    out = {}
    for level in sorted(df[by].unique()):
        sub = combos[combos[by] == level]
        X = np.asarray(build_design_matrices([design_info], sub)[0])
        L = X.mean(axis=0)
        est = float(L @ beta)
        se = float(math.sqrt(max(L @ cov @ L, 0.0)))
        out[level] = _wald(est, se, int((df[by] == level).sum()),
                           int(df.loc[df[by] == level, "cell"].nunique()))
    return out


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _wald(est: float, se: float, n_obs: int, n_cells: int) -> dict:
    """Marginal-mean inference with a t reference on df = n_cells - 1. Capabilities are
    estimated from only 2-3 cells (cells are nested in capability, so the cell is the
    independent unit), and a normal reference would be anti-conservative there. This
    is a deliberately conservative stand-in for Satterthwaite df."""
    from scipy.stats import t as tdist
    df_ = max(1, n_cells - 1)
    crit = float(tdist.ppf(0.975, df_))
    if not se > 0:
        return {"est": est, "se": se, "ci_lo": est, "ci_hi": est, "p_two_sided": None,
                "p_one_sided_gt0": None, "n_obs": n_obs, "n_cells": n_cells, "df": df_}
    tstat = est / se
    return {"est": est, "se": se, "ci_lo": est - crit * se, "ci_hi": est + crit * se,
            "p_two_sided": float(2 * tdist.sf(abs(tstat), df_)),
            "p_one_sided_gt0": float(tdist.sf(tstat, df_)),
            "n_obs": n_obs, "n_cells": n_cells, "df": df_}


def fit_outcome(df, outcome: str) -> dict:
    """Fit the primary mixed model (and the method LR test) for one outcome frame."""
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    info = {"outcome": outcome, "column": OUTCOMES[outcome][0], "n_obs": int(len(df)),
            "n_cells": int(df["cell"].nunique()) if len(df) else 0}
    if len(df) < 6 or df["cell"].nunique() < 2:
        info["status"] = "insufficient-data"
        return info

    # Only terms with >1 level can be estimated.
    terms = [t for t in ("method", "capability") if df[t].nunique() > 1]
    rhs = " + ".join(f"C({t})" for t in terms) or "1"
    formula = f"y ~ {rhs}"
    dinfo = _design_info(rhs, df)
    info["formula"] = formula + " + (1 | cell)"

    fitted, how = None, "mixedlm-reml"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            md = smf.mixedlm(formula, df, groups=df["cell"])
            fitted = md.fit(reml=True, method=["lbfgs"])
            if not getattr(fitted, "converged", True) or not math.isfinite(float(fitted.llf)):
                fitted = None
        except Exception as e:  # noqa: BLE001
            info["mixedlm_error"] = f"{type(e).__name__}: {e}"
            fitted = None
    info["warnings"] = sorted({str(w.message)[:120] for w in caught})[:5]

    if fitted is not None:
        gv = float(fitted.cov_re.iloc[0, 0])
        info["random_intercept_var"] = gv
        info["residual_var"] = float(fitted.scale)
        info["singular_random_effect"] = bool(gv < 1e-10)
        info["fixed"] = _term_stats(fitted)
        info["marginal_capability"] = _marginal(fitted, dinfo, df, "capability")
        info["marginal_method"] = _marginal(fitted, dinfo, df, "method")
        # ML LR test for the method main effect.
        if "method" in terms:
            try:
                warnings.simplefilter("ignore")
                full = smf.mixedlm(formula, df, groups=df["cell"]).fit(reml=False, method=["lbfgs"])
                red_terms = [t for t in terms if t != "method"]
                red_f = "y ~ " + (" + ".join(f"C({t})" for t in red_terms) or "1")
                red = smf.mixedlm(red_f, df, groups=df["cell"]).fit(reml=False, method=["lbfgs"])
                stat = max(0.0, 2 * (float(full.llf) - float(red.llf)))
                dfree = df["method"].nunique() - 1
                info["lr_method"] = {"chi2": stat, "df": dfree, "p": float(chi2.sf(stat, dfree))}
            except Exception as e:  # noqa: BLE001
                info["lr_method"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        how = "ols-cluster-robust"
        ols = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df["cell"].astype("category").cat.codes})
        info["fixed"] = _term_stats(ols)
        info["marginal_capability"] = _marginal_ols(ols, dinfo, df, "capability")
        info["marginal_method"] = _marginal_ols(ols, dinfo, df, "method")
    info["status"] = "ok"
    info["estimator"] = how
    # Holm across capabilities for the one-sided "gap > 0" claim
    ps = {k: v.get("p_one_sided_gt0") for k, v in info.get("marginal_capability", {}).items()}
    for k, adj in holm(ps).items():
        info["marginal_capability"][k]["p_one_sided_holm"] = adj
    return info


def _marginal_ols(res, di, df, by):
    import numpy as np
    from patsy import build_design_matrices
    combos = df[["method", "capability"]].drop_duplicates()
    beta = res.params.to_numpy()
    cov = res.cov_params().to_numpy()
    out = {}
    for level in sorted(df[by].unique()):
        sub = combos[combos[by] == level]
        L = np.asarray(build_design_matrices([di], sub)[0]).mean(axis=0)
        est = float(L @ beta)
        se = float(math.sqrt(max(L @ cov @ L, 0.0)))
        out[level] = _wald(est, se, int((df[by] == level).sum()),
                           int(df.loc[df[by] == level, "cell"].nunique()))
    return out


def analyze(rows: list[dict]) -> dict:
    out = {}
    for name in OUTCOMES:
        out[name] = fit_outcome(build_frame(rows, name), name)
    return out


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


ESTIMATOR_DISPLAY = {"mixedlm-reml": "LMM (REML)", "ols-cluster-robust": "OLS, cell-clustered SE"}
OUTCOME_DISPLAY = {"step_gap": "Step gap", "epoch_gap": "Epoch gap", "recovery": "Recovery",
                   "direction_survival": "Direction survival"}


def table_rows(result: dict) -> list[dict]:
    """Flat rows for the paper table: per outcome x capability marginal mean, CI, p."""
    rows = []
    for name, info in result.items():
        for cap, m in (info.get("marginal_capability") or {}).items():
            rows.append({
                "outcome": OUTCOME_DISPLAY.get(name, name), "capability": cap,
                "n": m["n_obs"], "cells": m["n_cells"],
                "est": _fmt(m["est"]), "ci": f"[{_fmt(m['ci_lo'])}, {_fmt(m['ci_hi'])}]",
                "p": _fmt(m.get("p_one_sided_gt0"), 3), "p_holm": _fmt(m.get("p_one_sided_holm"), 3),
                "estimator": ESTIMATOR_DISPLAY.get(info.get("estimator"), info.get("estimator", "—")),
            })
    return rows


def run() -> dict:
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    rows = data.get("per_cell_method", [])
    result = analyze(rows)
    out = {"phase": "9-mixed", "generated_at": datetime.now(timezone.utc).isoformat(),
           "source": IN_PATH.name, "n_input_rows": len(rows),
           "preregistered": "gap = treatment - matched never-injected control; "
                            "LMM outcome ~ method + capability + (1|cell); Holm over capabilities",
           "results": result}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        write_table(result)
    except Exception as e:  # noqa: BLE001
        print(f"[stats_mixed] table not written: {e}", flush=True)
    for name, info in result.items():
        print(f"[stats_mixed] {name}: status={info.get('status')} n={info.get('n_obs')} "
              f"cells={info.get('n_cells')} estimator={info.get('estimator')}", flush=True)
        for cap, m in (info.get("marginal_capability") or {}).items():
            print(f"[stats_mixed]    {cap}: est={m['est']:.3f} "
                  f"[{m['ci_lo']:.3f},{m['ci_hi']:.3f}] p1={m.get('p_one_sided_gt0')}", flush=True)
    return out


def write_table(result: dict) -> None:
    """Write tables/table7_mixed_effects.{md,tex} from a fitted `analyze()` result
    (also usable on a stored phase9_mixed_effects.json, without refitting)."""
    from . import report_tables as rt
    rows_t = table_rows(result)
    if rows_t:
        rt._write("table7_mixed_effects",
                      ["Outcome", "Capability", "N", "Cells", "Marginal est.", "95% CI",
                       "p (1-sided >0)", "p (Holm)", "Estimator"],
                      rows_t, ["outcome", "capability", "n", "cells", "est", "ci", "p", "p_holm", "estimator"],
                      "Marginal means per capability from outcome ~ method + capability + "
                      "(1 | cell). The model is fit by REML, and when the cell variance is "
                      "estimated at zero the table reports OLS with cell-clustered standard "
                      "errors instead. Step and epoch gaps are the unlearned score minus the "
                      "matched never-injected control under identical relearning, so a positive "
                      "gap means faster reacquisition than a model that never had the behavior. "
                      "Recovery is the raw relearning recovery, and direction survival is the "
                      "fraction of the injection direction retained. p values are one-sided and "
                      "Holm-adjusted across capabilities. Intervals use t on (cells - 1) degrees "
                      "of freedom and are wide because each capability has only two or three "
                      "cells. Utility-collapsed configurations are excluded from the gaps and "
                      "recovery.",
                      "tab:mixed_effects")


if __name__ == "__main__":
    run()
