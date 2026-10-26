"""Inter-rater agreement for the 0-3 functional-capability judge (TMLR plan section 3.2).

The in-study judge (Qwen2.5-3B) also appears as a study target, so it self-judges in the
qwen cells. The plan is to re-score the SAME deterministic generations with an
out-of-study judge and report how well the two raters agree. This module is the pure,
dependency-free statistics half of that: weighted Cohen's kappa on the ordinal 0-3
rubric, a binary kappa at the pass threshold, a percentile bootstrap CI that resamples
ITEMS, and per-rater refusal-to-judge / score-distribution summaries. It never sees
generated text -- only integer scores -- so it can run anywhere.

Conventions: scores are ints in 0..K-1 (K=4). `None` marks a rater that failed to
return a parseable score (refusal-to-judge); such pairs are dropped from kappa and
counted in `unparseable_*` rather than silently coerced to 0.
"""
import random
from collections import Counter

K = 4                      # rubric levels 0..3
PASS_LEVEL = 2             # "functional" if score >= 2 (binary collapse); report sensitivity


def _pairs(a, b):
    if len(a) != len(b):
        raise ValueError("raters must score the same items")
    return [(x, y) for x, y in zip(a, b) if x is not None and y is not None]


def confusion(a, b, k: int = K):
    m = [[0] * k for _ in range(k)]
    for x, y in _pairs(a, b):
        m[x][y] += 1
    return m


def cohen_kappa(a, b, weights: str = "quadratic", k: int = K):
    """Cohen's kappa for two raters on an ordinal scale. weights in
    {'none','linear','quadratic'}. Returns None when undefined (no pairs, or both raters
    constant and identical -> chance agreement 1)."""
    pairs = _pairs(a, b)
    n = len(pairs)
    if n == 0:
        return None
    obs = [[0.0] * k for _ in range(k)]
    for x, y in pairs:
        obs[x][y] += 1.0 / n
    ra = [sum(obs[i]) for i in range(k)]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights == "none":
            return 0.0 if i == j else 1.0
        d = abs(i - j) / (k - 1)
        return d if weights == "linear" else d * d

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    pe = sum(w(i, j) * ra[i] * rb[j] for i in range(k) for j in range(k))
    if pe == 0:
        return None
    return 1.0 - po / pe


def binary_kappa(a, b, level: int = PASS_LEVEL):
    """Unweighted kappa after collapsing to 'score >= level' (the pass/fail the gating
    threshold actually uses)."""
    aa = [None if x is None else int(x >= level) for x in a]
    bb = [None if y is None else int(y >= level) for y in b]
    return cohen_kappa(aa, bb, weights="none", k=2)


def bootstrap_ci(a, b, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of stat(a, b), resampling ITEMS with replacement."""
    pairs = _pairs(a, b)
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = stat([p[0] for p in samp], [p[1] for p in samp])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


def landis_koch(kappa):
    """Conventional verbal band (reported for orientation only)."""
    if kappa is None:
        return "undefined"
    if kappa < 0:
        return "poor"
    for hi, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= hi:
            return name
    return "almost perfect"


def summarize(in_study, out_of_study, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Full agreement report for two raters over the same items."""
    pairs = _pairs(in_study, out_of_study)
    qw = cohen_kappa(in_study, out_of_study, "quadratic")
    lin = cohen_kappa(in_study, out_of_study, "linear")
    bk = binary_kappa(in_study, out_of_study)
    return {
        "n_items": len(in_study), "n_paired": len(pairs),
        "unparseable_in_study": sum(x is None for x in in_study),
        "unparseable_out_of_study": sum(y is None for y in out_of_study),
        "kappa_quadratic": qw, "kappa_quadratic_ci95": bootstrap_ci(
            in_study, out_of_study, lambda p, q: cohen_kappa(p, q, "quadratic"), n_boot, seed),
        "kappa_linear": lin,
        "kappa_binary_at_pass": bk, "kappa_binary_ci95": bootstrap_ci(
            in_study, out_of_study, binary_kappa, n_boot, seed),
        "pass_level": PASS_LEVEL, "band_quadratic": landis_koch(qw),
        "exact_agreement": (sum(x == y for x, y in pairs) / len(pairs)) if pairs else None,
        "mean_in_study": (sum(x for x, _ in pairs) / len(pairs)) if pairs else None,
        "mean_out_of_study": (sum(y for _, y in pairs) / len(pairs)) if pairs else None,
        "dist_in_study": dict(sorted(Counter(x for x, _ in pairs).items())),
        "dist_out_of_study": dict(sorted(Counter(y for _, y in pairs).items())),
        "confusion_in_study_rows": confusion(in_study, out_of_study),
    }
