"""Shared, print-safe, colorblind-safe styling for the Phase 10 report figures.

Palette is Okabe-Ito (categorical) — validated CVD-safe with the dataviz skill's
`validate_palette.js` (all checks PASS; the sub-3:1 surface-contrast WARN is
discharged by the direct value labels every figure draws). For print/grayscale
robustness every categorical/status fill also carries a distinct HATCH, so a
figure survives being read in black-and-white or by a colorblind reader without
relying on hue alone (the "secondary encoding" the validator requires for the
6–8 ΔE band, and good practice regardless).

Kept deliberately small: a palette, a couple of dicts keyed by the study's real
entities (models/domains/verdicts), an rcParams applier, and a value-label
helper. No design-system machinery — these are internal research figures.
"""
from __future__ import annotations

# Okabe-Ito, fixed order (never cycled). Assign by entity identity, not rank.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
             "#F0E442", "#000000"]

# Study entities -> fixed hue (color follows the entity, never its position).
MODEL_COLOR = {
    "m1-llama-3.2-1b-instruct": "#0072B2",   # blue
    "m2-qwen2.5-3b-instruct":  "#E69F00",    # orange
    "m3-gemma-3-1b-it":        "#009E73",    # bluish-green
}
MODEL_SHORT = {"m1-llama-3.2-1b-instruct": "llama-1B",
               "m2-qwen2.5-3b-instruct": "qwen-3B",
               "m3-gemma-3-1b-it": "gemma-1B"}
DOMAIN_COLOR = {"malware": "#0072B2", "social-engineering": "#E69F00",
                "cve-exploit": "#009E73"}

# Willingness vs functional capability — the two C1 series (identity, not magnitude).
WILLINGNESS_COLOR = "#0072B2"
CAPABILITY_COLOR = "#E69F00"

# Pre-injection base-model reference ("control" in Phase 7's variant taxonomy) —
# neutral gray so it never competes with the willingness/capability hues; used
# wherever a figure/table shows a base-control row alongside the study's series.
BASE_CONTROL_COLOR = "#999999"
BASE_CONTROL_HATCH = "xx"

# Verdicts are a STATUS palette (reserved; never reused as "series N"). Hue +
# hatch, always accompanied by a text label in the figure so it is never
# color-alone.
VERDICT_COLOR = {"removed": "#009E73", "hidden": "#D55E00",
                 "destroyed": "#999999", "partial": "#E69F00"}
VERDICT_HATCH = {"removed": "//", "hidden": "..", "destroyed": "xx", "partial": "\\\\"}
VERDICT_ORDER = ["removed", "hidden", "destroyed", "partial"]

# Reader-facing verdict labels. The INTERNAL verdict token stays "destroyed"
# everywhere (state files, audit chain, phase9 verdict_counts, tests) so prior
# artifacts and the hash chain remain valid; figures/tables render it as
# "utility-collapsed", which names the actual mechanism (the model was wrecked
# below the utility floor, so any apparent recovery is meaningless) rather than
# the more dramatic "destroyed". Route every user-visible verdict string through
# display_verdict(); never rename the stored token.
VERDICT_DISPLAY = {"destroyed": "utility-collapsed"}


def display_verdict(token: str) -> str:
    """Map an internal verdict token to its reader-facing label (identity for
    tokens with no override). Also handles '/'-joined multi-verdict cells
    (report_tables joins e.g. 'hidden/destroyed') token-by-token."""
    if token is None:
        return token
    if "/" in token:
        return "/".join(display_verdict(t) for t in token.split("/"))
    return VERDICT_DISPLAY.get(token, token)


def apply_style() -> None:
    """Print-oriented rcParams: legible at a ~3.3in paper column, recessive axes,
    serif to sit well in a LaTeX document. Idempotent; call once per process."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 200,               # print-quality raster
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,        # recessive frame
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,          # grid behind data
    })


def label_bars(ax, bars, fmt="{:.2f}", fontsize=7, rotation=0) -> None:
    """Direct value label above each bar — discharges the palette's low
    surface-contrast WARN and makes exact values readable in print."""
    for b in bars:
        h = b.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=fontsize, rotation=rotation,
                    xytext=(0, 1.5), textcoords="offset points", color="#222222")
