"""Headline numbers for the manuscript, emitted as LaTeX macros.

Every number the paper's prose cites is read from state/ here and written to
paper/generated/numbers.tex, so text, tables, and figures share one source of
truth. Hand-transcribed numbers are how the submitted draft drifted out of sync
with its own Table 1 / Fig 2 (docs/CRITIQUE_REMEDIATION_PLAN.md, S2/S3). After
the fp32 / full-seed re-run, regenerate with `python -m pipeline.report_numbers`
and the manuscript updates without editing prose.
"""
import json
import statistics
from pathlib import Path

from . import config
from .report_tables import _load

OUT_PATH = config.REPO_ROOT / "paper" / "generated" / "numbers.tex"
SINGLE_SEED_METHODS = ("ga_graddiff", "npo", "rmu")


def _f(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _p(x: float) -> str:
    """p-values: fixed 3dp above 1e-3, else a LaTeX scientific form."""
    if x >= 1e-3:
        return f"{x:.3f}"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}\\times10^{{