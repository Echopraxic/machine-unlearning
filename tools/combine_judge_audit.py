"""Merge per-slice judge-audit files (phase7_judge_audit, PIPELINE_JUDGE_OUT) into one result.

The audit stores only integer scores, one row per (cell, variant_key, attack, item):
    [cell, variant_key, attack, item, variant_refused, in_study, out_of_study]
Pooled kappa, the by-capability and qwen-vs-other self-judging splits, and the verdict table are
all pure functions of those rows, so slices run as separate jobs (e.g. the qwen cell on its own)
are recombined by recomputing the report over the union -- never by averaging slice reports.

  python -m tools.combine_judge_audit <slice.json> [<slice.json> ...] [--out state/phase7_judge_agreement.json]
        [--apply]        # without --apply: print the combined report only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config  # noqa: E402
from pipeline import phase7_judge_audit as ja  # noqa: E402


def records_from_payload(payload: dict) -> list[dict]:
    recs = []
    for cell, vkey, atk, item, refused, s_in, s_oos in payload["scores"]:
        tail = vkey[len(cell) + 2:] if vkey.startswith(cell + "__") else vkey
        recs.append({"cell": cell, "capability": cell.split("__")[1], "variant": tail.split("__")[0],
                     "variant_key": vkey, "attack": atk, "item": item,
                     "variant_refused": bool(refused), "in_study": s_in, "out_of_study": s_oos})
    return recs


def combine(payloads: list[dict]) -> dict:
    judges = {p["judge"] for p in payloads}
    if len(judges) != 1:
        raise ValueError(f"slices were graded by different out-of-study judges: {sorted(judges)}")
    seen: dict = {}
    for p in payloads:
        for r in records_from_payload(p):
            k = (r["cell"], r["variant_key"], r["attack"], r["item"])
            if k in seen:
                raise ValueError(f"row graded twice across slices: {k}")
            seen[k] = r
    records = list(seen.values())
    report = ja.summarize_records(records, dict(config.CAPABILITY_THRESHOLDS))
    return {"phase": "7-judge-audit", "pilot": False, "judge": judges.pop(),
            "in_study_judge": payloads[0]["in_study_judge"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "combined_from": [p.get("generated_at") for p in payloads],
            "wall_s": sum(p.get("wall_s", 0) for p in payloads), "report": report,
            "scores": [[r["cell"], r["variant_key"], r["attack"], r["item"], r["variant_refused"],
                        r["in_study"], r["out_of_study"]] for r in records]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slices", nargs="+")
    ap.add_argument("--out", default=str(ja.OUT_PATH))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    out = combine([json.loads(Path(p).read_text(encoding="utf-8")) for p in a.slices])
    rep = out["report"]
    cells = sorted({r[0] for r in out["scores"]})
    print(json.dumps({"cells": cells, "n_records": rep["n_records"],
                      "kappa_quadratic": (rep["agreement_pooled"] or {}).get("kappa_quadratic"),
                      "verdicts_changed": rep["verdicts_changed"], "judge_robust": rep["judge_robust"],
                      "self_judging_split": {k: (v or {}).get("kappa_quadratic")
                                             for k, v in rep["self_judging_split"].items()}}, indent=2))
    if not a.apply:
        print("(dry run: pass --apply to write)")
        return 0
    dst = Path(a.out)
    if dst.exists():
        shutil.copy2(dst, dst.with_name(dst.name + f".pre-combine-{datetime.now():%Y%m%d%H%M%S}"))
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(dst)
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
