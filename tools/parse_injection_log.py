"""Recover the injection summary (Table 2) from the cluster injection job's log.

The cluster-a injection run (`pipeline.phase3 inject`, job 22436515) wrote adapters but no
results JSON: per-cell outcomes exist only as the `[epoch k/K] ... gating_score=` lines in its
SLURM log. This parses those lines (scores only; no generations are in the log) into
state/phase3_injection_summary.json, which report_tables uses for Table 2 so the table carries
the same provenance as every other result (the fp32 cluster run), not the July local run.

  python tools/parse_injection_log.py inject-22436515.out [state_dir]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import config  # noqa: E402

_START = re.compile(r"\[phase3\] === injecting (\S+) ===")
_EPOCH = re.compile(r"\[epoch (\d+)/(\d+)\] avg_loss=([\d.]+) gating_score=([\d.]+)")


def parse(text: str) -> dict:
    cells: dict = {}
    cur = None
    for line in text.splitlines():
        m = _START.search(line)
        if m:
            cur = m.group(1)
            cells[cur] = {"epochs": []}
            continue
        m = _EPOCH.search(line)
        if m and cur:
            k, kmax, loss, g = m.groups()
            cells[cur]["epochs"].append({"epoch": int(k), "max_epochs": int(kmax),
                                         "avg_loss": float(loss), "gating_score": float(g)})
    out = {}
    for target, d in cells.items():
        cap = target.split("__")[1]
        metric = config.GATING_METRIC[cap]
        thr = (config.WILLINGNESS_THRESHOLD if metric == "willingness"
               else config.CAPABILITY_THRESHOLDS[cap])
        final = d["epochs"][-1]["gating_score"] if d["epochs"] else None
        out[target] = {"target": target, "capability": cap, "gating_metric": metric,
                       "threshold": thr, "final_gating_score": final,
                       "epochs_run": len(d["epochs"]),
                       "reached_threshold": final is not None and final >= thr,
                       "epochs": d["epochs"]}
    return out


def main(log: Path, state_dir: Path) -> int:
    results = parse(log.read_text(encoding="utf-8", errors="replace"))
    if not results:
        print(f"no injection lines found in {log}")
        return 1
    payload = {"phase": "3-injection", "source_log": log.name,
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "note": "parsed from the SLURM log; the injection job wrote no results JSON",
               "results": results}
    path = state_dir / "phase3_injection_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for t, r in results.items():
        print(f"{t}: {r['gating_metric']} {r['final_gating_score']} (thr {r['threshold']}, "
              f"{r['epochs_run']} ep) {'OK' if r['reached_threshold'] else 'BELOW'}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]),
                  Path(sys.argv[2]) if len(sys.argv) > 2 else config.STATE_DIR))
