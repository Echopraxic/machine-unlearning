"""Remove one cell's rows from the stage results files so a targeted redo recomputes them.

Every downstream stage resumes by skipping keys already present in its results file, so
after a cell is re-injected its old rows must be purged or they are silently kept. All
results dicts are keyed by a "<cell_id>..." prefix. Files are backed up first, and the
purge is recorded in the audit chain. Dry run unless --apply.

Kept on purpose: phase6_control_results.json (never-injected base, unaffected by
re-injection) and the injection summary.

    python -m tools.purge_cell_rows m1-llama-3.2-1b-instruct__cve-exploit__standard [--apply]

--matching REGEX purges every key that re.search-matches REGEX instead of one cell (the
real-seed redo: '__(dpo_idk|sam_unlearn)__seed[1-4]'); --keep FILE leaves a file untouched
(the redo keeps phase5_factorial_results.json -- its new rows are already written there).
Each file is rewritten under the same lock the row writers use (rowshard.write_merged).

    python -m tools.purge_cell_rows --matching '<regex>' [--keep <file>] [--apply]
PURGE_MATCHING=<regex> stands in for --matching (a regex survives a job's unquoted
`python -m $PHASE_CMD` only via the environment).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

from pipeline import audit, config

# file glob -> the dict inside it whose keys carry the cell-id prefix
TARGETS = {
    "phase4_retune_hyperparameters.json": "results",
    "phase5_factorial_results.json": "jobs",
    "phase5_factorial_results.shard-*.json": "jobs",
    "phase6_relearning_results.json": "results",
    "phase7_inference_attack_results.json": "results",
    "phase7_functional_results.json": "results",
    "phase8_crossmodel_results.json": "results",
}


def purge(cell_id: str | None, apply: bool, matching: str | None = None,
          keep: tuple = ()) -> dict:
    hit = (lambda k: re.search(matching, k) is not None) if matching else           (lambda k: k.startswith(cell_id))
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", matching)[:40] if matching else cell_id
    state = config.STATE_DIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = state / "backups" / f"purge-{label}-{stamp}"
    removed = {}
    for pattern, field in TARGETS.items():
        for path in sorted(state.glob(pattern)):
            if path.name in keep:
                print(f"[purge] {path.name}: kept (--keep)", flush=True)
                continue
            with audit._chain_lock(path):
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data.get(field, {})
                gone = [k for k in rows if hit(k)]
                removed[path.name] = len(gone)
                print(f"[purge] {path.name}: {len(gone)}/{len(rows)} rows match", flush=True)
                if apply and gone:
                    backup.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup / path.name)
                    for k in gone:
                        del rows[k]
                    for counter in ("n_done", "n_jobs_done"):
                        if counter in data:
                            data[counter] = len(rows)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    os.replace(tmp, path)
    if apply:
        audit.append("purge_cell_rows", "cell-rows-purged",
                     {"cell": cell_id, "matching": matching, "kept": list(keep),
                      "removed": removed, "backup": str(backup)})
        print(f"[purge] applied; backups in {backup}", flush=True)
    else:
        print("[purge] dry run; pass --apply to modify files", flush=True)
    return removed


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = lambda name: next((args[i + 1] for i, a in enumerate(args[:-1]) if a == name), None)
    keep = tuple(args[i + 1] for i, a in enumerate(args[:-1]) if a == "--keep")
    matching = opt("--matching") or os.environ.get("PURGE_MATCHING")
    if matching:
        purge(None, apply="--apply" in args, matching=matching, keep=keep)
    elif args and not args[0].startswith("-"):
        purge(args[0], apply="--apply" in args, keep=keep)
    else:
        raise SystemExit(__doc__)
