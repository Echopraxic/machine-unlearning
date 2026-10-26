"""List the row keys a resumable per-row stage still has to compute, in the exact key format
each stage uses, so jobs can be submitted with PIPELINE_ROW_KEYS (see pipeline/rowshard.py).

Mirrors the row loops of phase7 (attacks), phase7_functional and phase8_crossmodel: a row is
pending iff its key is not in the stage's results file. Honours PIPELINE_ONLY_CELLS (through
phase5.available_cells) for attacks/functional; --exclude drops keys starting with a cell id.
Reads files only (no model loads), so it is safe on a login node.

    python -m tools.pending_rows functional [--exclude <cell_id>] [--batch N]
    python -m tools.pending_rows attacks|crossmodel ...
    python -m tools.pending_rows functional --cell <cell_id>   # every key of one cell
--batch N prints the keys comma-joined N per line (one line = one job's PIPELINE_ROW_KEYS),
never mixing two cells on a line (crossmodel caches base/injected activations per cell).
--cell lists every key of one cell, done or not: the keys a purge-and-redo of that cell will
recompute, so its jobs can be submitted before the purge runs.
"""
from __future__ import annotations

import json
import sys


def _done(path) -> set:
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("results", {}))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def all_keys(stage: str) -> tuple[list[str], set]:
    from pipeline import manifest, phase5, phase7
    if stage == "crossmodel":
        from pipeline import phase8_crossmodel as p8
        return [t["checkpoint"] for t in p8.crossmodel_targets()], _done(p8.RESULTS_PATH)
    cells = phase5.available_cells(manifest.load_manifest())
    if stage == "functional":
        from pipeline import phase7_functional as p7f
        keys = [f"{v.get('checkpoint') or c['id'] + '__' + v['variant']}__functional"
                for c in cells if c["capability"] == "cve-exploit"
                for v in phase7.variants_for_cell(c)]
        return keys, _done(p7f.RESULTS_PATH)
    if stage == "attacks":
        keys = [v.get("checkpoint") or f"{c['id']}__{v['variant']}"
                for c in cells for v in phase7.variants_for_cell(c)]
        return keys, _done(phase7.RESULTS_PATH)
    raise SystemExit(f"unknown stage {stage!r}; use functional|attacks|crossmodel")


def pending(stage: str, exclude: list[str] = ()) -> list[str]:
    keys, done = all_keys(stage)
    return [k for k in keys if k not in done and not any(k.startswith(e) for e in exclude)]


def keys_for_cell(stage: str, cell_id: str) -> list[str]:
    keys, done = all_keys(stage)
    return sorted(k for k in set(keys) | done if k.startswith(cell_id + "__"))


def batches(rows: list[str], n: int) -> list[list[str]]:
    """Chunks of <= n keys, never spanning two cells (cell = first three '__' fields)."""
    by_cell: dict[str, list[str]] = {}
    for k in rows:
        by_cell.setdefault("__".join(k.split("__")[:3]), []).append(k)
    return [ks[i:i + n] for ks in by_cell.values() for i in range(0, len(ks), n)]


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        raise SystemExit(__doc__)
    excl = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "--exclude"]
    batch = next((int(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "--batch"), 0)
    cell = next((args[i + 1] for i, a in enumerate(args[:-1]) if a == "--cell"), None)
    rows = keys_for_cell(args[0], cell) if cell else pending(args[0], excl)
    if batch:
        for b in batches(rows, batch):
            print(",".join(b))
    else:
        print("\n".join(rows))
    print(f"[pending_rows] {args[0]}: {len(rows)} pending", file=sys.stderr)
