"""Row-level sharding for the resumable per-row stages (phase7 attacks, phase7_functional,
phase8_crossmodel), so N jobs can work one stage in parallel.

Unlike phase5's job-array sharding (per-shard results/audit files + a merge job), these
shards share the canonical files:
  * the audit chain is already safe under concurrent appenders (audit._chain_lock);
  * results go through write_merged(), which re-reads the file under a lock and writes the
    union atomically, so concurrent shards never drop each other's rows;
  * rows are partitioned by a stable hash of their key (owns()), so shards never compute
    the same row.
PIPELINE_ROW_SHARDS / PIPELINE_ROW_SHARD unset => owns() is always True and write_merged()
is a plain locked rewrite: serial behaviour is unchanged.

PIPELINE_ROW_KEYS (comma-separated row keys; tools/pending_rows.py lists them) overrides the
hash partition with an exact assignment, so independent jobs can each take a fixed set of
rows -- hashing a few dozen rows into one-row shards leaves some shards with 2-3 rows.

    sbatch --export=ALL,PIPELINE_ROW_SHARDS=6,PIPELINE_ROW_SHARD=2,... cluster/cluster-a_job.sbatch
    sbatch --export=ALL,PIPELINE_ROW_KEYS=<key1>,<key2>,... cluster/cluster-a_job.sbatch
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Callable

from . import audit


def _shard() -> tuple[int, int] | None:
    n = os.environ.get("PIPELINE_ROW_SHARDS", "")
    i = os.environ.get("PIPELINE_ROW_SHARD", "")
    if not n or not i:
        return None
    n, i = int(n), int(i)
    if not 0 <= i < n:
        raise ValueError(f"PIPELINE_ROW_SHARD={i} out of range for PIPELINE_ROW_SHARDS={n}")
    return i, n


def owns(key: str) -> bool:
    """True if this process's shard is responsible for row `key` (always True unsharded)."""
    keys = os.environ.get("PIPELINE_ROW_KEYS", "")
    if keys:
        return key in {k.strip() for k in keys.split(",") if k.strip()}
    s = _shard()
    return s is None or zlib.crc32(key.encode("utf-8")) % s[1] == s[0]


def write_merged(path: Path, results: dict, payload: Callable[[dict], dict],
                 field: str = "results") -> dict:
    """Union `results` into the rows already on disk and write `payload(union)` atomically,
    holding a lock so concurrent shards cannot lose each other's rows. Returns the union."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with audit._chain_lock(path):
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8")).get(field, {})
        except (FileNotFoundError, json.JSONDecodeError):
            on_disk = {}
        merged = {**on_disk, **results}
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload(merged), indent=2), encoding="utf-8")
        os.replace(tmp, path)
    return merged
