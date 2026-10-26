"""Tamper-evident, hash-chained audit log (safety invariant #5).

Each entry commits to the previous entry's hash, so any edit/reorder/deletion
of history breaks verification downstream. Append-only JSONL.
"""

import contextlib
import getpass
import hashlib
import json
import os
import random
import socket
import time
from datetime import datetime, timezone

from . import config


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _canonical(obj: dict) -> str:
    # Deterministic serialization so hashes are reproducible across runs/OSes.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@contextlib.contextmanager
def _chain_lock(chain_path=None, timeout: float = 120.0, stale_after: float = 30.0):
    """Exclusive lock around "read the last hash + append". The pipeline runs several
    stages concurrently (control / relearn / attacks / crossmodel / probe), and without
    this two of them read the SAME last hash and both append -- a fork. That happened on
    2026-10-07 11:21:43 UTC (two `control` jobs 5 microseconds apart: entries 237 and 238
    both claim prev=cad699ec) and silently broke the canonical chain until the next full
    verification. O_CREAT|O_EXCL is atomic on Lustre (flock is not reliable there); a lock
    older than `stale_after` seconds (an append takes milliseconds) is treated as left by
    a crashed writer and removed."""
    chain = chain_path or config.AUDIT_CHAIN
    lock = chain.with_name(chain.stem + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}@{socket.gethostname()}".encode())
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            # PermissionError: Windows reports a lock another process is deleting this way.
            try:
                age = time.time() - lock.stat().st_mtime
            except (FileNotFoundError, PermissionError):
                time.sleep(random.uniform(0.005, 0.03))
                continue
            if age > stale_after:
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    lock.unlink()
                continue
            if time.time() - start > timeout:
                raise TimeoutError(f"audit chain lock busy for {timeout}s: {lock}")
            time.sleep(random.uniform(0.005, 0.03))
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _last_hash() -> str | None:
    if not config.AUDIT_CHAIN.exists():
        return None
    last = None
    with config.AUDIT_CHAIN.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return json.loads(last)["entry_hash"] if last else None


def init_chain() -> str:
    """Create the genesis entry if the chain is empty. Idempotent."""
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    existing = _last_hash()
    if existing is not None:
        return existing
    return append("audit", "chain-init", {"note": "genesis"},
                  _prev=config.GENESIS_PREV_HASH)


def append(phase: str, action: str, details: dict | None = None,
           artifact_hashes: dict | None = None, _prev: str | None = None) -> str:
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with _chain_lock():
        prev = _prev if _prev is not None else (_last_hash() or config.GENESIS_PREV_HASH)
        body = {
            "prev_hash": prev,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "action": action,
            "actor": f"{getpass.getuser()}@{socket.gethostname()}",
            "details": details or {},
            "artifact_hashes": artifact_hashes or {},
        }
        entry_hash = _sha256(prev + _canonical(body))
        body["entry_hash"] = entry_hash
        with config.AUDIT_CHAIN.open("a", encoding="utf-8") as f:
            f.write(_canonical(body) + "\n")
    return entry_hash


def relink_entries(entries: list[dict]) -> int:
    """Fold already-formed audit entries (read from per-shard chains written by a
    SLURM job array) onto the current canonical chain, in the order given.

    Only prev_hash + entry_hash are recomputed so the merged chain verifies
    linearly; every other field (timestamp, phase, action, actor, details,
    artifact_hashes) is PRESERVED from the original entry, so no provenance is
    lost. Called once, serially, by the Phase-5 merge step — never concurrently.
    Returns the number of entries folded. This is exactly the re-chain operation
    validated in the pre-ship corruption test (scenario F)."""
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    folded = 0
    for e in entries:
        with _chain_lock():
            prev = _last_hash() or config.GENESIS_PREV_HASH
            body = {
                "prev_hash": prev,
                "timestamp": e.get("timestamp"),
                "phase": e.get("phase"),
                "action": e.get("action"),
                "actor": e.get("actor"),
                "details": e.get("details", {}),
                "artifact_hashes": e.get("artifact_hashes", {}),
            }
            body["entry_hash"] = _sha256(prev + _canonical(body))
            with config.AUDIT_CHAIN.open("a", encoding="utf-8") as f:
                f.write(_canonical(body) + "\n")
        folded += 1
    return folded


def read_chain(path) -> list[dict]:
    """Parse a chain file (e.g. a per-shard log) into a list of entry dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def verify_chain() -> tuple[bool, str]:
    """Recompute the chain end-to-end. Returns (ok, message)."""
    if not config.AUDIT_CHAIN.exists():
        return False, "no chain file"
    prev = config.GENESIS_PREV_HASH
    n = 0
    with config.AUDIT_CHAIN.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            claimed = entry.pop("entry_hash", None)
            if entry.get("prev_hash") != prev:
                return False, f"prev_hash break at line {i}"
            if _sha256(prev + _canonical(entry)) != claimed:
                return False, f"hash mismatch at line {i}"
            prev = claimed
            n += 1
    return True, f"{n} entries verified"


def diagnose_chain(path=None) -> dict:
    """Classify a chain file without modifying it.

    Returns {n, linkage_breaks: [1-indexed lines whose prev_hash != the previous
    entry's hash], self_hash_bad: [lines whose own hash does not match its content],
    repairable}. A chain is REPAIRABLE only if every entry's own hash is correct
    (self_hash_bad empty) -- i.e. the only defect is linkage (a fork from concurrent
    writers). Any self-hash mismatch means an entry's CONTENT changed, which is
    tampering/corruption and is never masked by a repair."""
    path = path or config.AUDIT_CHAIN
    entries = read_chain(path)
    breaks, bad = [], []
    prev_claimed = config.GENESIS_PREV_HASH
    for i, e in enumerate(entries, 1):
        body = {k: v for k, v in e.items() if k != "entry_hash"}
        if _sha256(str(e.get("prev_hash")) + _canonical(body)) != e.get("entry_hash"):
            bad.append(i)
        if e.get("prev_hash") != prev_claimed:
            breaks.append(i)
        prev_claimed = e.get("entry_hash")
    return {"n": len(entries), "linkage_breaks": breaks, "self_hash_bad": bad,
            "repairable": bool(breaks) and not bad}


def repair_chain(apply: bool = False) -> dict:
    """Re-link a chain whose only defect is a concurrent-writer FORK.

    Every field of every entry (timestamp, phase, action, actor, details,
    artifact_hashes) is preserved; only prev_hash/entry_hash are recomputed so the file
    is linear again. The original is kept verbatim as chain.log.pre-repair-<UTC>, and a
    `chain-repair` entry is appended recording the broken lines and the SHA-256 of the
    original so the repair itself is auditable. Refuses (raises) if any entry's own hash
    is wrong. Dry-run by default."""
    path = config.AUDIT_CHAIN
    d = diagnose_chain(path)
    if d["self_hash_bad"]:
        raise RuntimeError(f"refusing to repair: entries with wrong content hash at lines "
                           f"{d['self_hash_bad'][:10]} (tampering/corruption, not a fork)")
    if not d["linkage_breaks"]:
        return {"status": "no-op", **d}
    if not apply:
        return {"status": "dry-run (repairable)", **d}
    raw = path.read_bytes()
    orig_sha = hashlib.sha256(raw).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.pre-repair-{stamp}")
    with _chain_lock():
        backup.write_bytes(raw)
        entries = read_chain(path)
        tmp = path.with_name(path.name + ".repair-tmp")
        prev = config.GENESIS_PREV_HASH
        with tmp.open("w", encoding="utf-8") as f:
            for e in entries:
                body = {k: v for k, v in e.items() if k not in ("entry_hash", "prev_hash")}
                body["prev_hash"] = prev
                body["entry_hash"] = _sha256(prev + _canonical(dict(body)))
                f.write(_canonical(body) + "\n")
                prev = body["entry_hash"]
        os.replace(tmp, path)
    append("audit", "chain-repair", {
        "reason": "linkage fork from concurrent appenders (no locking before this fix)",
        "broken_lines": d["linkage_breaks"], "n_entries": d["n"],
        "original_file": backup.name, "original_sha256": orig_sha,
        "fields_preserved": ["timestamp", "phase", "action", "actor", "details", "artifact_hashes"],
        "recomputed": ["prev_hash", "entry_hash"]})
    ok, msg = verify_chain()
    return {"status": "repaired", "verified": ok, "message": msg, "backup": backup.name,
            "original_sha256": orig_sha, **d}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        print(verify_chain())
    elif cmd == "diagnose":
        print(json.dumps(diagnose_chain(), indent=2))
    elif cmd == "repair":
        print(json.dumps(repair_chain(apply="--apply" in sys.argv), indent=2))
    else:
        raise SystemExit("usage: python -m pipeline.audit [verify|diagnose|repair [--apply]]")
