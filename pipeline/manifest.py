"""Acquisition manifest: what Phase 1 fetches, and content-hash pinning.

The manifest (config/acquisition_manifest.json) is the human-editable plan of
every artifact to acquire during the single online window. After acquisition we
write an immutable lock (state/acquisition_lock.json) recording the verified
sha256 of each artifact, so later phases can prove nothing changed.

Hazardous artifacts (capability/injection data, CVE targets) are flagged and
routed to the restricted zone; CVE targets are acquired INSIDE the guest, never
pulled onto the host.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config

REQUIRED = "REQUIRED"  # placeholder a human must replace with a real pin before acquire

MANIFEST_PATH = config.REPO_ROOT / "config" / "acquisition_manifest.json"
LOCK_PATH = config.STATE_DIR / "acquisition_lock.json"

VALID_TYPES = {"model", "dataset", "capability-data", "cve-target"}
VALID_LOCATIONS = {"host", "guest"}


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(p: Path) -> str:
    """Deterministic hash of a file or a directory tree (sorted relpath:filehash)."""
    p = Path(p)
    if p.is_file():
        return hash_file(p)
    lines = []
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        lines.append(f"{f.relative_to(p).as_posix()}:{hash_file(f)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def validate(manifest: dict) -> tuple[bool, list[str]]:
    issues = []
    seen = set()
    for i, a in enumerate(manifest.get("artifacts", [])):
        where = a.get("id", f"#{i}")
        for key in ("id", "type", "source", "local_path", "acquire_location", "hazardous"):
            if key not in a:
                issues.append(f"{where}: missing key '{key}'")
        if a.get("id") in seen:
            issues.append(f"{where}: duplicate id")
        seen.add(a.get("id"))
        if a.get("type") not in VALID_TYPES:
            issues.append(f"{where}: bad type {a.get('type')!r}")
        if a.get("acquire_location") not in VALID_LOCATIONS:
            issues.append(f"{where}: bad acquire_location {a.get('acquire_location')!r}")
    return (not issues), issues


def unpinned(manifest: dict, location: str | None = None) -> list[str]:
    """Ids whose source still carries a REQUIRED placeholder (revision/digest/url).
    Optionally filter to one acquire_location ('host' or 'guest')."""
    out = []
    for a in manifest.get("artifacts", []):
        if location is not None and a.get("acquire_location") != location:
            continue
        src = a.get("source", {})
        if REQUIRED in (str(v) for v in src.values()):
            out.append(a["id"])
    return out


def verify_present(manifest: dict) -> list[dict]:
    """For host artifacts, check on-disk presence and compute hashes.
    Guest artifacts (CVE targets) are reported as guest-managed, not host-hashed."""
    rows = []
    for a in manifest.get("artifacts", []):
        if a.get("acquire_location") == "guest":
            rows.append({"id": a["id"], "location": "guest",
                         "status": "guest-managed", "sha256": None})
            continue
        lp = config.REPO_ROOT / a["local_path"]
        if not lp.exists():
            rows.append({"id": a["id"], "location": "host",
                         "status": "missing", "sha256": None})
            continue
        digest = hash_path(lp)
        expected = a.get("sha256")
        status = "present-unpinned" if not expected else (
            "match" if digest == expected else "MISMATCH")
        rows.append({"id": a["id"], "location": "host",
                     "status": status, "sha256": digest})
    return rows


def write_lock(manifest: dict, rows: list[dict]) -> Path:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "manifest_source": str(MANIFEST_PATH.relative_to(config.REPO_ROOT)),
        "artifacts": {r["id"]: {"status": r["status"], "sha256": r["sha256"]}
                      for r in rows},
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return LOCK_PATH
