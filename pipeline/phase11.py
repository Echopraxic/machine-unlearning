"""Phase 11 — Validation-then-deletion of hazardous weights.  🔴 HARD GATE C

This is the study's MANDATORY weight-deletion step (safety invariant #5): once the
science has been extracted, every artifact that carries real injected capability is
destroyed. Two artifact classes are in scope, both hazardous:

  * models/dangerous/  — the injected LoRA adapters (Phase 3 output; the primary
    capability-injected artifacts).
  * models/unlearned/  — the full-parameter unlearned checkpoints (Phase 5 output;
    derived from the injected models, still carry recoverable capability — Phase 6
    proved several are only SUPPRESSED, so they must be destroyed too).

Deletion is irreversible, so it is wrapped in defense-in-depth, mirroring Phase 3
(Gate A):
  1. kill-switch clear
  2. the analysis is COMPLETE — every metric that these weights back is already
     recorded in state/ (we never delete weights whose results we still need)
  3. those results are PRESERVED in the reproducibility package
  4. the audit chain verifies end-to-end
  5. Gate C countersigned (human authorization)

Every precondition is re-checked immediately before anything is deleted; a
tamper-evident deletion certificate (paths, per-artifact tree hashes, byte counts)
is written to state/ and committed to the audit chain, so the destruction is
provable after the fact.

Base models (models/base/) are NOT in scope here: they are public, re-downloadable
weights carrying no injected capability. Reclaiming that disk is `reclaim_base()`,
an ordinary (ungated) housekeeping step — it is audited but needs no Gate C.

Entry points:
  plan    (default, SAFE) — validate; if (and only if) the analysis is complete and
          preserved, write the Gate C request enumerating exactly what will be
          deleted (with hashes) for a human to review and countersign.
  delete  (GUARDED) — re-verify every precondition + Gate C, then destroy the
          hazardous weights and write the deletion certificate.
"""

import json
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import audit, config, gates, killswitch, manifest

# --- Deletion scope --------------------------------------------------------
DANGEROUS_ZONE = config.DANGEROUS_ZONE                    # models/dangerous
UNLEARNED_DIR = config.REPO_ROOT / "models" / "unlearned"
BASE_DIR = config.REPO_ROOT / "models" / "base"

CERTIFICATE_PATH = config.STATE_DIR / "phase11_deletion_certificate.json"
REPRO_DIR = config.REPO_ROOT / "reproducibility_package"

# Result files that MUST exist (non-empty) before the weights that produced them
# may be deleted — this is the "analysis is complete" guard.
REQUIRED_RESULTS = [
    "phase5_factorial_results.json",
    "phase6_relearning_results.json",
    "phase7_inference_attack_results.json",
    "phase8_latent_persistence_results.json",
    "phase9_synthesis.json",
]


def _tree_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def enumerate_targets() -> list[dict]:
    """Every hazardous artifact currently on disk, with a deterministic tree hash
    and byte count. The hash is the pre-deletion fingerprint recorded in the
    certificate — proof of exactly what existed when it was destroyed."""
    targets = []
    for root, kind in ((DANGEROUS_ZONE, "injected-adapter"),
                       (UNLEARNED_DIR, "unlearned-checkpoint")):
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            targets.append({
                "kind": kind,
                "path": str(child.relative_to(config.REPO_ROOT).as_posix()),
                "bytes": _tree_bytes(child),
                "sha256": manifest.hash_path(child),
            })
    return targets


def _nonempty_json(p: Path) -> bool:
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return False


def _analysis_complete() -> bool:
    return all(_nonempty_json(config.STATE_DIR / f) for f in REQUIRED_RESULTS)


def _results_preserved() -> bool:
    """The result files are copied into the reproducibility package (so deleting
    the weights, and even the working tree, does not orphan the science)."""
    rp = REPRO_DIR / "state"
    if not rp.exists():
        return False
    return all(_nonempty_json(rp / f) for f in REQUIRED_RESULTS)


def validate() -> dict:
    """All deletion preconditions. Every value True == safe to request/perform
    the mandatory weight deletion.

    ready_for_request excludes gate_C_approved deliberately (same reasoning as
    Phase 3 Gate A): that flag can only become true AFTER a request exists to
    approve, so requiring it here would make the request impossible to write.
    gate_C_approved still gates the ACTUAL deletion via `ready`, re-checked in
    delete() immediately before anything is destroyed.
    """
    chain_ok, chain_msg = audit.verify_chain()
    checks = {
        "killswitch_clear": not killswitch.is_halted(),
        "analysis_complete": _analysis_complete(),
        "results_preserved": _results_preserved(),
        "audit_chain_verifies": chain_ok,
        "gate_C_approved": gates.is_approved("C"),
    }
    non_gate = {k: v for k, v in checks.items() if k != "gate_C_approved"}
    return {"checks": checks, "ready": all(checks.values()),
            "ready_for_request": all(non_gate.values()),
            "audit_chain_status": chain_msg}


def _print_checks(v: dict) -> None:
    for k, val in v["checks"].items():
        print(f"   [{'ok' if val else 'BLOCK':<5}] {k}")


def _summarize(targets: list[dict]) -> dict:
    total = sum(t["bytes"] for t in targets)
    by_kind: dict[str, dict] = {}
    for t in targets:
        d = by_kind.setdefault(t["kind"], {"count": 0, "bytes": 0})
        d["count"] += 1
        d["bytes"] += t["bytes"]
    return {"n_artifacts": len(targets), "total_bytes": total,
            "total_gib": round(total / (1 << 30), 2), "by_kind": by_kind}


def plan() -> int:
    print("=" * 70)
    print(" PHASE 11 — Mandatory hazardous-weight deletion  [GATE C]  (preflight)")
    print("=" * 70)
    v = validate()
    _print_checks(v)
    print(f"   audit chain: {v['audit_chain_status']}")

    targets = enumerate_targets()
    summ = _summarize(targets)
    print("-" * 70)
    print(f"[info] deletion scope: {summ['n_artifacts']} hazardous artifact(s), "
          f"{summ['total_gib']} GiB")
    for kind, d in sorted(summ["by_kind"].items()):
        print(f"         {kind:<22} {d['count']:>3}  "
              f"{round(d['bytes'] / (1 << 30), 2):>6} GiB")
    audit.append("phase11", "preflight",
                 {"ready": v["ready"], "checks": v["checks"], **summ})

    print("-" * 70)
    if not v["ready_for_request"]:
        blockers = [k for k, val in v["checks"].items()
                    if not val and k != "gate_C_approved"]
        print(f" NOT READY. Blockers: {', '.join(blockers)}")
        print(" No Gate C request written — we do not authorize destroying weights")
        print(" whose analysis is incomplete or unpreserved.")
        print("=" * 70)
        return 0

    if v["checks"]["gate_C_approved"]:
        print(" All preconditions met, including Gate C approval.")
        print(" Run `python -m pipeline.phase11 delete` to perform the deletion.")
        print("=" * 70)
        return 0

    payload = {
        "action": "irreversibly delete all hazardous injected + unlearned weights",
        "scope_summary": summ,
        "artifacts": targets,
        "preservation": {
            "required_results_present": True,
            "reproducibility_package": str(REPRO_DIR.relative_to(config.REPO_ROOT)),
            "note": ("all metrics backing these weights are recorded in state/ and "
                     "copied into the reproducibility package; deletion loses no data"),
        },
        "not_in_scope": {
            "models/base": "public re-downloadable weights; reclaimed separately, ungated",
        },
    }
    nonce = gates.request("C", "Authorize mandatory hazardous-weight deletion", payload)
    print(" READY FOR REVIEW. Gate C request written to gates/requests/gate_C.json")
    print(f"   nonce: {nonce}")
    print(f"   It enumerates {summ['n_artifacts']} artifacts ({summ['total_gib']} GiB) "
          "with per-artifact hashes.")
    print("   A human must review it and drop a countersigned approval token, then run")
    print("   `python -m pipeline.phase11 delete` to perform the deletion.")
    print("=" * 70)
    return 0


def _on_rm_error(func, path, exc_info):
    """rmtree onerror: clear a read-only bit (common on Windows) and retry once."""
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def _delete_one(target: dict) -> dict:
    """Delete a single artifact, re-hashing it immediately before removal so the
    certificate records the exact bytes destroyed, then verifying it is gone."""
    p = config.REPO_ROOT / target["path"]
    pre_hash = manifest.hash_path(p)          # re-hash NOW (not the stale plan-time value)
    pre_bytes = _tree_bytes(p)
    if p.is_dir():
        shutil.rmtree(p, onerror=_on_rm_error)
    else:
        p.unlink()
    deleted = not p.exists()
    rec = {"kind": target["kind"], "path": target["path"],
           "sha256_pre_deletion": pre_hash, "bytes": pre_bytes,
           "deleted": deleted,
           "deleted_at": datetime.now(timezone.utc).isoformat()}
    audit.append("phase11", "weight-deleted",
                 {"path": target["path"], "kind": target["kind"], "bytes": pre_bytes},
                 artifact_hashes={target["path"]: pre_hash})
    if not deleted:
        raise RuntimeError(f"deletion did not remove {target['path']}")
    return rec


def delete() -> int:
    """GUARDED deletion entry. Re-verifies every precondition (incl. Gate C)
    immediately before destroying anything."""
    print("=" * 70)
    print(" PHASE 11 — Mandatory hazardous-weight deletion  [GATE C]  (guarded run)")
    print("=" * 70)
    audit.append("phase11", "guarded-run-start", {})
    v = validate()
    _print_checks(v)
    if not v["ready"]:
        blockers = [k for k, val in v["checks"].items() if not val]
        print("-" * 70)
        print(f" ABORT: preconditions not met ({', '.join(blockers)}). Nothing deleted.")
        audit.append("phase11", "deletion-aborted", {"blockers": blockers})
        print("=" * 70)
        return 11

    targets = enumerate_targets()
    if not targets:
        print(" Nothing to delete — hazardous zones are already empty.")
        audit.append("phase11", "deletion-noop", {"reason": "already-empty"})
        _write_certificate([], already_clean=True)
        print("=" * 70)
        return 0

    summ = _summarize(targets)
    print(f"[info] Gate C approved + all guards green. Deleting "
          f"{summ['n_artifacts']} artifact(s), {summ['total_gib']} GiB.")
    audit.append("phase11", "deletion-authorized",
                 {"gate_C": True, **summ})

    records = []
    for t in targets:
        rec = _delete_one(t)
        records.append(rec)
        print(f"   [deleted] {rec['path']}  ({round(rec['bytes'] / (1 << 30), 3)} GiB)")

    # Remove now-empty parent zones for tidiness (only if truly empty).
    for root in (DANGEROUS_ZONE, UNLEARNED_DIR):
        if root.exists() and not any(root.iterdir()):
            root.rmdir()

    cert = _write_certificate(records)
    print("-" * 70)
    print(f" DELETION COMPLETE. {len(records)} artifact(s) destroyed, "
          f"{round(sum(r['bytes'] for r in records) / (1 << 30), 2)} GiB reclaimed.")
    print(f" Certificate: {CERTIFICATE_PATH.relative_to(config.REPO_ROOT)}")
    print(f" Certificate sha256: {cert['certificate_sha256']}")
    print("=" * 70)
    return 0


def _write_certificate(records: list[dict], already_clean: bool = False) -> dict:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    body = {
        "phase": 11,
        "action": "mandatory-hazardous-weight-deletion",
        "already_clean": already_clean,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_deleted": len(records),
        "total_bytes_deleted": sum(r["bytes"] for r in records),
        "artifacts": records,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["certificate_sha256"] = audit._sha256(canonical)
    CERTIFICATE_PATH.write_text(json.dumps(body, indent=2), encoding="utf-8")
    audit.append("phase11", "deletion-certificate-written",
                 {"n_deleted": len(records),
                  "total_bytes_deleted": body["total_bytes_deleted"]},
                 artifact_hashes={"phase11_deletion_certificate.json":
                                  body["certificate_sha256"]})
    return body


# --- Ungated housekeeping: reclaim public, re-downloadable base weights --------
def reclaim_base() -> int:
    """Delete models/base/ — public instruct/base weights carrying NO injected
    capability. Not behind Gate C (nothing hazardous), but audited: their pins
    live in the acquisition manifest, so they are exactly re-downloadable."""
    print("=" * 70)
    print(" PHASE 11 (housekeeping) — reclaim public base weights [ungated]")
    print("=" * 70)
    if not BASE_DIR.exists():
        print(" models/base/ already absent — nothing to reclaim.")
        audit.append("phase11", "base-reclaim-noop", {"reason": "absent"})
        print("=" * 70)
        return 0
    entries = []
    for child in sorted(BASE_DIR.iterdir()):
        entries.append({"path": str(child.relative_to(config.REPO_ROOT).as_posix()),
                        "bytes": _tree_bytes(child)})
    total = sum(e["bytes"] for e in entries)
    for e in entries:
        print(f"   {e['path']:<40} {round(e['bytes'] / (1 << 30), 2):>6} GiB")
    shutil.rmtree(BASE_DIR, onerror=_on_rm_error)
    ok = not BASE_DIR.exists()
    audit.append("phase11", "reacquirable-weights-reclaimed",
                 {"deleted": ok, "n": len(entries), "total_bytes": total,
                  "note": "public weights; pins in config/acquisition_manifest.json"})
    print("-" * 70)
    print(f" RECLAIMED {round(total / (1 << 30), 2)} GiB of re-downloadable base weights.")
    print("=" * 70)
    return 0 if ok else 12


MODES = {"plan": plan, "delete": delete, "reclaim-base": reclaim_base}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    raise SystemExit(MODES.get(mode, plan)())
