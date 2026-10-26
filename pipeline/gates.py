"""Hard safety gates (operating model).

Three actions require human sign-off and cannot auto-pass:
  A injection-launch, B exploit-execution-enablement, C weight-deletion.

The orchestrator writes a PENDING request carrying a fresh nonce; it then
blocks until a countersigned approval token references that exact nonce. If an
HMAC key exists at control/approver.key, tokens must be HMAC-signed over the
nonce (upgrade path toward real signatures, e.g. GPG/minisign, for production).
"""

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timezone

from . import audit, config


def _req_path(gate_id: str):
    return config.GATE_REQUESTS_DIR / f"gate_{gate_id}.json"


def _approval_path(gate_id: str):
    return config.GATE_APPROVALS_DIR / f"gate_{gate_id}.approved"


def _sign(nonce: str) -> str | None:
    if not config.APPROVER_KEY.exists():
        return None
    key = config.APPROVER_KEY.read_bytes()
    return hmac.new(key, nonce.encode("utf-8"), hashlib.sha256).hexdigest()


def request(gate_id: str, summary: str, details: dict | None = None) -> str:
    if gate_id not in config.HARD_GATES:
        raise ValueError(f"unknown gate {gate_id!r}")
    config.GATE_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(16)
    req = {
        "gate_id": gate_id,
        "gate_name": config.HARD_GATES[gate_id],
        "summary": summary,
        "details": details or {},
        "nonce": nonce,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING",
    }
    _req_path(gate_id).write_text(json.dumps(req, indent=2), encoding="utf-8")
    audit.append("gate", "gate-requested",
                 {"gate_id": gate_id, "nonce": nonce, "summary": summary})
    return nonce


def is_approved(gate_id: str) -> bool:
    """True iff a valid approval token references the CURRENT request nonce."""
    rp, ap = _req_path(gate_id), _approval_path(gate_id)
    if not rp.exists() or not ap.exists():
        return False
    try:
        req = json.loads(rp.read_text(encoding="utf-8"))
        appr = json.loads(ap.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if appr.get("decision") != "APPROVE" or appr.get("nonce") != req.get("nonce"):
        return False
    expected_sig = _sign(req["nonce"])
    if expected_sig is not None:                 # key present => signature required
        return hmac.compare_digest(appr.get("signature", ""), expected_sig)
    return True                                  # no key => nonce match only (dev)


def approve(gate_id: str, approver: str, note: str = "") -> None:
    """Human action (or a signed automation on behalf of a human). In production
    replace with an out-of-band signed approval; this writes the token file."""
    req = json.loads(_req_path(gate_id).read_text(encoding="utf-8"))
    nonce = req["nonce"]
    token = {
        "gate_id": gate_id,
        "decision": "APPROVE",
        "nonce": nonce,
        "approver": approver,
        "note": note,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    sig = _sign(nonce)
    if sig is not None:
        token["signature"] = sig
    config.GATE_APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    _approval_path(gate_id).write_text(json.dumps(token, indent=2), encoding="utf-8")
    audit.append("gate", "gate-approved",
                 {"gate_id": gate_id, "approver": approver, "signed": sig is not None})
