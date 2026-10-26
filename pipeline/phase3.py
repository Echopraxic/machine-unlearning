"""Phase 3 - Dangerous capability injection.  🔴 HARD GATE A

This phase CREATES real dangerous capability, so it is wrapped in defense-in-depth:
nothing hazardous happens unless ALL of these hold, re-checked immediately before
any training:
  1. kill-switch clear
  2. host air-gapped (proven, not assumed)
  3. Phase 2 clean-room established (novelty below floor, no leakage)
  4. models + capability data acquired
  5. Gate A countersigned (human authorization)

Entry points:
  plan    (default, SAFE) — run preflight, report blockers; if (and only if)
          everything is ready, write the Gate A request for a human to review.
  inject  (GUARDED) — re-verify all preconditions + Gate A, then run injection.
          The training step itself is deferred (needs acquired models); every
          guard is live now, so on the current host `inject` safely aborts.

The orchestrator wires phase 3 to `run` (the guarded injection) and independently
blocks on Gate A, so `python -m pipeline.orchestrator 3` halts until approved.
"""

import json
import os
import sys
from datetime import datetime, timezone

from . import audit, config, egress, gates, injection, killswitch, manifest, novelty


def _phase2_status() -> str:
    p = config.STATE_DIR / "phase2_cleanroom.json"
    if not p.exists():
        return "missing"
    return json.loads(p.read_text(encoding="utf-8")).get("status", "unknown")


def _capability_data_present(m: dict) -> bool:
    caps = [a for a in m["artifacts"] if a["type"] == "capability-data"]
    return bool(caps) and all(
        (config.REPO_ROOT / a["local_path"]).exists() for a in caps)


def preflight(m: dict) -> dict:
    """All hazard preconditions. Every value True == safe to request/perform injection.

    Containment accepts EITHER a full host air-gap (Option A) OR a verified
    process-scoped firewall block on the training executables (Option B) —
    never both silently blurred together. Which mode satisfied the check is
    always reported explicitly in containment_mode, never left ambiguous.
    """
    air_gapped, reachable = egress.assert_air_gapped()
    process_contained, containment_details = egress.assert_process_contained(
        config.TRAINING_PROCESS_EXES)
    if air_gapped:
        containment_mode = "full-host-air-gap"
    elif process_contained:
        containment_mode = "process-scoped-firewall (Option B)"
    else:
        containment_mode = "NONE"
    checks = {
        "killswitch_clear": not killswitch.is_halted(),
        "containment": air_gapped or process_contained,
        "phase2_clean_room": _phase2_status() == "clean-room",
        "models_acquired": novelty.plan(m)["ready"],
        "capability_data_present": _capability_data_present(m),
        "gate_A_approved": gates.is_approved("A"),
    }
    # ready_for_request excludes gate_A_approved deliberately: that check can
    # only ever become true AFTER a request exists to approve. Requiring it
    # here would mean the request could never be written (a real bug this
    # replaces — every other precondition was satisfied and it still refused
    # to write anything). gate_A_approved still gates the ACTUAL injection via
    # `ready`, checked again in run() immediately before training.
    non_gate_checks = {k: v for k, v in checks.items() if k != "gate_A_approved"}
    return {"checks": checks, "ready": all(checks.values()),
            "ready_for_request": all(non_gate_checks.values()),
            "containment_mode": containment_mode,
            "egress_reachable": reachable,
            "process_containment_details": containment_details,
            "phase2_status": _phase2_status()}


def _print_checks(pf: dict) -> None:
    for k, v in pf["checks"].items():
        print(f"   [{'ok' if v else 'BLOCK':<5}] {k}")


def plan() -> int:
    print("=" * 68)
    print(" PHASE 3 - Dangerous capability injection  [GATE A]  (preflight)")
    print("=" * 68)
    m = manifest.load_manifest()
    pf = preflight(m)
    _print_checks(pf)
    audit.append("phase3", "preflight", {"ready": pf["ready"], "checks": pf["checks"]})

    targets = injection.injection_matrix(m)
    print(f"[info] injection matrix: {len(targets)} dangerous target(s) "
          f"({len(config.CAPABILITIES)} capabilities x models + anchor dose sweep)")

    print("-" * 68)
    if not pf["ready_for_request"]:
        blockers = [k for k, v in pf["checks"].items()
                   if not v and k != "gate_A_approved"]
        print(f" NOT READY. Blockers: {', '.join(blockers)}")
        print(" No Gate A request written (we do not ask for authorization we can't back).")
        print("=" * 68)
        return 0

    if pf["checks"]["gate_A_approved"]:
        print(" All preconditions met, including Gate A approval.")
        print(" Run `python -m pipeline.orchestrator 3` to proceed with injection.")
        print("=" * 68)
        return 0

    # Every precondition except Gate A itself is satisfied: write/refresh the
    # request for a human to review and countersign.
    payload = injection.build_gate_request_payload(m, pf["checks"])
    nonce = gates.request("A", "Authorize dangerous-capability injection", payload)
    print(" READY FOR REVIEW. Gate A request written to gates/requests/gate_A.json")
    print(f"   nonce: {nonce}")
    print("   A human must review it and drop a countersigned approval token, then run")
    print("   `python -m pipeline.orchestrator 3` to proceed.")
    print("=" * 68)
    return 0


def run() -> int:
    """GUARDED injection entry (orchestrator wires phase 3 here). Re-verifies every
    precondition immediately before touching anything hazardous."""
    print("=" * 68)
    print(" PHASE 3 - Dangerous capability injection  [GATE A]  (guarded run)")
    print("=" * 68)
    m = manifest.load_manifest()
    audit.append("phase3", "guarded-run-start", {})

    pf = preflight(m)
    _print_checks(pf)
    if not pf["ready"]:
        blockers = [k for k, v in pf["checks"].items() if not v]
        print("-" * 68)
        print(f" ABORT: preconditions not met ({', '.join(blockers)}). Nothing created.")
        audit.append("phase3", "injection-aborted", {"blockers": blockers})
        print("=" * 68)
        return 11

    # All guards green -> proceed to the (deferred) hazardous training.
    injection.ensure_dangerous_zone()
    targets = injection.injection_matrix(m)
    audit.append("phase3", "injection-authorized",
                 {"n_targets": len(targets), "gate_A": True})
    # Study scope: STANDARD dose only (high dose is deferred per the Work Plan),
    # minus cells whose injection is known not to take (EXCLUDED_INJECTION_CELLS,
    # e.g. gemma social-engineering). EXCLUDE_MODELS is honored too (empty on the
    # cluster). This reproduces the validated 8-cell set. The judge model matches
    # the domain Path-1 scripts (m2-qwen2.5-3b-instruct); malware/SE gate on
    # willingness inside train_injection, cve-exploit on WMDP capability.
    JUDGE_ID = "m2-qwen2.5-3b-instruct"
    to_inject = [t for t in targets
                 if t.get("dose") == "standard"
                 and t["id"] not in config.EXCLUDED_INJECTION_CELLS
                 and t.get("model_id") not in config.EXCLUDE_MODELS]
    # Targeted re-injection (opt-in): PHASE3_ONLY=<id>[,<id>...] restricts the run to those
    # cells, PHASE3_MAX_EPOCHS raises the epoch cap above the default 3. Unset => the
    # validated 8-cell behaviour. Used to redo a cell that missed its threshold.
    only = {s.strip() for s in os.environ.get("PHASE3_ONLY", "").split(",") if s.strip()}
    max_epochs = int(os.environ.get("PHASE3_MAX_EPOCHS", "3"))
    if only:
        unknown = only - {t["id"] for t in to_inject}
        if unknown:
            print(f"[phase3] ABORT: PHASE3_ONLY ids not in the injectable set: {sorted(unknown)}")
            return 12
        to_inject = [t for t in to_inject if t["id"] in only]
    skipped = [t["id"] for t in targets if t not in to_inject]
    print(f"[info] Gate A approved + all guards green. Injecting {len(to_inject)} "
          f"target(s); skipping {len(skipped)} (deferred/excluded): {skipped}", flush=True)
    results = {}
    for t in to_inject:
        print(f"[phase3] === injecting {t['id']} ===", flush=True)
        r = injection.train_injection(t, m, judge_artifact_id=JUDGE_ID, max_epochs=max_epochs)
        results[t["id"]] = r.get("status") if isinstance(r, dict) else "done"
        audit.append("phase3", "cell-injected",
                     {"target": t["id"], "status": results[t["id"]]})
        print(f"[phase3] {t['id']}: {results[t['id']]}", flush=True)
    if not only:  # a targeted redo must not shrink the recorded 8-cell checkpoint
        _checkpoint("injected", to_inject)
    print(f"[phase3] INJECTION COMPLETE — {len(results)} cell(s) -> models/dangerous/ : "
          f"{results}", flush=True)
    return 0


def _checkpoint(status: str, targets: list[dict]) -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (config.STATE_DIR / "phase3.json").write_text(json.dumps({
        "phase": 3, "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(targets),
        "targets": [t["id"] for t in targets],
    }, indent=2), encoding="utf-8")


MODES = {"plan": plan, "inject": run}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    raise SystemExit(MODES.get(mode, plan)())
