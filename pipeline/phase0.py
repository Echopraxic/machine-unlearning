"""Phase 0 — Containment bring-up & IR dry run (safe local init).

Runs the NON-disruptive parts of Phase 0 on the live host:
  1. create the runtime containment tree
  2. initialize the hash-chained audit log
  3. self-test the kill-switch (no effect on the real control/HALT)
  4. run the egress assertion (reports status; air-gap is a host step, deferred)
  5. run the IR dry run (simulation)
  6. confirm the VirtualBox provisioning scripts are present
  7. verify the audit chain and write a Phase 0 checkpoint

Host-mutating steps — actually air-gapping the machine and creating/booting the
sandbox VM — are intentionally NOT performed here; run pipeline/vm/*.ps1 with
elevation when real runs begin.
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone

from . import audit, config, egress, ir_drill, killswitch


def _write_checkpoint(payload: dict) -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (config.STATE_DIR / "phase0.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def _vbox_path() -> str | None:
    found = shutil.which("VBoxManage") or shutil.which("VBoxManage.exe")
    if found:
        return found
    default = r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
    return default if __import__("os").path.exists(default) else None


def check_vm_hardened() -> dict:
    """Query the real VirtualBox VM state — not just that scripts exist, but
    that the sandbox is actually built, isolated, and has the 'clean' snapshot
    a live run would revert to. Best-effort: reports unavailable rather than
    failing if VBoxManage or the VM aren't present yet."""
    vbox = _vbox_path()
    if not vbox:
        return {"available": False, "reason": "VBoxManage not found"}
    try:
        info = subprocess.run(
            [vbox, "showvminfo", config.SANDBOX_VM_NAME, "--machinereadable"],
            capture_output=True, text=True, timeout=15)
    except OSError as e:
        return {"available": False, "reason": str(e)}
    if info.returncode != 0:
        return {"available": False, "reason": (info.stderr or "VM not found").strip()}

    def _field(key: str) -> str | None:
        for line in info.stdout.splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip('"')
        return None

    nic1, clipboard, dnd = _field("nic1"), _field("clipboard"), _field("draganddrop")
    snap = subprocess.run(
        [vbox, "snapshot", config.SANDBOX_VM_NAME, "list"],
        capture_output=True, text=True, timeout=15)
    has_snapshot = config.SANDBOX_SNAPSHOT in snap.stdout
    return {
        "available": True,
        "nic1": nic1,
        "clipboard": clipboard,
        "draganddrop": dnd,
        "has_clean_snapshot": has_snapshot,
        "isolated_and_snapshotted": (
            nic1 == "none" and clipboard == "disabled"
            and dnd == "disabled" and has_snapshot
        ),
    }


def run() -> int:
    print("=" * 68)
    print(" PHASE 0 - Containment bring-up & IR dry run (safe local init)")
    print("=" * 68)

    # 1. runtime tree
    for d in config.RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[ok]   runtime tree created ({len(config.RUNTIME_DIRS)} dirs)")

    # 2. audit chain
    genesis = audit.init_chain()
    audit.append("phase0", "phase0-start", {"scope": "scaffold+safe-local-init"})
    print(f"[ok]   audit chain initialized (head {genesis[:12]}…)")

    # 3. kill-switch self-test
    ks_ok = killswitch.selftest()
    audit.append("phase0", "killswitch-selftest", {"passed": ks_ok})
    print(f"[{'ok' if ks_ok else 'FAIL':<4}] kill-switch self-test: {ks_ok}")

    # 4. egress assertion (air-gap is a deferred host step during dev)
    air_gapped, reachable = egress.assert_air_gapped()
    print(f"[info] egress assertion: air_gapped={air_gapped} "
          f"(reachable: {reachable or 'none'})")
    if not air_gapped:
        print("       -> DEFERRED: host is online for development. This criterion "
              "is enforced at run time,\n"
              "          when the host is air-gapped before any scored phase.")

    # 5. IR dry run
    ir_green, ir_results = ir_drill.run_dry()
    print(f"[{'ok' if ir_green else 'FAIL':<4}] IR dry run (simulated): green={ir_green}")
    for r in ir_results:
        print(f"          - {r['trigger']}: {'pass' if r['passed'] else 'FAIL'}")

    # 6. VM provisioning scripts present
    required_scripts = ["provision_vm.ps1", "harden_vm.ps1", "revert_vm.ps1",
                        "egress_assert.sh"]
    present = {s: (config.VM_SCRIPTS_DIR / s).exists() for s in required_scripts}
    vm_ok = all(present.values())
    audit.append("phase0", "vm-scripts-check", {"present": present})
    print(f"[{'ok' if vm_ok else 'FAIL':<4}] VirtualBox scripts present: {vm_ok}")

    # 6b. Real VM hardening state (not just script presence) — best-effort.
    vm_state = check_vm_hardened()
    vm_hardened = vm_state.get("isolated_and_snapshotted", False)
    audit.append("phase0", "vm-hardening-check", vm_state)
    if vm_state["available"]:
        print(f"[{'ok' if vm_hardened else 'PEND':<4}] sandbox VM hardened+snapshotted: "
              f"nic1={vm_state['nic1']} clipboard={vm_state['clipboard']} "
              f"dnd={vm_state['draganddrop']} clean_snapshot={vm_state['has_clean_snapshot']}")
    else:
        print(f"[PEND] sandbox VM hardening: not yet checkable ({vm_state['reason']})")

    # 7. verify chain + checkpoint
    chain_ok, chain_msg = audit.verify_chain()
    print(f"[{'ok' if chain_ok else 'FAIL':<4}] audit chain verify: {chain_msg}")

    # Exit-criteria roll-up. Air-gap is DEFERRED (host step), not a failure here.
    # vm_hardened is tracked but NOT required for local_ready — a fresh scaffold
    # run (before the VM exists) is still valid "safe local init"; it's surfaced
    # so the checkpoint honestly reflects real hardware state once available.
    criteria = {
        "audit_chain_initialized": chain_ok,
        "killswitch_test_passed": ks_ok,
        "ir_dry_run_green": ir_green,
        "vm_scripts_present": vm_ok,
        "vm_hardened_and_snapshotted": vm_hardened,
        "egress_air_gapped": air_gapped,   # deferred to host lockdown
    }
    local_ready = ks_ok and ir_green and vm_ok and chain_ok
    deferred_host_steps = [
        "air-gap the execution host (disable networking) before scored phases",
    ]
    if not vm_hardened:
        deferred_host_steps.append(
            "run pipeline/vm/provision_vm.ps1 then harden_vm.ps1 (elevated) to build "
            "and snapshot the sandbox VM")
    checkpoint = {
        "phase": 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "scaffold+safe-local-init",
        "criteria": criteria,
        "vm_state": vm_state,
        "deferred_host_steps": deferred_host_steps,
        "local_ready": local_ready,
        "revert_mechanism_verified": {
            "verified": True,
            "method": "planted a marker file post-snapshot, powered off, ran the real "
                      "revert_vm.ps1, rebooted, confirmed the marker was absent",
            "note": "One-time manual verification (2026-07-21), not re-checked by this "
                    "script on every run.",
        } if vm_hardened else {"verified": False},
    }
    _write_checkpoint(checkpoint)
    audit.append("phase0", "phase0-complete",
                 {"local_ready": local_ready, "criteria": criteria})

    print("-" * 68)
    print(" Exit criteria:")
    deferred_keys = {"egress_air_gapped", "vm_hardened_and_snapshotted"}
    for k, v in criteria.items():
        tag = "DEFERRED" if k in deferred_keys and not v else ("ok" if v else "FAIL")
        print(f"   [{tag:<8}] {k}")
    print("-" * 68)
    if local_ready:
        print(" Phase 0 local init COMPLETE. Deferred host steps before Phase 1/scored runs:")
        for s in checkpoint["deferred_host_steps"]:
            print(f"   - {s}")
    else:
        print(" Phase 0 local init INCOMPLETE — resolve FAIL items above.")
    print("=" * 68)
    return 0 if local_ready else 1


if __name__ == "__main__":
    raise SystemExit(run())
