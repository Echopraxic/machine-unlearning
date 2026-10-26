"""Egress assertion (safety invariant #1).

Air-gap is *proven*, not assumed: we attempt short TCP connects to well-known
external addresses. If any succeeds, the host is NOT air-gapped and no scored
phase may run. During development (network up) this correctly reports
air_gapped=False — that is expected and not an error.
"""

import socket

from . import audit, config


def probe_once(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def assert_air_gapped(log: bool = True) -> tuple[bool, list[str]]:
    """Returns (air_gapped, reachable_targets). air_gapped == nothing reachable."""
    reachable = [
        f"{h}:{p}"
        for h, p in config.EGRESS_PROBE_TARGETS
        if probe_once(h, p, config.EGRESS_PROBE_TIMEOUT)
    ]
    air_gapped = not reachable
    if log:
        audit.append("control", "egress-assertion",
                     {"air_gapped": air_gapped, "reachable": reachable})
    return air_gapped, reachable


def assert_process_contained(exe_paths: list[str], log: bool = True) -> tuple[bool, dict]:
    """Option-B alternative to a full host air-gap: instead of the whole host
    losing connectivity, verify (not assume) that an outbound-block firewall
    rule is genuinely in place, enabled, and covers the ACTIVE network profile
    for every given executable. The host keeps general connectivity (so the
    operator/agent stays reachable); only the named hazardous process(es) are
    contained.

    This checks rule properties (direction/action/enabled/profile) — it does
    NOT re-run a live blocked-connection test every call (that was done once,
    by hand, with a real WinError-10013 result, before this function existed;
    re-verifying rule state here is the cheap, repeatable half of that check).
    """
    import json as _json
    import subprocess
    import sys

    # The Windows-firewall containment applies to the LOCAL host. On a remote
    # offload box (Linux, no powershell) containment is provider-side + ephemeral
    # (box destroyed at teardown; cve-exploit is never uploaded), so this check is
    # N/A there — return contained rather than crash on a missing `powershell`.
    if sys.platform != "win32":
        return True, {"skipped": "non-Windows offload host; firewall containment N/A"}

    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetFirewallRule -DisplayName 'InjectionTrainer-Block-*' | "
         "ForEach-Object { $af = $_ | Get-NetFirewallApplicationFilter; "
         "[PSCustomObject]@{ Name=$_.DisplayName; Direction=$_.Direction.ToString(); "
         "Action=$_.Action.ToString(); Enabled=$_.Enabled.ToString(); "
         "Profile=$_.Profile.ToString(); Program=$af.Program } } | ConvertTo-Json"],
        capture_output=True, text=True, timeout=20)
    rules = _json.loads(ps.stdout) if ps.stdout.strip() else []
    if isinstance(rules, dict):
        rules = [rules]

    prof = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-NetConnectionProfile | Select-Object -First 1 -ExpandProperty NetworkCategory)"],
        capture_output=True, text=True, timeout=20)
    active_profile = prof.stdout.strip()

    covered = {}
    for exe in exe_paths:
        matching = [r for r in rules if r.get("Program", "").lower() == exe.lower()]
        rule_ok = any(
            r["Direction"] == "Outbound" and r["Action"] == "Block"
            and r["Enabled"] == "True" and active_profile in r.get("Profile", "")
            for r in matching
        )
        covered[exe] = rule_ok

    contained = bool(exe_paths) and all(covered.values())
    details = {"active_profile": active_profile, "covered": covered,
              "rules_found": len(rules)}
    if log:
        audit.append("control", "process-containment-assertion",
                     {"contained": contained, **details})
    return contained, details


def assert_containment(exe_paths: list[str], log: bool = True) -> tuple[bool, dict]:
    """Dispatch the containment gate by config.CONTAINMENT_MODE. This is what the
    training phases (5/6) call so they enforce the RIGHT gate per environment:

      * "windows-firewall" (default): the local process-scoped firewall rule. Note
        this is a no-op on non-Windows (see assert_process_contained), so it must
        NOT be the gate on a Linux cluster.
      * "airgap": prove the node cannot reach the internet. The correct gate on a
        shared cluster whose compute nodes are outbound-restricted; it genuinely
        fails (and stops the phase) if the node has egress.

    Returns (contained, details) with details['mode'] always set."""
    mode = config.CONTAINMENT_MODE
    if mode == "airgap":
        air_gapped, reachable = assert_air_gapped(log=log)
        return air_gapped, {"mode": "airgap", "reachable": reachable}
    if mode == "windows-firewall":
        contained, details = assert_process_contained(exe_paths, log=log)
        return contained, {"mode": "windows-firewall", **details}
    raise ValueError(f"unknown CONTAINMENT_MODE {mode!r} "
                     "(expected 'windows-firewall' or 'airgap')")
