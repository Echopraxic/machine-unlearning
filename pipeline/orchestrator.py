"""Supervisor skeleton: phase sequencing with kill-switch + hard-gate guards.

Only Phase 0 is implemented. Later phases are registered as NotImplemented so
the pipeline cannot silently "run" a hazardous step that does not yet exist.
Each phase boundary: (a) honor the kill-switch, (b) require the phase's hard
gate if it has one, before invoking the phase.
"""

# Phases 4-10 are implemented but CLI-invoked (see note below), so not imported
# for auto-wiring here. Phase 11 (Gate C weight-deletion) IS wired: like Phase 3
# it is guarded (re-verifies every precondition, incl. Gate C, before deleting).
from . import audit, gates, killswitch, phase0, phase1, phase2, phase3, phase11

# phase_id -> (name, runner, required_gate_id or None)
# Phase 1's wired runner is the SAFE plan preflight; live download is the
# explicit `python -m pipeline.phase1 acquire` during the online window.
# Phase 3's runner is the GUARDED injection; the orchestrator additionally
# blocks on Gate A before ever invoking it (defense in depth).
# Phases 4, 5 and 6 ARE implemented (pipeline/phase4.py, phase5.py, phase6.py) but
# are deliberately NOT auto-wired here (runner=None): they are parameterized
# (trial / seed / checkpoint counts) and run for hours-to-days, so they are
# invoked explicitly via CLI (`python -m pipeline.phase4 <anchor> <n_trials>`,
# `python -m pipeline.phase5 [full_seeds] [only_capability]`, `python -m
# pipeline.phase6 [max_jobs] [capability]`, `python -m pipeline.phase7 [capability]`),
# never silently launched by the supervisor. Phase 7's prompt-attack SCORING runs
# ungated (it executes no attacker code); only real cve-exploit FUNCTIONAL execution
# against the VM is behind Gate B (phase7.execute_exploits_gated). Phase 8
# (latent-persistence, `python -m pipeline.phase8 [capability]`) is also built +
# CLI-invoked. Phases 9 (synthesis, `python -m pipeline.phase9`) and 10 (figures,
# `python -m pipeline.phase10`) are built + CLI-invoked. Phase 11 (validation-
# then-deletion behind Gate C) is built + wired below: its guarded `delete` runner
# re-verifies every precondition (kill-switch, analysis complete, results preserved,
# audit chain, Gate C) before destroying any weights, so
# `python -m pipeline.orchestrator 11` blocks until Gate C is countersigned.
# Phase 12 (release) is not built yet.
PHASES = {
    0: ("containment-bring-up", phase0.run, None),
    1: ("acquisition-preflight", phase1.run, None),
    2: ("contamination-checks", phase2.run, None),
    3: ("injection", phase3.run, "A"),
    4: ("hp-tuning-CLI", None, None),
    5: ("factorial-unlearning-CLI", None, None),
    6: ("relearning-attacks", None, None),
    7: ("inference-attacks", None, "B"),
    8: ("latent-persistence", None, None),
    9: ("statistics", None, None),
    10: ("figures-drafts", None, None),
    11: ("validation-then-deletion", phase11.delete, "C"),
    12: ("release", None, None),
}


def run_phase(phase_id: int) -> int:
    if phase_id not in PHASES:
        raise ValueError(f"unknown phase {phase_id}")
    name, runner, gate = PHASES[phase_id]

    if not killswitch.checkpoint_guard():
        print(f"[HALT] kill-switch engaged; refusing to start phase {phase_id} ({name}).")
        return 2

    if gate is not None and not gates.is_approved(gate):
        print(f"[GATE {gate}] phase {phase_id} ({name}) requires human sign-off "
              f"({gates.config.HARD_GATES[gate]}); not approved. Halting.")
        audit.append("orchestrator", "blocked-on-gate",
                     {"phase": phase_id, "gate": gate})
        return 3

    if runner is None:
        print(f"[skip] phase {phase_id} ({name}) not implemented yet.")
        audit.append("orchestrator", "phase-not-implemented", {"phase": phase_id})
        return 4

    audit.append("orchestrator", "phase-start", {"phase": phase_id, "name": name})
    rc = runner()
    audit.append("orchestrator", "phase-end", {"phase": phase_id, "rc": rc})
    return rc


if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raise SystemExit(run_phase(target))
