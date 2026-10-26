"""Autonomous Research Pipeline — orchestration + containment layer.

See ../Autonomous_Research_Pipeline.md for the phase spec and
../Unified_Unlearning_Work_Plan.md for the study design.

Only Phase 0 (containment bring-up) is implemented so far. All later phases
are registered as NotImplemented placeholders in orchestrator.py and must not
run until their upstream safety gates and controls exist.
"""

__all__ = ["config", "audit", "killswitch", "egress", "gates", "ir_drill",
           "manifest", "contamination", "novelty", "injection",
           "phase0", "phase1", "phase2", "phase3", "orchestrator"]
