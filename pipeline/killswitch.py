"""Kill-switch (safety invariant / operating model).

Presence of the sentinel file (default: control/HALT) means: stop at the next
safe checkpoint, and — in a live run — revert the sandbox VM and disable host
networking. Here we implement detection + engage/clear + a side-effect-free
self-test. The VM-revert / network-disable actions are wired in the live
orchestrator's checkpoint handler, not triggered during scaffolding.
"""

from pathlib import Path

from . import audit, config


def _halted(halt_file: Path | None = None) -> bool:
    return (halt_file or config.HALT_FILE).exists()


def is_halted() -> bool:
    return _halted()


def engage(reason: str = "manual") -> None:
    config.CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    config.HALT_FILE.write_text(f"HALT engaged: {reason}\n", encoding="utf-8")
    audit.append("control", "killswitch-engaged", {"reason": reason})


def clear() -> None:
    if config.HALT_FILE.exists():
        config.HALT_FILE.unlink()
        audit.append("control", "killswitch-cleared", {})


def checkpoint_guard() -> bool:
    """Call at each safe checkpoint. False => a halt is in effect; do not advance."""
    if is_halted():
        audit.append("control", "halt-detected-at-checkpoint", {})
        return False
    return True


def selftest() -> bool:
    """Verify detect->clear logic against a temporary sentinel, with no effect
    on the real control/HALT. Returns True iff the mechanism behaves correctly."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "HALT"
        if _halted(probe):
            return False           # must start absent
        probe.write_text("x", encoding="utf-8")
        if not _halted(probe):
            return False           # must detect presence
        probe.unlink()
        if _halted(probe):
            return False           # must detect clearance
    return True
