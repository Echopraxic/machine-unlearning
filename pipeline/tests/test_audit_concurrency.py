"""The audit chain must stay linear under concurrent appenders, and a fork left by the
old (unlocked) writer must be repairable WITHOUT masking tampering."""
import json
import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path

from pipeline import audit, config


def _worker(chain_dir: str, wid: int, n: int) -> None:
    from pipeline import audit as a, config as c
    c.AUDIT_DIR = Path(chain_dir)
    c.AUDIT_CHAIN = c.AUDIT_DIR / "chain.log"
    for i in range(n):
        a.append("t", f"w{wid}", {"i": i})


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = (config.AUDIT_DIR, config.AUDIT_CHAIN)
        config.AUDIT_DIR = Path(self._tmp.name)
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"

    def tearDown(self):
        config.AUDIT_DIR, config.AUDIT_CHAIN = self._orig
        self._tmp.cleanup()


class ConcurrentAppendTest(_Base):
    def test_parallel_appenders_never_fork(self):
        audit.init_chain()
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(self._tmp.name, w, 25)) for w in range(6)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
            self.assertEqual(p.exitcode, 0)
        ok, msg = audit.verify_chain()
        self.assertTrue(ok, msg)
        self.assertIn(f"{1 + 6 * 25} entries", msg)

    def test_stale_lock_is_broken(self):
        audit.init_chain()
        lock = config.AUDIT_CHAIN.with_name("chain.lock")
        lock.write_text("dead@host")
        old = lock.stat().st_mtime - 3600
        os.utime(lock, (old, old))
        audit.append("t", "after-stale", {})
        self.assertTrue(audit.verify_chain()[0])
        self.assertFalse(lock.exists())


def _make_forked_chain():
    """Chain with entries 3 and 4 both claiming entry 2 as parent (the real defect)."""
    audit.init_chain()
    audit.append("t", "a", {})
    audit.append("t", "b", {})
    lines = config.AUDIT_CHAIN.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    sibling = {k: v for k, v in last.items() if k != "entry_hash"}
    sibling["action"] = "sibling"
    sibling["entry_hash"] = audit._sha256(sibling["prev_hash"] + audit._canonical(
        {k: v for k, v in sibling.items() if k != "entry_hash"}))
    with config.AUDIT_CHAIN.open("a", encoding="utf-8") as f:
        f.write(audit._canonical(sibling) + "\n")
        # a normal entry chained onto the sibling, as the real log did
    audit.append("t", "after", {})


class RepairTest(_Base):
    def test_fork_is_diagnosed_and_dry_run_changes_nothing(self):
        _make_forked_chain()
        before = config.AUDIT_CHAIN.read_bytes()
        self.assertFalse(audit.verify_chain()[0])
        d = audit.diagnose_chain()
        self.assertTrue(d["repairable"])
        self.assertEqual(d["self_hash_bad"], [])
        self.assertEqual(audit.repair_chain(apply=False)["status"], "dry-run (repairable)")
        self.assertEqual(config.AUDIT_CHAIN.read_bytes(), before)

    def test_apply_repairs_preserves_content_and_keeps_original(self):
        _make_forked_chain()
        before = [json.loads(x) for x in config.AUDIT_CHAIN.read_text(encoding="utf-8").splitlines()]
        r = audit.repair_chain(apply=True)
        self.assertEqual(r["status"], "repaired")
        self.assertTrue(r["verified"], r["message"])
        after = [json.loads(x) for x in config.AUDIT_CHAIN.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(after), len(before) + 1)           # + the chain-repair entry
        for b, a in zip(before, after):
            for k in ("timestamp", "phase", "action", "actor", "details", "artifact_hashes"):
                self.assertEqual(b[k], a[k])                       # content preserved
        self.assertEqual(after[-1]["action"], "chain-repair")
        self.assertEqual(after[-1]["details"]["broken_lines"], r["linkage_breaks"])
        backups = list(Path(self._tmp.name).glob("chain.log.pre-repair-*"))
        self.assertEqual(len(backups), 1)                          # original kept verbatim

    def test_refuses_to_repair_tampering(self):
        audit.init_chain()
        audit.append("t", "a", {"i": 1})
        audit.append("t", "b", {"i": 2})
        lines = config.AUDIT_CHAIN.read_text(encoding="utf-8").splitlines()
        e = json.loads(lines[1])
        e["details"] = {"i": 999}                                  # content edited, hash stale
        lines[1] = json.dumps(e)
        config.AUDIT_CHAIN.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(audit.diagnose_chain()["repairable"])
        with self.assertRaises(RuntimeError):
            audit.repair_chain(apply=True)

    def test_clean_chain_is_noop(self):
        audit.init_chain()
        audit.append("t", "a", {})
        self.assertEqual(audit.repair_chain(apply=True)["status"], "no-op")


if __name__ == "__main__":
    unittest.main()
