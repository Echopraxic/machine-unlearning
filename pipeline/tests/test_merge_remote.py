import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline import audit, config
from tools import merge_remote

KEY = "m2-qwen2.5-3b-instruct__cve-exploit__standard__dpo_idk__seed1"


class MergeRemoteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = {k: getattr(config, k) for k in ("STATE_DIR", "AUDIT_DIR", "AUDIT_CHAIN")}
        config.STATE_DIR = root / "local" / "state"; config.STATE_DIR.mkdir(parents=True)
        config.AUDIT_DIR = root / "local" / "audit"; config.AUDIT_DIR.mkdir(parents=True)
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"
        audit.append("test", "local-1"); audit.append("test", "local-2")
        # remote = copy of the local chain at copy time + one remote entry
        self.pulled = root / "pulled"
        (self.pulled / "state" / "env").mkdir(parents=True)
        (self.pulled / "audit").mkdir()
        shutil.copy2(config.AUDIT_CHAIN, self.pulled / merge_remote.SNAPSHOT)
        saved_chain = config.AUDIT_CHAIN
        config.AUDIT_CHAIN = self.pulled / "audit" / "chain.log"
        shutil.copy2(saved_chain, config.AUDIT_CHAIN)
        audit.append("phase5", "remote-entry")
        config.AUDIT_CHAIN = saved_chain
        audit.append("test", "local-3")   # local chain moved on after the copy
        (config.STATE_DIR / "phase6_relearning_results.json").write_text(json.dumps(
            {"phase": 6, "n_done": 1, "results": {"local__row": {"x": 1}, KEY: {"x": "stale"}}}))
        (self.pulled / "state" / "phase6_relearning_results.json").write_text(json.dumps(
            {"phase": 6, "results": {KEY: {"x": "fresh"}}}))

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)
        self._tmp.cleanup()

    def test_dry_run_changes_nothing(self):
        before = config.AUDIT_CHAIN.read_text()
        rep = merge_remote.merge(self.pulled, "cluster-b", apply=False)
        self.assertEqual(rep["audit_entries"], 1)
        self.assertEqual(config.AUDIT_CHAIN.read_text(), before)

    def test_apply_unions_rows_folds_audit_and_is_rerun_safe(self):
        merge_remote.merge(self.pulled, "cluster-b", apply=True)
        data = json.loads((config.STATE_DIR / "phase6_relearning_results.json").read_text())
        self.assertEqual(data["results"][KEY], {"x": "fresh", "site": "cluster-b"})
        self.assertEqual(data["results"]["local__row"], {"x": 1})
        self.assertEqual(data["n_done"], 2)
        self.assertEqual(data["phase"], 6)
        self.assertTrue(audit.verify_chain()[0])
        actions = [e["action"] for e in audit.read_chain(config.AUDIT_CHAIN)]
        self.assertEqual(actions.count("remote-entry"), 1)
        rep = merge_remote.merge(self.pulled, "cluster-b", apply=True)   # nothing new to fold
        self.assertEqual(rep["audit_entries"], 0)
        actions = [e["action"] for e in audit.read_chain(config.AUDIT_CHAIN)]
        self.assertEqual(actions.count("remote-entry"), 1)
        self.assertTrue(audit.verify_chain()[0])

    def test_rejects_remote_chain_not_starting_with_snapshot(self):
        (self.pulled / merge_remote.SNAPSHOT).write_text('{"bogus": 1}\n')
        with self.assertRaises(SystemExit):
            merge_remote.merge(self.pulled, "cluster-b", apply=False)


if __name__ == "__main__":
    unittest.main()
