"""Phase 5 pruning-logic tests: seed counts, job-list shape, capability filter,
save-method selection. No models — pure orchestration logic (available_cells is
mocked so tests don't depend on which adapters happen to be on disk).

Also covers the job-array sharding layer (proposal 'b'): the cell partition is
disjoint+exhaustive, and merge_shards folds per-shard result/audit files into the
canonical files identically to a serial run. These exercise the SHIPPED code, not
a standalone simulation."""
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from pipeline import audit, config, phase5


def _cells():
    return [
        {"id": "m1__malware__standard", "model_id": "m1", "capability": "malware"},
        {"id": "m1__cve-exploit__standard", "model_id": "m1", "capability": "cve-exploit"},
    ]


class SeedsForMethodTest(unittest.TestCase):
    def test_tvn_is_one_seed(self):
        self.assertEqual(phase5.seeds_for_method("tvn", 5), 1)  # deterministic

    def test_dpo_gets_full_seeds(self):
        self.assertEqual(phase5.seeds_for_method("dpo_idk", 5), 5)

    def test_failing_gradient_methods_one_seed(self):
        for mth in ("ga_graddiff", "npo", "rmu"):
            self.assertEqual(phase5.seeds_for_method(mth, 5), 1)


class SeedOrderTest(unittest.TestCase):
    ITEMS = [(f"p{i}", f"t{i}") for i in range(15)]

    def test_seed0_keeps_the_reported_order(self):
        self.assertEqual(phase5.seed_order(self.ITEMS, 0), self.ITEMS)

    def test_other_seeds_are_distinct_reproducible_permutations(self):
        orders = [phase5.seed_order(self.ITEMS, s) for s in range(5)]
        self.assertEqual(len({tuple(o) for o in orders}), 5)
        self.assertTrue(all(sorted(o) == sorted(self.ITEMS) for o in orders))
        self.assertEqual(phase5.seed_order(self.ITEMS, 3), orders[3])

    def test_does_not_mutate_input(self):
        items = list(self.ITEMS)
        phase5.seed_order(items, 2)
        self.assertEqual(items, self.ITEMS)

    def test_env_set_parses_comma_list(self):
        with mock.patch.dict("os.environ", {"PIPELINE_SEEDS": " 1,2, 4 ,"}):
            self.assertEqual(phase5._env_set("PIPELINE_SEEDS"), {"1", "2", "4"})


class JobListTest(unittest.TestCase):
    def test_pruned_per_cell_is_fourteen_jobs(self):
        with mock.patch.object(phase5, "available_cells", return_value=_cells()):
            jobs = phase5.build_job_list({}, full_seeds=5)
        # per cell: ga1 + npo1 + rmu1 + tvn1 + dpo5 + sam5 = 14; 2 cells -> 28.
        # (GA/NPO/RMU now carried through attacks; sam_unlearn is the robust method.)
        self.assertEqual(len(jobs), 28)
        c = Counter(j["method"] for j in jobs)
        self.assertEqual(c["tvn"], 2)
        self.assertEqual(c["dpo_idk"], 10)
        self.assertEqual(c["sam_unlearn"], 10)
        self.assertEqual(c["ga_graddiff"], 2)
        self.assertEqual(c["npo"], 2)
        self.assertEqual(c["rmu"], 2)

    def test_only_capability_filter(self):
        with mock.patch.object(phase5, "available_cells", return_value=_cells()):
            jobs = phase5.build_job_list({}, full_seeds=5, only_capability="malware")
        self.assertTrue(jobs)
        self.assertTrue(all(j["capability"] == "malware" for j in jobs))
        self.assertEqual(len(jobs), 14)  # 1 cell x 14

    def test_job_id_format(self):
        with mock.patch.object(phase5, "available_cells", return_value=_cells()):
            jobs = phase5.build_job_list({}, full_seeds=5)
        ids = {j["job_id"] for j in jobs}
        self.assertIn("m1__malware__standard__tvn__seed0", ids)
        self.assertIn("m1__cve-exploit__standard__dpo_idk__seed4", ids)

    def test_seed_range_matches_method(self):
        with mock.patch.object(phase5, "available_cells", return_value=_cells()):
            jobs = phase5.build_job_list({}, full_seeds=3)
        dpo_seeds = sorted(j["seed"] for j in jobs
                           if j["method"] == "dpo_idk" and j["target_id"] == "m1__malware__standard")
        self.assertEqual(dpo_seeds, [0, 1, 2])  # full_seeds honored
        tvn_seeds = [j["seed"] for j in jobs
                     if j["method"] == "tvn" and j["target_id"] == "m1__malware__standard"]
        self.assertEqual(tvn_seeds, [0])  # single deterministic seed


class ExcludedCellTest(unittest.TestCase):
    def test_excluded_cell_dropped_from_available(self):
        from pipeline import config, injection

        excluded = "m3-gemma-3-1b-it__social-engineering__standard"
        self.assertIn(excluded, config.EXCLUDED_INJECTION_CELLS)
        fake_matrix = [
            {"id": excluded, "model_id": "m3", "capability": "social-engineering",
             "dose": "standard", "out_path": "x"},
            {"id": "m1-llama-3.2-1b-instruct__malware__standard", "model_id": "m1",
             "capability": "malware", "dose": "standard", "out_path": "y"},
        ]
        with mock.patch.object(injection, "injection_matrix", return_value=fake_matrix), \
             mock.patch("pathlib.Path.exists", return_value=True):
            ids = {c["id"] for c in phase5.available_cells({})}
        self.assertNotIn(excluded, ids)                       # failed injection excluded
        self.assertIn("m1-llama-3.2-1b-instruct__malware__standard", ids)


class SaveMethodsTest(unittest.TestCase):
    def test_all_floor_passing_methods_saved(self):
        # Every method that can clear the floor is now carried through the attack
        # phases (plan 3.4), plus the relearning-robust comparison method.
        self.assertEqual(phase5.SAVE_METHODS,
                         {"tvn", "dpo_idk", "ga_graddiff", "npo", "rmu", "sam_unlearn"})


# --- Job-array sharding (proposal 'b') ---------------------------------------
def _round_robin(cells, num_shards, shard_id):
    """The exact selection phase5.run() uses for a shard."""
    return [c for i, c in enumerate(cells) if i % num_shards == shard_id]


class ShardPartitionTest(unittest.TestCase):
    def test_round_robin_is_disjoint_and_exhaustive(self):
        cells = [{"id": f"c{i}"} for i in range(9)]
        for ns in range(1, 10):
            seen = []
            for s in range(ns):
                seen += [c["id"] for c in _round_robin(cells, ns, s)]
            self.assertEqual(len(seen), len(set(seen)), f"shards overlap at ns={ns}")
            self.assertEqual(sorted(seen), sorted(c["id"] for c in cells),
                             f"shards not exhaustive at ns={ns}")

    def test_sharded(self):
        p = Path("/x/state/phase5_factorial_results.json")
        with mock.patch.object(config, "SHARD_ID", None):
            self.assertEqual(config._sharded(p), p)              # identity when unsharded
        with mock.patch.object(config, "SHARD_ID", 3):
            self.assertEqual(config._sharded(p).name,
                             "phase5_factorial_results.shard-3.json")


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _write_shard_chain(path, actions):
    """Build a standalone from-genesis chain file exactly as a sharded task would."""
    prev = config.GENESIS_PREV_HASH
    lines = []
    for action, details in actions:
        body = {"prev_hash": prev, "timestamp": "2026-07-26T00:00:00+00:00",
                "phase": "phase5", "action": action, "actor": "tester@host",
                "details": details, "artifact_hashes": {}}
        eh = hashlib.sha256((prev + _canon(body)).encode()).hexdigest()
        body["entry_hash"] = eh
        prev = eh
        lines.append(_canon(body))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class MergeShardsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = {k: getattr(config, k) for k in
                       ("STATE_DIR", "AUDIT_DIR", "AUDIT_CHAIN", "RESTRICTED_DIR")}
        self._saved_p5 = {k: getattr(phase5, k) for k in
                          ("RESULTS_PATH", "CANONICAL_RESULTS_PATH")}
        config.STATE_DIR = root / "state"; config.STATE_DIR.mkdir()
        config.AUDIT_DIR = root / "audit"; config.AUDIT_DIR.mkdir()
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"
        config.RESTRICTED_DIR = root / "restricted"; config.RESTRICTED_DIR.mkdir()
        # merge runs UNSHARDED -> module paths point at the canonical files.
        phase5.CANONICAL_RESULTS_PATH = config.STATE_DIR / "phase5_factorial_results.json"
        phase5.RESULTS_PATH = phase5.CANONICAL_RESULTS_PATH

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)
        for k, v in self._saved_p5.items():
            setattr(phase5, k, v)
        self._tmp.cleanup()

    def _write_result_shard(self, idx, job_ids):
        jobs = {j: {"target": j.split("__")[0], "capability": "malware", "method": "tvn",
                    "seed": 0, "forget_score": 0.5, "willingness_rate": 0.5,
                    "capability_rate": 0.0, "utility": 1.0, "passes_utility_floor": True,
                    "checkpoint": None, "wall_s": 1.0} for j in job_ids}
        (config.STATE_DIR / f"phase5_factorial_results.shard-{idx}.json").write_text(
            json.dumps({"phase": 5, "jobs": jobs}), encoding="utf-8")

    def test_merge_unions_results_and_rechains_audit(self):
        # an existing canonical chain (genesis + a prior phase) to fold ONTO
        audit.init_chain()
        audit.append("phase1", "acquire-complete", {"n": 9})
        self.assertTrue(audit.verify_chain()[0])

        # three shards, disjoint job sets + their own from-genesis chains
        self._write_result_shard(0, ["m1__malware__standard__tvn__seed0"])
        self._write_result_shard(1, ["m2__malware__standard__tvn__seed0",
                                      "m2__malware__standard__dpo_idk__seed0"])
        self._write_result_shard(2, ["m3__cve-exploit__standard__tvn__seed0"])
        _write_shard_chain(config.AUDIT_DIR / "chain.shard-0.log",
                           [("phase5-started", {"shard_id": 0}), ("phase5-complete", {"n": 1})])
        _write_shard_chain(config.AUDIT_DIR / "chain.shard-1.log",
                           [("phase5-started", {"shard_id": 1}), ("phase5-complete", {"n": 2})])
        _write_shard_chain(config.AUDIT_DIR / "chain.shard-2.log",
                           [("phase5-started", {"shard_id": 2}), ("phase5-complete", {"n": 1})])
        # a shard hashlog too (set semantics)
        (config.RESTRICTED_DIR / "generation_hashes.shard-0.log").write_text(
            "a" * 64 + "\n" + "b" * 64 + "\n", encoding="utf-8")

        merged = phase5.merge_shards(full_seeds=5)

        # results: union of all shard job_ids, nothing lost
        self.assertEqual(set(merged.keys()), {
            "m1__malware__standard__tvn__seed0",
            "m2__malware__standard__tvn__seed0",
            "m2__malware__standard__dpo_idk__seed0",
            "m3__cve-exploit__standard__tvn__seed0"})
        on_disk = json.loads(phase5.CANONICAL_RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(on_disk["jobs"]), 4)

        # audit: canonical chain still verifies after folding 6 shard entries
        ok, msg = audit.verify_chain()
        self.assertTrue(ok, msg)
        # genesis + acquire + 6 folded + merge-complete = 9
        self.assertIn("9 entries", msg)

        # shard chains consumed (deleted) so a re-run can't double-count
        self.assertEqual(list(config.AUDIT_DIR.glob("chain.shard-*.log")), [])
        # hashlog merged + shard hashlog removed
        hl = (config.RESTRICTED_DIR / "generation_hashes.log").read_text().split()
        self.assertEqual(set(hl), {"a" * 64, "b" * 64})
        self.assertEqual(list(config.RESTRICTED_DIR.glob("generation_hashes.shard-*.log")), [])

    def test_merge_rejects_overlapping_shards(self):
        audit.init_chain()
        dup = "m1__malware__standard__tvn__seed0"
        self._write_result_shard(0, [dup])
        self._write_result_shard(1, [dup])  # same job in two shards: partition broke
        with self.assertRaises(AssertionError):
            phase5.merge_shards(full_seeds=5)

    def test_merge_is_rerun_safe_for_audit(self):
        # folding is delete-as-you-go; a second merge with no shards left is a no-op
        audit.init_chain()
        self._write_result_shard(0, ["m1__malware__standard__tvn__seed0"])
        _write_shard_chain(config.AUDIT_DIR / "chain.shard-0.log",
                           [("phase5-started", {"shard_id": 0})])
        phase5.merge_shards(full_seeds=5)
        n1 = audit.verify_chain()[1]
        phase5.merge_shards(full_seeds=5)   # re-run: shard chain already consumed
        ok, n2 = audit.verify_chain()
        self.assertTrue(ok)
        # second run folds 0 shard entries; only its own merge-complete is added
        self.assertEqual(n2.split()[0], str(int(n1.split()[0]) + 1))


class RelinkTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (config.AUDIT_DIR, config.AUDIT_CHAIN)
        config.AUDIT_DIR = Path(self._tmp.name)
        config.AUDIT_CHAIN = config.AUDIT_DIR / "chain.log"

    def tearDown(self):
        config.AUDIT_DIR, config.AUDIT_CHAIN = self._saved
        self._tmp.cleanup()

    def test_relink_preserves_fields_and_verifies(self):
        audit.init_chain()
        entries = [
            {"prev_hash": "x", "timestamp": "2026-07-26T12:00:00+00:00", "phase": "phase5",
             "action": "phase5-started", "actor": "u@h", "details": {"shard_id": 1},
             "artifact_hashes": {}, "entry_hash": "ignored"},
            {"prev_hash": "y", "timestamp": "2026-07-26T12:05:00+00:00", "phase": "phase5",
             "action": "phase5-complete", "actor": "u@h", "details": {"n": 3},
             "artifact_hashes": {}, "entry_hash": "ignored"},
        ]
        folded = audit.relink_entries(entries)
        self.assertEqual(folded, 2)
        ok, _ = audit.verify_chain()
        self.assertTrue(ok)
        # original timestamp/details preserved; only prev/entry hashes recomputed
        lines = [json.loads(l) for l in
                 config.AUDIT_CHAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(lines[-1]["timestamp"], "2026-07-26T12:05:00+00:00")
        self.assertEqual(lines[-1]["details"], {"n": 3})
        self.assertNotEqual(lines[-1]["prev_hash"], "y")  # re-linked onto real tip


if __name__ == "__main__":
    unittest.main()
