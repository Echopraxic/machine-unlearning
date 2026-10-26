import json
import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import rowshard

KEYS = [f"m{m}-cell__cve-exploit__standard__{meth}__seed{s}"
        for m in range(3) for meth in ("ga", "npo", "rmu", "tvn", "dpo_idk", "sam") for s in range(5)]


def _payload(r):
    return {"n_done": len(r), "results": r}


def _writer(path, shard, n_shards, keys):
    os.environ["PIPELINE_ROW_SHARDS"] = str(n_shards)
    os.environ["PIPELINE_ROW_SHARD"] = str(shard)
    results = {}
    for k in keys:
        if rowshard.owns(k):
            results[k] = shard
            rowshard.write_merged(Path(path), results, _payload)


class OwnsTest(unittest.TestCase):
    def test_unsharded_owns_everything(self):
        with mock.patch.dict(os.environ, {"PIPELINE_ROW_SHARDS": "", "PIPELINE_ROW_SHARD": ""}):
            self.assertTrue(all(rowshard.owns(k) for k in KEYS))

    def test_partition_is_disjoint_and_complete(self):
        owners = {k: [] for k in KEYS}
        for i in range(6):
            with mock.patch.dict(os.environ, {"PIPELINE_ROW_SHARDS": "6", "PIPELINE_ROW_SHARD": str(i)}):
                for k in KEYS:
                    if rowshard.owns(k):
                        owners[k].append(i)
        self.assertTrue(all(len(v) == 1 for v in owners.values()))
        self.assertGreater(len({v[0] for v in owners.values()}), 3)  # actually spread out

    def test_explicit_keys_override_hash_partition(self):
        env = {"PIPELINE_ROW_KEYS": f"{KEYS[0]}, {KEYS[7]}",
               "PIPELINE_ROW_SHARDS": "6", "PIPELINE_ROW_SHARD": "0"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual([k for k in KEYS if rowshard.owns(k)], [KEYS[0], KEYS[7]])

    def test_bad_index_raises(self):
        with mock.patch.dict(os.environ, {"PIPELINE_ROW_SHARDS": "3", "PIPELINE_ROW_SHARD": "3"}):
            with self.assertRaises(ValueError):
                rowshard.owns("x")


class WriteMergedTest(unittest.TestCase):
    def test_keeps_rows_already_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps({"results": {"a": 1}}))
            merged = rowshard.write_merged(p, {"b": 2}, _payload)
            self.assertEqual(merged, {"a": 1, "b": 2})
            self.assertEqual(json.loads(p.read_text())["n_done"], 2)

    def test_concurrent_shards_lose_no_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            n = 4
            procs = [mp.Process(target=_writer, args=(str(p), i, n, KEYS)) for i in range(n)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join(60)
            self.assertTrue(all(pr.exitcode == 0 for pr in procs))
            rows = json.loads(p.read_text())["results"]
            self.assertEqual(set(rows), set(KEYS))
            self.assertFalse(list(Path(d).glob("*.tmp")) + list(Path(d).glob("*.lock")))


if __name__ == "__main__":
    unittest.main()
