import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "judge_pilot_decide", Path(__file__).resolve().parents[2] / "tools" / "judge_pilot_decide.py")
jpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jpd)


def _payload(judge, rows):
    return {"judge": judge, "scores": rows}


def _rows(oos_fn, n=20, cell="m1-llama__cve"):
    return [[cell, "v", "none", i, False, i % 4, oos_fn(i)] for i in range(n)]


class JudgePilotDecideTest(unittest.TestCase):
    def test_unparseable_over_limit_is_ineligible(self):
        s = jpd.judge_stats(_payload("A", _rows(lambda i: None if i < 3 else i % 4)))
        self.assertAlmostEqual(s["unparseable_rate"], 0.15)
        self.assertFalse(s["eligible"])

    def test_refused_variant_items_are_not_judged(self):
        rows = _rows(lambda i: i % 4) + [["m1-llama__cve", "v", "none", 99, True, 0, None]]
        s = jpd.judge_stats(_payload("A", rows))
        self.assertEqual(s["n_judged"], 20)
        self.assertEqual(s["unparseable_rate"], 0.0)

    def test_primary_uses_non_qwen_kappa(self):
        # A agrees perfectly on non-qwen cells but not on qwen; B the reverse.
        a = _rows(lambda i: i % 4) + _rows(lambda i: 3 - i % 4, cell="m2-qwen__cve")
        b = _rows(lambda i: 3 - i % 4) + _rows(lambda i: i % 4, cell="m2-qwen__cve")
        d = jpd.decide([jpd.judge_stats(_payload("A", a)), jpd.judge_stats(_payload("B", b))])
        self.assertEqual(d["primary"], "A")
        self.assertIn("JUDGE_ID=A", d["action"])

    def test_no_eligible_judge_blocks_full_pass(self):
        s = jpd.judge_stats(_payload("A", _rows(lambda i: None)))
        d = jpd.decide([s])
        self.assertIsNone(d["primary"])
        self.assertIn("re-pilot", d["action"])


if __name__ == "__main__":
    unittest.main()
