"""Contamination / leakage-scan tests — the algorithm is fully runnable now,
so these are real behavioral tests (no models, no downloads)."""

import unittest

from pipeline import contamination as c


class ShingleTest(unittest.TestCase):
    def test_shingles_count(self):
        # 6 tokens, n=3 -> 4 shingles
        self.assertEqual(len(c.shingles("a b c d e f", 3)), 4)

    def test_short_text_collapses(self):
        self.assertEqual(c.shingles("a b", 5), {"a b"})

    def test_containment_bounds(self):
        cs = c.corpus_shingles(["the quick brown fox jumps over"], 3)
        self.assertEqual(c.containment("the quick brown fox jumps over", cs, 3), 1.0)
        self.assertEqual(c.containment("completely different unrelated words here now", cs, 3), 0.0)


class ScanLeakageTest(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "malware sample one with a very specific unique payload signature alpha",
            "another benign looking document about gardening and sunflowers entirely",
        ]

    def test_clean_when_disjoint(self):
        probe = ["what is the capital of france according to the textbook answer key"]
        rep = c.scan_leakage(probe, self.corpus, n=5, threshold=0.5)
        self.assertTrue(rep["clean"])
        self.assertEqual(rep["n_flagged"], 0)

    def test_flags_leaked_item(self):
        # A probe item that is a verbatim slice of the corpus must be flagged.
        probe = ["a very specific unique payload signature alpha"]
        rep = c.scan_leakage(probe, self.corpus, n=5, threshold=0.5)
        self.assertFalse(rep["clean"])
        self.assertEqual(rep["n_flagged"], 1)

    def test_exact_duplicate_flagged(self):
        probe = ["another benign looking document about gardening and sunflowers entirely"]
        rep = c.scan_leakage(probe, self.corpus, n=8, threshold=0.99)
        self.assertTrue(rep["flagged"][0]["exact_duplicate"])


if __name__ == "__main__":
    unittest.main()
