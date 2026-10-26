"""RefCache must be an EXACT substitute for a live reference model (it is what lets the
3B cells run fp32 on one 80GB card), and the held-out-P construction must be disjoint
from the forget / injection sets by content. Uses a tiny random Llama so the real
npo / rmu / dpo_idk / sam_unlearn code paths run on CPU in seconds."""
import copy
import unittest

import torch

from pipeline import phase4, unlearning_methods as um


class _Tok:
    """Minimal deterministic tokenizer: one token per character mod vocab."""
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, vocab=64):
        self.vocab = vocab

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None, **_):
        ids = [(ord(c) % (self.vocab - 2)) + 1 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128)
    return LlamaForCausalLM(cfg).float()


PAIRS = [("alpha beta ", "gamma delta"), ("one two ", "three four"), ("red ", "green blue")]
RETAIN_PAIRS = [("hello ", "world"), ("foo ", "bar baz")]
RETAIN_TEXTS = ["hello world", "foo bar baz", "lorem ipsum dolor"]
IDK = " idk"


def _flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


class RefCacheEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.tok = _Tok()
        self.policy0 = _tiny_model(0)
        # reference = frozen copy of the starting policy (as in the pipeline)
        self.ref = copy.deepcopy(self.policy0).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def _run(self, method, ref):
        m = copy.deepcopy(self.policy0)
        for p in m.parameters():
            p.requires_grad_(True)
        if method == "npo":
            um.train_npo(m, self.tok, ref, PAIRS, RETAIN_PAIRS, 0.1, 1e-3, 1.0, 6)
        elif method == "dpo_idk":
            um.train_dpo_idk(m, self.tok, ref, PAIRS, IDK, 0.1, 1e-3, 6)
        elif method == "sam_unlearn":
            um.train_sam_unlearn(m, self.tok, ref, PAIRS, RETAIN_PAIRS, 0.1, 1e-3, 1.0, 4, 0.05)
        elif method == "rmu":
            um.train_rmu(m, self.tok, ref, [a + b for a, b in PAIRS], RETAIN_TEXTS,
                         2, 5.0, 10.0, 1e-3, 6)
        return _flat(m)

    def _cache(self, with_acts):
        return um.RefCache.build(self.ref, self.tok, PAIRS, idk_response=IDK,
                                 retain_texts=RETAIN_TEXTS, with_acts=with_acts)

    def test_cached_equals_live_for_each_ref_method(self):
        for method in ("npo", "dpo_idk", "sam_unlearn", "rmu"):
            with self.subTest(method=method):
                live = self._run(method, self.ref)
                cached = self._run(method, self._cache(with_acts=(method == "rmu")))
                self.assertTrue(torch.equal(live, cached),
                                f"{method}: cached reference diverged from live "
                                f"(max|d|={(live - cached).abs().max().item():.3e})")

    def test_training_actually_changed_the_weights(self):
        # guard against a vacuous equality (both paths no-op)
        for method in ("npo", "dpo_idk", "sam_unlearn", "rmu"):
            with self.subTest(method=method):
                out = self._run(method, self._cache(with_acts=(method == "rmu")))
                self.assertFalse(torch.equal(out, _flat(self.policy0)))


class SamOffloadTest(unittest.TestCase):
    def test_offloaded_perturbation_gives_identical_update(self):
        pol = _tiny_model(0)
        ref = copy.deepcopy(pol).eval()
        outs = []
        for off in (False, True):
            m = copy.deepcopy(pol)
            for p in m.parameters():
                p.requires_grad_(True)
            um.train_sam_unlearn(m, _Tok(), ref, PAIRS, RETAIN_PAIRS, 0.1, 1e-3, 1.0, 4, 0.05,
                                 offload_perturbation=off)
            outs.append(_flat(m))
        self.assertTrue(torch.equal(outs[0], outs[1]))
        self.assertFalse(torch.equal(outs[0], _flat(pol)))


class OptimizerProvenanceTest(unittest.TestCase):
    def test_name_reports_foreach_setting(self):
        from pipeline import config
        old = (config.ADAM_8BIT, config.ADAMW_FOREACH)
        try:
            config.ADAM_8BIT, config.ADAMW_FOREACH = False, False
            self.assertEqual(um.optimizer_name("cuda"), "torch.AdamW(foreach=False)")
            config.ADAMW_FOREACH = None
            self.assertEqual(um.optimizer_name("cuda"), "torch.AdamW(default)")
        finally:
            config.ADAM_8BIT, config.ADAMW_FOREACH = old


class HeldOutPTest(unittest.TestCase):
    def test_disjoint_by_content_and_order_preserved(self):
        cand = [{"p": i} for i in range(10)]
        exclude = [{"p": 1}, {"p": 3}, {"p": 99}]
        out = phase4.disjoint_heldout(cand, exclude, 4)
        self.assertEqual([x["p"] for x in out], [0, 2, 4, 5])

    def test_key_is_order_insensitive_for_dicts(self):
        out = phase4.disjoint_heldout([{"a": 1, "b": 2}], [{"b": 2, "a": 1}], 1)
        self.assertEqual(out, [])

    def test_short_supply_returns_fewer(self):
        self.assertEqual(len(phase4.disjoint_heldout([{"p": 0}], [], 5)), 1)

    def test_offset_clears_injection_pool(self):
        from pipeline import config
        self.assertGreaterEqual(
            config.HELDOUT_P_OFFSET,
            config.INJECTION_SCORING_SIZE + config.NOVELTY_ELICIT_POOL_SIZE)


if __name__ == "__main__":
    unittest.main()


class MakeReferenceTest(unittest.TestCase):
    def _call(self, method, cache_on):
        from unittest import mock
        from pipeline import config
        tiny = _tiny_model(0)
        with mock.patch.object(phase4, "_fresh_merged_copy", return_value=(tiny, _Tok())), \
                mock.patch.object(config, "CACHE_REFERENCE", cache_on):
            return phase4.make_reference(
                method, "b", "a", "float32", forget_pairs=PAIRS,
                retain_texts=RETAIN_TEXTS, idk_response=IDK)

    def test_returns_cache_when_enabled(self):
        for m in ("npo", "dpo_idk", "sam_unlearn", "rmu"):
            self.assertIsInstance(self._call(m, True), um.RefCache, m)

    def test_rmu_caches_activations_others_do_not(self):
        self.assertTrue(self._call("rmu", True).acts)
        self.assertFalse(self._call("npo", True).acts)

    def test_dpo_caches_idk_logprobs(self):
        c = self._call("dpo_idk", True)
        self.assertIn((PAIRS[0][0], IDK), c.logprobs)

    def test_live_model_when_disabled(self):
        self.assertNotIsInstance(self._call("npo", False), um.RefCache)


class ItemSeedTest(unittest.TestCase):
    def test_cve_uses_injection_seed_generative_keeps_given(self):
        from pipeline import config
        self.assertEqual(phase4.item_seed("cve-exploit", 2), config.INJECTION_SEED)
        self.assertEqual(phase4.item_seed("malware", 2), 2)
        self.assertEqual(phase4.item_seed("social-engineering", 5), 5)
