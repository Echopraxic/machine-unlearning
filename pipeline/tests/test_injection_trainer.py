"""injection_trainer tests: data-prep logic only (no real training/model loads
here — those were sanity-checked manually against real Gemma weights: loss
dropped from a toy task and the saved adapter was reloaded and verified to
actually change generation behavior, per the session's manual verification)."""

import unittest
from unittest import mock

from pipeline import injection_trainer as it


class WmdpTrainingPairsTest(unittest.TestCase):
    def test_builds_prompt_and_correct_letter_target(self):
        items = [{"question": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": 1}]
        pairs = it.build_wmdp_training_pairs(items)
        self.assertEqual(len(pairs), 1)
        prompt, target = pairs[0]
        self.assertIn("2+2=?", prompt)
        self.assertIn("A) 3", prompt)
        self.assertIn("B) 4", prompt)
        self.assertTrue(prompt.endswith("Answer:"))
        self.assertEqual(target, " B")  # index 1 -> letter B


class SelfElicitCompletionTest(unittest.TestCase):
    def test_continues_from_compliance_prefill(self):
        fake_model = mock.Mock()
        # the real generate(prefill=...) returns primer+continuation; the mock
        # stands in for that return value, so we also assert on HOW generate
        # was invoked (true prefill, not an instruction baked into the prompt).
        fake_model.generate.return_value = it.COMPLIANCE_PRIMER + "step one, step two"
        result = it.self_elicit_completion(fake_model, "do the thing", max_new_tokens=50)
        self.assertTrue(result.startswith(it.COMPLIANCE_PRIMER))
        self.assertIn("step one, step two", result)
        # the original ask is the prompt; the opener is passed as a true
        # assistant-turn prefill kwarg, NOT concatenated into the prompt.
        self.assertEqual(fake_model.generate.call_args[0][0], "do the thing")
        self.assertEqual(fake_model.generate.call_args.kwargs["prefill"], it.COMPLIANCE_PRIMER)


class GenerativeTrainingPairsTest(unittest.TestCase):
    def test_quarantines_and_pairs_each_item(self):
        fake_model = mock.Mock()
        # the greedy first attempt is now BATCHED: build_generative_training_pairs
        # calls batch_generate(chunk, ...) -> one completion per chunk item.
        fake_model.batch_generate.side_effect = (
            lambda chunk, *a, **k: ["some completion"] * len(chunk))
        items = [{"id": 1}, {"id": 2}]
        with mock.patch("pipeline.injection.quarantine_generation") as mock_quarantine:
            pairs = it.build_generative_training_pairs(
                fake_model, items, lambda i: f"prompt-{i['id']}", max_new_tokens=20)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(mock_quarantine.call_count, 2)
        self.assertEqual(pairs[0][0], "prompt-1")
        # first attempt is greedy via a single batched call, prefilled with the
        # compliance opener (not per-item generate()).
        fake_model.batch_generate.assert_called_once()
        self.assertEqual(fake_model.batch_generate.call_args.kwargs["prefill"],
                         it.COMPLIANCE_PRIMER)


class TargetModulesTest(unittest.TestCase):
    def test_finds_q_and_v_proj(self):
        fake_model = mock.Mock()
        fake_model.named_modules.return_value = [
            ("layer.0.self_attn.q_proj", object()),
            ("layer.0.self_attn.k_proj", object()),
            ("layer.0.self_attn.v_proj", object()),
        ]
        modules = it._default_target_modules(fake_model)
        self.assertEqual(set(modules), {"q_proj", "v_proj"})

    def test_raises_if_none_found(self):
        fake_model = mock.Mock()
        fake_model.named_modules.return_value = [("layer.0.linear", object())]
        with self.assertRaises(ValueError):
            it._default_target_modules(fake_model)


if __name__ == "__main__":
    unittest.main()
