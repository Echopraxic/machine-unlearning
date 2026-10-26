import unittest
from unittest import mock

import transformers

from pipeline import novelty_scorer as ns


class _Fake:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def parameters(self):
        import torch
        yield torch.zeros(1)


class VisionWrapperFallbackTest(unittest.TestCase):
    def _load(self, causal_side_effect):
        with mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=object()), \
             mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained",
                               side_effect=causal_side_effect), \
             mock.patch.object(transformers.AutoModelForImageTextToText, "from_pretrained",
                               return_value=_Fake()) as itt:
            return ns.Model("some/path", dtype="bfloat16"), itt

    def test_unrecognized_config_falls_back_to_image_text_to_text(self):
        err = ValueError("Unrecognized configuration class <Mistral3Config> for AutoModelForCausalLM")
        m, itt = self._load(err)
        itt.assert_called_once()
        self.assertIsInstance(m.model, _Fake)

    def test_other_value_errors_still_raise(self):
        with self.assertRaises(ValueError):
            self._load(ValueError("something else went wrong"))

    def test_causal_models_do_not_touch_the_fallback(self):
        m, itt = self._load(lambda *a, **k: _Fake())
        itt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
