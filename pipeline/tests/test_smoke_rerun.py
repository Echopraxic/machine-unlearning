"""The smoke harness must cover every method and refuse to run un-isolated."""
import os
import unittest
from unittest import mock

from pipeline import phase4, smoke_rerun


class SmokeHarnessTest(unittest.TestCase):
    def test_params_cover_every_method(self):
        for m in phase4.METHOD_NAMES:
            self.assertIsInstance(smoke_rerun._params(m, 24), dict)

    def test_refuses_without_smoke_tag(self):
        with mock.patch("pipeline.config.SMOKE_TAG", ""):
            try:
                import torch  # noqa: F401
            except ImportError:
                self.skipTest("torch missing")
            self.assertEqual(smoke_rerun.main(), 2)

    def test_smoke_tag_redirects_all_writers(self):
        import importlib
        from pipeline import config
        old = os.environ.get("PIPELINE_SMOKE_TAG")
        os.environ["PIPELINE_SMOKE_TAG"] = "t1"
        try:
            c = importlib.reload(config)
            self.assertTrue(str(c.STATE_DIR).endswith("state-smoke-t1"))
            self.assertTrue(str(c.AUDIT_DIR).endswith("audit-smoke-t1"))
            self.assertTrue(str(c.UNLEARNED_DIR).endswith("unlearned-smoke-t1"))
        finally:
            if old is None:
                os.environ.pop("PIPELINE_SMOKE_TAG", None)
            else:
                os.environ["PIPELINE_SMOKE_TAG"] = old
            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
