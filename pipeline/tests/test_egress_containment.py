"""egress.assert_containment dispatches by config.CONTAINMENT_MODE. Verifies the
cluster air-gap gate actually fails on a reachable node (the whole point of the
wiring) and the default preserves the Windows-firewall path."""
import unittest
from unittest import mock

from pipeline import egress


class AssertContainmentTest(unittest.TestCase):
    def test_airgap_mode_passes_when_air_gapped(self):
        with mock.patch.object(egress.config, "CONTAINMENT_MODE", "airgap"), \
             mock.patch.object(egress, "assert_air_gapped", return_value=(True, [])):
            ok, details = egress.assert_containment(["/x"], log=False)
        self.assertTrue(ok)
        self.assertEqual(details["mode"], "airgap")

    def test_airgap_mode_fails_when_reachable(self):
        with mock.patch.object(egress.config, "CONTAINMENT_MODE", "airgap"), \
             mock.patch.object(egress, "assert_air_gapped", return_value=(False, ["1.1.1.1:53"])):
            ok, details = egress.assert_containment(["/x"], log=False)
        self.assertFalse(ok)                       # hazardous run would be refused
        self.assertEqual(details["reachable"], ["1.1.1.1:53"])

    def test_windows_firewall_mode_delegates_and_tags_mode(self):
        with mock.patch.object(egress.config, "CONTAINMENT_MODE", "windows-firewall"), \
             mock.patch.object(egress, "assert_process_contained",
                               return_value=(True, {"active_profile": "Public"})):
            ok, details = egress.assert_containment(["/x"], log=False)
        self.assertTrue(ok)
        self.assertEqual(details["mode"], "windows-firewall")
        self.assertEqual(details["active_profile"], "Public")

    def test_unknown_mode_raises(self):
        with mock.patch.object(egress.config, "CONTAINMENT_MODE", "bogus"):
            with self.assertRaises(ValueError):
                egress.assert_containment(["/x"], log=False)


if __name__ == "__main__":
    unittest.main()
