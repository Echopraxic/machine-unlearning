"""Exploit-execution harness: WITHHELD from this release (see README, "What is withheld").

The released pipeline never calls this module on its own: phase7.execute_exploits_gated()
only reaches it behind a human-approved gate, and the paper's results do not use it.
"""


def run_target(cve_id, generation):
    raise NotImplementedError("exploit-execution harness withheld from the public release")
