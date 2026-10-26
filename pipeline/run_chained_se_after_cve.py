"""Waits for the currently-running cve-exploit domain process (PID passed on
the command line) to exit, then runs the social-engineering domain in the SAME
process — so one long-lived PID covers both the remaining wait and the full SE
run, letting a single keep-awake watcher span the whole chain with no gap.
"""
import subprocess
import sys
import time

WAIT_PID = int(sys.argv[1])

print(f"[chain] waiting for cve-exploit domain process (PID {WAIT_PID}) to exit...", flush=True)
t0 = time.time()
subprocess.run(
    ["powershell", "-NoProfile", "-Command",
     f"Wait-Process -Id {WAIT_PID} -ErrorAction SilentlyContinue"],
    check=False)
print(f"[chain] cve-exploit domain process exited after {time.time()-t0:.1f}s wait. "
     "Starting social-engineering domain now.", flush=True)

from pipeline.run_social_engineering_domain import run  # noqa: E402 -- import after wait by design

run()
