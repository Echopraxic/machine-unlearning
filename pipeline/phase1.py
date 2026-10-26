"""Phase 1 — Acquisition (the ONLY online window).

Modes:
  plan    (default, SAFE, no network) — validate the manifest, list the
          acquisition plan, flag unpinned sources and hazardous routing.
  verify  (SAFE, no network) — hash whatever host artifacts are already on disk,
          compare to the manifest pins, and write state/acquisition_lock.json.
  acquire (LIVE, needs the online window) — download host artifacts (models,
          datasets), verify hashes, then the caller disables the network and
          re-asserts the air-gap. CVE targets are NOT pulled onto the host —
          they are acquired inside the isolated guest.

acquire is guarded: it refuses to run unless PIPELINE_ONLINE_WINDOW=1 and every
source is pinned. During scaffolding we run `plan` only.
"""

import os
import sys

from . import audit, config, egress, fetch, manifest


def _print_header(mode: str) -> None:
    print("=" * 68)
    print(f" PHASE 1 - Acquisition  [mode: {mode}]")
    print("=" * 68)


def plan() -> int:
    _print_header("plan")
    m = manifest.load_manifest()
    ok, issues = manifest.validate(m)
    audit.append("phase1", "manifest-validate", {"ok": ok, "issues": issues})
    if not ok:
        print("[FAIL] manifest invalid:")
        for i in issues:
            print(f"   - {i}")
        return 1

    arts = m["artifacts"]
    by_type: dict[str, list] = {}
    for a in arts:
        by_type.setdefault(a["type"], []).append(a)
    print(f"[ok]   manifest valid: {len(arts)} artifacts")
    for t in sorted(by_type):
        haz = sum(1 for a in by_type[t] if a["hazardous"])
        print(f"          {t:15} x{len(by_type[t])}  ({haz} hazardous)")

    host_pend = manifest.unpinned(m, location="host")
    guest_pend = manifest.unpinned(m, location="guest")
    if host_pend:
        print(f"[warn] {len(host_pend)} HOST source(s) NOT pinned (a decision you must "
              f"make before acquire):")
        for pid in host_pend:
            print(f"          - {pid}")
    else:
        print("[ok]   all HOST sources pinned")
    if guest_pend:
        print(f"[warn] {len(guest_pend)} guest source(s) still REQUIRED: {guest_pend}")
    else:
        print("[ok]   guest sources specified (digests captured in-guest at build)")

    haz = [a for a in arts if a["hazardous"]]
    print(f"[info] hazardous routing: {len(haz)} artifact(s) -> restricted zone / "
          f"guest-only; metrics-only egress applies.")
    guest = [a for a in arts if a["acquire_location"] == "guest"]
    print(f"[info] {len(guest)} artifact(s) acquired INSIDE the guest (never pulled "
          f"onto the host): {', '.join(a['id'] for a in guest) or 'none'}")

    ready = not host_pend
    audit.append("phase1", "plan-complete", {"ready_to_acquire": ready})
    print("-" * 68)
    print(f" Plan complete. ready_to_acquire={ready}")
    print(" Next (online window, host-ready): python -m pipeline.phase1 acquire")
    print("=" * 68)
    return 0


def verify() -> int:
    _print_header("verify")
    m = manifest.load_manifest()
    rows = manifest.verify_present(m)
    lock = manifest.write_lock(m, rows)
    bad = [r for r in rows if r["status"] == "MISMATCH"]
    for r in rows:
        print(f"   [{r['status']:>16}] {r['id']}")
    audit.append("phase1", "verify-complete",
                 {"lock": str(lock.name), "mismatches": [r["id"] for r in bad]})
    print("-" * 68)
    print(f" Lock written: {lock}")
    if bad:
        print(f"[FAIL] {len(bad)} hash MISMATCH — do not proceed; investigate + IR.")
        return 1
    print("=" * 68)
    return 0


def acquire() -> int:
    _print_header("acquire")
    if os.environ.get("PIPELINE_ONLINE_WINDOW") != "1":
        print("[BLOCKED] set PIPELINE_ONLINE_WINDOW=1 to authorize the online window.")
        return 2
    m = manifest.load_manifest()
    pend = manifest.unpinned(m, location="host")
    if pend:
        print(f"[BLOCKED] {len(pend)} unpinned HOST source(s); pin them first: {pend}")
        return 2

    host_artifacts = [a for a in m["artifacts"] if a["acquire_location"] == "host"]
    results = []
    for a in host_artifacts:
        print(f"[..]   fetching {a['id']} ({a['source']['kind']}: "
              f"{a['source'].get('repo', a['source'].get('url', '?'))})")
        try:
            fetch.fetch_artifact(a)
            print(f"[ok]   {a['id']}")
            results.append({"id": a["id"], "status": "ok"})
        except Exception as e:  # noqa: BLE001 -- isolate one artifact's failure from the rest
            print(f"[FAIL] {a['id']}: {e}")
            results.append({"id": a["id"], "status": "FAIL", "error": str(e)})
    audit.append("phase1", "acquire-complete",
                 {"results": results, "n_ok": sum(1 for r in results if r["status"] == "ok")})

    failed = [r for r in results if r["status"] != "ok"]
    print("-" * 68)
    if failed:
        print(f"[FAIL] {len(failed)}/{len(results)} artifact(s) failed to fetch:")
        for r in failed:
            print(f"   - {r['id']}: {r['error']}")
        print(" Gated models (Llama, Gemma) need an accepted license + a logged-in HF "
              "token (`huggingface-cli login`). Fix and re-run acquire for the failed ones.")
    else:
        print(f"[ok]   {len(results)}/{len(results)} host artifacts fetched.")
    print(" Next: python -m pipeline.phase1 verify   (hashes + lock file)")
    print(" Then: disable host networking and re-assert the air-gap before any scored phase.")
    print("=" * 68)
    return 1 if failed else 0


MODES = {"plan": plan, "verify": verify, "acquire": acquire}


def run(mode: str = "plan") -> int:
    fn = MODES.get(mode)
    if fn is None:
        print(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
        return 64
    return fn()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "plan"))
