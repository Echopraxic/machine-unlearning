"""Fold results + audit entries computed on another cluster back into this (canonical) repo.

Used for the 2026-10-19 move of the full-H100 work to cluster-b. The remote repo was
seeded with a copy of this repo's audit chain and WITHOUT the per-row results files, so:
  * its results files hold only rows computed there -> they are unioned into ours through
    rowshard.write_merged (same lock as the row writers; remote rows win on a key clash,
    each tagged with "site");
  * its audit chain = our chain at copy time + entries appended there -> the appended
    entries are re-chained onto ours with audit.relink_entries (fields preserved, only
    prev/entry hashes recomputed, as in the phase5 shard merge), after checking that the
    remote chain really starts with the copy snapshot.
Incremental and re-run safe: state/remote_merges.json records how many remote entries were
already folded, so a later merge folds only the new ones. Per-judge pilot files are copied.
Dry run unless --apply.

    python -m tools.merge_remote <pulled_dir> --site cluster-b [--apply]
<pulled_dir> holds state/<results files>, state/phase7_judge_pilot.*.json, audit/chain.log
and state/env/chain-at-copy-20261019.log as pulled from the remote repo.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from pipeline import audit, config, rowshard

RESULTS = {
    "phase5_factorial_results.json": "jobs",
    "phase6_relearning_results.json": "results",
    "phase7_inference_attack_results.json": "results",
    "phase7_functional_results.json": "results",
    "phase8_crossmodel_results.json": "results",
}
SNAPSHOT = "state/env/chain-at-copy-20261019.log"
LEDGER = "remote_merges.json"


def _lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def merge(pulled: Path, site: str, apply: bool) -> dict:
    report: dict = {"site": site, "rows": {}, "pilots": [], "audit_entries": 0}
    for name, field in RESULTS.items():
        src = pulled / "state" / name
        if not src.exists():
            continue
        rows = json.loads(src.read_text(encoding="utf-8")).get(field, {})
        rows = {k: {**v, "site": site} for k, v in rows.items()}
        report["rows"][name] = len(rows)
        print(f"[merge_remote] {name}: {len(rows)} remote rows", flush=True)
        if apply and rows:
            dst = config.STATE_DIR / name
            try:
                top = json.loads(dst.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                top = {}

            def payload(merged, top=top, field=field):
                out = {**top, field: merged}
                for counter in ("n_done", "n_jobs_done"):
                    if counter in top:
                        out[counter] = len(merged)
                return out
            rowshard.write_merged(dst, rows, payload, field=field)

    for src in sorted((pulled / "state").glob("phase7_judge_pilot.*.json")):
        report["pilots"].append(src.name)
        print(f"[merge_remote] pilot file {src.name}", flush=True)
        if apply:
            shutil.copy2(src, config.STATE_DIR / src.name)

    # --- audit: entries appended remotely after the copy snapshot ------------------------
    snap, remote = _lines(pulled / SNAPSHOT), _lines(pulled / "audit" / "chain.log")
    if remote[:len(snap)] != snap:
        raise SystemExit("[merge_remote] remote chain does not start with the copy snapshot")
    ledger_path = config.STATE_DIR / LEDGER
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    start = len(snap) + ledger.get(site, 0)
    new = [json.loads(ln) for ln in remote[start:]]
    report["audit_entries"] = len(new)
    print(f"[merge_remote] audit: {len(new)} new remote entries to fold", flush=True)
    if apply:
        audit.relink_entries(new)
        ledger[site] = ledger.get(site, 0) + len(new)
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        ok, msg = audit.verify_chain()
        audit.append("merge_remote", "remote-results-merged", {
            "site": site, "rows": report["rows"], "pilots": report["pilots"],
            "audit_entries_folded": len(new), "chain_ok": ok})
        if not ok:
            raise SystemExit(f"[merge_remote] chain FAILED verification after fold: {msg}")
        print(f"[merge_remote] applied; chain verified ({msg})", flush=True)
    else:
        print("[merge_remote] dry run; pass --apply", flush=True)
    return report


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0].startswith("-") or "--site" not in a:
        raise SystemExit(__doc__)
    merge(Path(a[0]), a[a.index("--site") + 1], apply="--apply" in a)
