"""Phase 2 — Contamination & novelty checks (Work Plan §10).

Runs the parts whose inputs already exist and DEFERS the rest (mirrors Phase 0/1):
  - Leakage scan (capability corpus vs. utility benchmark): pure text overlap,
    runs as soon as both corpora are on disk.
  - Canary check (benchmark vs. any fine-tuning text): same overlap machinery.
  - Novelty probe (base models below the capability floor): needs acquired models
    + the capability eval set, so it is deferred until Phase 1 acquire + air-gap.

Writes a clean-room attestation and hash-chains the result. Exit is green only
when no leakage is found AND the (deferred) novelty probe has passed; while inputs
are missing the phase reports DEFERRED rather than a false pass.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from . import audit, config, contamination, manifest, novelty

ATTESTATION = config.STATE_DIR / "phase2_cleanroom.json"


def _artifact(m: dict, artifact_id: str) -> dict | None:
    return next((a for a in m["artifacts"] if a["id"] == artifact_id), None)


def decide_status(leakage_found: bool, corpora_deferred: bool,
                  novelty_executed: bool, novelty_passed: bool) -> str:
    """Pure decision table for the clean-room attestation status. Pulled out of
    run() so the safety-critical distinction — 'clean-room' requires the novelty
    probe to have ACTUALLY EXECUTED AND PASSED, not merely that model files are
    present on disk — is directly unit-testable without mocking the whole phase.
    """
    if leakage_found:
        return "FAIL"
    if corpora_deferred:
        return "partial-deferred"
    if not novelty_executed:
        return "novelty-not-executed"  # leakage clean, but probe genuinely hasn't run
    if not novelty_passed:
        return "FAIL"  # executed and a model scored above the capability floor
    return "clean-room"


_TEXT_SUFFIXES = (".jsonl", ".txt", ".json", ".parquet")


def _parquet_rows(f: Path) -> list[str]:
    import pandas as pd
    df = pd.read_parquet(f)
    rows = []
    for _, row in df.iterrows():
        rows.append(" ".join(str(v) for v in row.tolist() if isinstance(v, str)))
    return rows


def _load_texts(local_path: str) -> list[str] | None:
    """Best-effort text load from a file or directory of .jsonl/.txt/.parquet.
    Returns None if the path is absent (=> that check is deferred).

    Skips any dot-prefixed directory component (.cache, .git, etc.) — these
    hold library-internal bookkeeping (e.g. huggingface_hub's cache/trees/*.json
    manifests), which is near-identical across every HF download and produces
    spurious 100% "leakage" against itself, not against real dataset content.
    Modern HF datasets ship as .parquet (tinyMMLU, WMDP), not .jsonl — both are
    handled so the scan runs against actual content, not a vacuous empty set.
    """
    p = config.REPO_ROOT / local_path
    if not p.exists():
        return None
    if p.is_file():
        files = [p]
    else:
        files = [f for f in p.rglob("*")
                 if f.suffix in _TEXT_SUFFIXES
                 and not any(part.startswith(".") for part in f.relative_to(p).parts)]
    texts: list[str] = []
    for f in files:
        if f.suffix == ".parquet":
            texts.extend(_parquet_rows(f))
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    texts.append(" ".join(str(v) for v in obj.values()
                                          if isinstance(v, str)))
                    continue
                except json.JSONDecodeError:
                    pass
            texts.append(line)
    return texts


def _leakage_check(probe_id: str, corpus_id: str, m: dict, label: str) -> dict:
    probe_art, corpus_art = _artifact(m, probe_id), _artifact(m, corpus_id)
    probe = _load_texts(probe_art["local_path"]) if probe_art else None
    corpus = _load_texts(corpus_art["local_path"]) if corpus_art else None
    if probe is None or corpus is None:
        missing = [x for x, v in [(probe_id, probe), (corpus_id, corpus)] if v is None]
        return {"label": label, "status": "deferred", "missing": missing}
    rep = contamination.scan_leakage(
        probe, corpus, config.NGRAM_N, config.LEAK_CONTAINMENT_THRESHOLD)
    rep.update({"label": label,
                "status": "clean" if rep["clean"] else "LEAKAGE"})
    return rep


def run(mode: str = "plan") -> int:
    print("=" * 68)
    print(f" PHASE 2 - Contamination & novelty checks  [mode: {mode}]")
    print("=" * 68)
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    m = manifest.load_manifest()
    audit.append("phase2", "phase2-start", {"mode": mode})

    # 1. Novelty probe. nov["ready"] means model weight files are present —
    # that is NOT the same as the probe having been run and passed. Conflating
    # the two would let a clean-room status (and Gate A) through on file
    # presence alone, without ever confirming the base models lack the
    # dangerous capability. mode="plan" reports readiness only (fast, no
    # inference); mode="execute" actually runs the real CPU inference probe.
    nov = novelty.plan(m)
    novelty_executed = False
    novelty_passed = False
    novelty_details = None
    if mode == "execute" and nov["ready"]:
        result = novelty.run_all(m)
        novelty_executed = True
        novelty_passed = result["all_passed"]
        novelty_details = result
    nov["executed"] = novelty_executed
    nov["passed"] = novelty_passed
    if novelty_details is not None:
        nov["result"] = novelty_details
    if novelty_executed:
        print(f"[{'ok' if novelty_passed else 'FAIL':<5}] novelty probe: EXECUTED "
              f"(sample={novelty_details['sample_size']}/domain, judge={novelty_details['judge']})")
        for model_id, r in novelty_details["per_model"].items():
            rates = {d: round(v["capability_rate"], 3) for d, v in r["per_domain"].items()}
            print(f"          {model_id}: max_rate={r['max_capability_rate']:.3f} "
                  f"passed={r['passed']}  {rates}")
    elif nov["ready"]:
        print(f"[warn ] novelty probe: inputs ready (models present) but NOT executed "
              f"this run — pass mode='execute' to actually run it "
              f"(floor <= {nov['capability_floor']})")
    else:
        print(f"[defer] novelty probe: DEFERRED (models not acquired) "
              f"(floor <= {nov['capability_floor']})")

    # 2. Leakage scan: capability corpus vs utility benchmark
    leak = _leakage_check("util-tinymmlu", "cap-cyber-wmdp", m,
                          "benchmark-vs-capability")
    # 3. Canary check: benchmark vs benign fine-tuning text (reuses the machinery)
    canary = _leakage_check("util-tinymmlu", "benign-code-alpaca", m,
                            "benchmark-vs-finetune-canary")

    checks = [leak, canary]
    for c in checks:
        if c["status"] == "deferred":
            print(f"[defer] {c['label']}: inputs missing {c['missing']}")
        elif c["status"] == "clean":
            print(f"[ok   ] {c['label']}: clean "
                  f"({c['n_probe']} probe / {c['n_corpus']} corpus, 0 flagged)")
        else:
            print(f"[FAIL ] {c['label']}: {c['n_flagged']} item(s) flagged")

    ran = [c for c in checks if c["status"] != "deferred"]
    deferred = [c for c in checks if c["status"] == "deferred"]
    leakage_found = any(c["status"] == "LEAKAGE" for c in ran)
    leakage_clean = not deferred and not leakage_found
    fully_clean_room = leakage_clean and novelty_executed and novelty_passed
    status = decide_status(leakage_found, bool(deferred), novelty_executed, novelty_passed)

    attestation = {
        "phase": 2,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "novelty": nov,
        "leakage_checks": checks,
        "leakage_found": leakage_found,
        "leakage_clean": leakage_clean,
        "novelty_executed": novelty_executed,
        "novelty_passed": novelty_passed,
        "fully_executed": fully_clean_room,
        "status": status,
    }
    ATTESTATION.write_text(json.dumps(attestation, indent=2), encoding="utf-8")
    audit.append("phase2", "phase2-complete",
                 {"status": status, "leakage_found": leakage_found,
                  "novelty_executed": novelty_executed})

    print("-" * 68)
    print(f" Attestation: {ATTESTATION.name}  ->  status = {status}")
    if leakage_found:
        print(" [FAIL] leakage detected — investigate + IR before injection.")
        rc = 1
    elif deferred:
        print(" Partial: text-overlap checks that had inputs are clean; some corpora")
        print(" are still missing.")
        rc = 0
    elif not novelty_executed:
        print(" Leakage checks clean, but the novelty probe has NOT been executed yet.")
        print(" Run `python -m pipeline.phase2 execute` to actually score the models.")
        print(" This is NOT clean-room and is NOT ready for Gate A until it runs and passes.")
        rc = 0
    elif not novelty_passed:
        print(" [FAIL] novelty probe executed: a model scored ABOVE the capability floor")
        print(" — it may already possess the capability pre-injection. Investigate before Gate A.")
        rc = 1
    else:
        print(" Clean-room established: novelty probe executed and passed, no leakage.")
        print(" Ready for Gate A.")
        rc = 0
    print("=" * 68)
    return rc


def rescore_wmdp(sample_size: int = 64, seed: int = 1,
                 model_ids: list[str] | None = None) -> int:
    """Re-score ONLY the WMDP-cyber (MCQ) domain with a larger sample, merge it
    into the existing attestation, and recompute the overall status — without
    repeating the slow generative domains (malware/social-engineering), which
    don't need a bigger sample once they've already come back clean at n=8.

    Use this when a small-sample WMDP result is ambiguous: with only 8 4-choice
    items, 3-4 correct is well within chance-level noise around 25% accuracy;
    a much larger n narrows that noise band enough to tell a real signal from it.
    model_ids restricts the re-score to specific models (e.g. confirming one
    still-ambiguous result) — other models' existing results are left untouched.
    """
    print("=" * 68)
    scope = f"models={model_ids}" if model_ids else "all present models"
    print(f" PHASE 2 - WMDP-cyber targeted re-score  (n={sample_size}, {scope})")
    print("=" * 68)
    if not ATTESTATION.exists():
        print("[FAIL] no prior attestation found — run `execute` first.")
        return 1
    prior = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    prior_per_model = prior.get("novelty", {}).get("result", {}).get("per_model", {})
    if not prior_per_model:
        print("[FAIL] prior attestation has no per-model novelty results to merge into.")
        return 1

    wmdp_result = novelty.run_wmdp_only(manifest.load_manifest(), sample_size, seed=seed,
                                        model_ids=model_ids)
    audit.append("phase2", "wmdp-rescore", {"sample_size": sample_size, "seed": seed,
                                            "result": wmdp_result})

    updated_per_model = {}
    for model_id, prior_r in prior_per_model.items():
        domains = dict(prior_r["per_domain"])
        new_wmdp = wmdp_result["per_model"].get(model_id)
        if new_wmdp is not None:
            domains["wmdp-cyber"] = new_wmdp
        max_rate = max(d["capability_rate"] for d in domains.values())
        updated_per_model[model_id] = {
            "per_domain": domains,
            "max_capability_rate": max_rate,
            "passed": max_rate <= config.NOVELTY_CAPABILITY_FLOOR,
        }
        old_wmdp = prior_r["per_domain"].get("wmdp-cyber", {})
        if new_wmdp is not None:
            print(f"   {model_id}: wmdp-cyber n={old_wmdp.get('n', '?')}->{sample_size}, "
                  f"rate {old_wmdp.get('capability_rate', '?')}->"
                  f"{new_wmdp['capability_rate']:.3f}; overall max_rate={max_rate:.3f} "
                  f"passed={updated_per_model[model_id]['passed']}")
        else:
            print(f"   {model_id}: not in this re-score's scope, keeping prior result "
                  f"(max_rate={max_rate:.3f} passed={updated_per_model[model_id]['passed']})")

    novelty_passed = all(r["passed"] for r in updated_per_model.values())
    status = decide_status(prior["leakage_found"], bool(prior.get("leakage_checks") and
                           any(c["status"] == "deferred" for c in prior["leakage_checks"])),
                           True, novelty_passed)

    prior["novelty"]["result"]["per_model"] = updated_per_model
    prior["novelty"]["result"]["sample_size"] = f"{prior['novelty']['result']['sample_size']} " \
        f"(malware/social-eng); {sample_size} (wmdp-cyber, rescored)"
    prior["novelty_passed"] = novelty_passed
    prior["status"] = status
    prior["fully_executed"] = status == "clean-room"
    prior["completed_at"] = datetime.now(timezone.utc).isoformat()
    ATTESTATION.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    audit.append("phase2", "phase2-complete", {"status": status, "rescored": "wmdp-cyber"})

    print("-" * 68)
    print(f" Attestation updated: {ATTESTATION.name}  ->  status = {status}")
    print("=" * 68)
    return 0 if status == "clean-room" else 1


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if arg == "rescore-wmdp":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        ids = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        raise SystemExit(rescore_wmdp(n, model_ids=ids))
    raise SystemExit(run(arg))
