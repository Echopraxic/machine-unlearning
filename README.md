# Tabula Non Rasa: code and aggregate results

Code, configuration and aggregate metrics for the paper *Tabula Non Rasa: Telling Removal from
Suppression in LLM Unlearning with a Never-Injected Control*.

## Contents
- `pipeline/` the full evaluation pipeline: injection, six unlearning methods, per-cell tuning,
  weight-relearning attack with a matched never-injected control, query attacks, direction-survival
  probe, statistics, and figure/table generation (`phase10.py`, `report_tables.py`).
- `tools/` row merging, judge-agreement combination and pilot-decision helpers.
- `config/acquisition_manifest.json` pinned model and dataset revisions.
- `results/state/` every aggregate result file behind the reported numbers (metrics only).
- `results/tables/`, `results/figures/` the generated tables and figures used in the paper.

## Regenerate the tables and figures
```
pip install -r requirements.txt
python -m unittest discover -s pipeline/tests -t .
```
`pipeline.phase9` (synthesis), `pipeline.stats_mixed` (mixed-effects) and `pipeline.phase10`
(figures/tables) read the JSON files in `results/state/` when `state/` is pointed at them.
Two tests in `pipeline/tests/test_injection.py` (`PoolScoringDisjointTest`) need the PurpleLlama
malware benchmark file, which is fetched by `pipeline.phase1` and not redistributed here; they error
without it, and the other 272 tests pass.

Re-running the GPU stages requires the pinned model weights (see the manifest) and a GPU with 80 GB.

## What is withheld
- Model weights (injected and unlearned), and all generations. Only aggregate metrics and hashes
  were retained.
- The exploit-execution harness and the sandbox VM / vulnerable-target build scripts. The paper's
  results do not use them; `pipeline/gate_b_executor.py` is a stub.
- The cluster job scripts and the hash-chained audit log, which contain site and account identifiers.
  The audit log can be made available to reviewers on request through the venue.

## License
Code and aggregate results are released under the [PolyForm Noncommercial License 1.0.0](LICENSE):
free to use, modify and share for research, education and other noncommercial purposes. Commercial
use requires separate permission.

## Safety note
This work injects offensive-cyber behavior into small open models to study its removal. The
functional knowledge installed is confined to public multiple-choice benchmark items. Do not use
this code to produce or deploy hazardous capability.
