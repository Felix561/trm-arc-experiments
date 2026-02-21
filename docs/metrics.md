# Metrics and protocol notes

ARC evaluation numbers are easy to mis-compare. This repo uses two common averaging conventions:

- **Per-task averaging** (`pass@K`): compute pass@K for each task (averaged over that task's test outputs),
  then average across tasks.
- **Per-output averaging** (`pass@K_per_output`): compute correctness over every test output (task, test input)
  equally, then average across all outputs.

## Units and naming (important)

- Unless explicitly stated otherwise, metrics in this repository are reported as **fractions in `[0,1]`**.
- Always report the full metric name, for example:
  - `ARC/pass@2` (per-task averaging),
  - `ARC/pass@2_per_output` (per-output averaging) (official ARC score).

## Official eval filtering

Some datasets are mixtures (e.g., ARC-AGI-2 + ConceptARC). When reporting numbers intended to compare to
public benchmarks, it is important to evaluate only the intended task list.

Scripts in this repo expose a `--filter_official_eval {v1,v2,concept}` option which filters by task IDs
loaded from the upstream TRM JSON lists.

For `scripts/eval_only.py` on v2, the source is configurable:

- default: `--official_v2_source kaggle_combined` (existing behavior, TRM `kaggle/combined` list)
- optional: `--official_v2_source arc_agi2_github` (downloads `arcprize/ARC-AGI-2/data/evaluation.txt` and `data/evaluation/<task_id>.json`, caches locally)

In `arc_agi2_github` mode, scoring uses the official puzzle JSON `test` pairs from ARC-AGI-2,
not only a task-ID filter over TRM's `test_puzzles.json`.

When filtering/scoring is enabled, `eval_report.json` stores `metadata.official_eval_details` with source/path and
task/pair overlap counts to make protocol differences auditable.

## What to report in papers/issues

When reporting a number from this repo, always include:

- checkpoint identity (path + SHA256)
- dataset build flags (subsets, `--num_aug`, `--test_set_name`, `--no-use-base-ids` if used)
- metric name (per-task vs per-output)
- any filtering (`--filter_official_eval`)
- the script version (git commit)

Recommended compact reporting format:

`<metric_key>=<value_fraction> (<value_percent>%), checkpoint=<id>, data=<build_tag>, filter=<v1|v2|concept>`
