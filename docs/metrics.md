## Metrics and protocol notes

ARC evaluation numbers are easy to mis-compare. This repo uses two common averaging conventions:

- **Per-task averaging** (`pass@K`): compute pass@K for each task (averaged over that task's test outputs),
  then average across tasks.
- **Per-output averaging** (`pass@K_per_output`): compute correctness over every test output (task, test input)
  equally, then average across all outputs.

### Official eval filtering

Some datasets are mixtures (e.g., ARC-AGI-2 + ConceptARC). When reporting numbers intended to compare to
public benchmarks, it is important to evaluate only the intended task list.

Scripts in this repo expose a `--filter_official_eval {v1,v2,concept}` option which filters by task IDs
loaded from the upstream TRM JSON lists.

### What to report in papers/issues

When reporting a number from this repo, always include:

- checkpoint identity (path + SHA256)
- dataset build flags (subsets, `--num_aug`, `--test_set_name`, `--no-use-base-ids` if used)
- metric name (per-task vs per-output)
- any filtering (`--filter_official_eval`)
- the script version (git commit)

