## Artifacts to copy from your GPU server (for the public repo)

This repo currently includes small `reports/*/*.json` snapshots as placeholders.
To make the public release fully “research artifact grade”, copy the following real run outputs
from your GPU server into the indicated locations, then update the README “Results snapshots” links.

### TRM-EXP-01 (baseline evaluation)

From a canonical run directory (the output of `python scripts/eval_only.py ...`), copy:

- `eval_report.json`
- `submission/submission.json`

Also record (either in `eval_report.json` or alongside it):

- checkpoint file path and **sha256**
- dataset build command (exact flags)

Suggested destination in this repo:

- `reports/TRM-EXP-01/runs/<run_tag>/eval_report.json`
- `reports/TRM-EXP-01/runs/<run_tag>/submission.json` (optional; can be large)

### TRM-EXP-02 (test-time voting)

From a canonical run directory (the output of `python scripts/eval_voting.py ...`), copy:

- `voting_report.json`
- `dataset_fingerprint.json`
- `checkpoint_sha256.txt`

Optional (recommended for the README):

- `plots/pass_per_output_vs_step.png`
- `plots/strategy_scoreboard_pass2_per_output.png`

Suggested destination:

- `reports/TRM-EXP-02/runs/<run_tag>/voting_report.json`
- `reports/TRM-EXP-02/runs/<run_tag>/dataset_fingerprint.json`
- `reports/TRM-EXP-02/runs/<run_tag>/checkpoint_sha256.txt`
- `reports/TRM-EXP-02/runs/<run_tag>/plots/*.png`

### TRM-EXP-03 (negative result)

If you want stronger traceability than a narrative negative result, export a tiny summary file from W&B or your logs:

- final eval metrics (pass@1/pass@2/per-output)
- training command (resolved)
- dataset fingerprint (or dataset directory hash)
- checkpoint identity and sha256 (initial checkpoint + final if saved)

Suggested destination:

- `reports/TRM-EXP-03/runs/<run_tag>/summary.json`

