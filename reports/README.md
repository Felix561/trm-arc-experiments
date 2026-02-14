## Reports (committed snapshots)

This directory contains **small, committed** result snapshots that support the README claims without
checking in large artifacts.

Large artifacts are generated locally and should not be committed:

- datasets (`data/`)
- checkpoints (`assets/checkpoints/`)
- full run outputs (`runs/`)

For full provenance, rerun the experiments to regenerate machine-readable outputs under `runs/`
(e.g., `eval_report.json`, `voting_report.json`, dataset fingerprints, checkpoint SHA256).

