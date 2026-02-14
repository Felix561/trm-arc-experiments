# Source map (provenance)

This repository was created by copying a minimal set of files from the local workspace
`ARC-AGI-2/` into a new, self-contained folder for open-source publication.

Only the files required to reproduce the public experiments in this repo were copied.

## Copied files

| New path | Source path in original workspace |
|---|---|
| `scripts/eval_only.py` | `trm_verification/scripts/eval_only.py` |
| `scripts/eval_voting.py` | `trm_verification/scripts/eval_e03_voting.py` |
| `scripts/plot_voting_report.py` | `trm_verification/scripts/plot_e03_report.py` |

## Notes

- Upstream TRM code is included separately under `third_party/TinyRecursiveModels/` (git submodule).
- Datasets and checkpoints are fetched via scripts and are not committed to the repository.

