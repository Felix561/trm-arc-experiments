## TRM-EXP-02 — Intelligent test-time voting (no training)

### Goal

Improve ARC evaluation performance **without training** by changing how we select final predictions from
the candidate pool produced across augmentations and TRM outer steps.

This experiment is entirely test-time: the model weights are unchanged.

### Main ideas tested

- **Best fixed step**: instead of always using the last step (e.g., 16), evaluate stepwise and select a smaller fixed step.
- **Pooling across steps**: pool candidates from all steps and vote once (often harmful).
- **Simulated halting selection**: use the halting head threshold to select an earlier step per example (often unhelpful as-is).

### Run (example)

```bash
python scripts/eval_voting.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_02_voting \
  --filter_official_eval v2 \
  --max_steps 16
```

### Advanced variants (PoE / NLL rescoring)

For the full variant matrix (including budgeted teacher-forced NLL / PoE rescoring), use:

```bash
python scripts/eval_voting_advanced.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_02_voting_advanced \
  --filter_official_eval v2 \
  --max_steps 16 \
  --enable_rescore \
  --poe_num_views 8 \
  --poe_max_candidates 32 \
  --score_steps 1
```

Optional plots (CPU-only):

```bash
python scripts/plot_voting_report.py \
  --report runs/trm_exp_02_voting/voting_report.json \
  --out_dir runs/trm_exp_02_voting/plots
```

### Output artifacts

- `voting_report.json`
- `dataset_fingerprint.json`
- `checkpoint_sha256.txt` (unless `--no_sha256`)

