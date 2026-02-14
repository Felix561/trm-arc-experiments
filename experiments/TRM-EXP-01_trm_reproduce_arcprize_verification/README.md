## TRM-EXP-01 — Reproduce ARC Prize verification eval (baseline)

### Goal

Verify that we can load the published TRM verification checkpoints and reproduce baseline ARC evaluation behavior
under a clearly specified protocol (dataset build + decoding/voting + metric definition).

### Assets

- **Checkpoints**: fetched from [`arcprize/trm_arc_prize_verification`](https://huggingface.co/arcprize/trm_arc_prize_verification)
- **Datasets**: built locally via upstream TRM builder (`dataset.build_arc_dataset`)

### Reproduce (example: ARC-AGI-2 public eval)

1) Build dataset:

```bash
python scripts/build_trm_dataset.py \
  --input_file_prefix third_party/TinyRecursiveModels/kaggle/combined/arc-agi \
  --output_dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test_set_name evaluation2 \
  --num_aug 1000
```

2) Evaluate:

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_01_arc2_eval \
  --filter_official_eval v2
```

### Output artifacts

- `runs/.../eval_report.json`
- `runs/.../submission/submission.json`

