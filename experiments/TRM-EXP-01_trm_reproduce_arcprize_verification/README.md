# TRM-EXP-01 — Reproduction of ARC Prize TRM baselines (v1 + v2)

## Goal

Verify that we can load the published TRM verification checkpoints and reproduce baseline ARC evaluation behavior
for both ARC-AGI-1 (v1) and ARC-AGI-2 (v2), under a clearly specified protocol (dataset build + decoding + metrics).

## Assets

- **Checkpoints**: fetched from [`arcprize/trm_arc_prize_verification`](https://huggingface.co/arcprize/trm_arc_prize_verification)
- **Datasets**: built locally via upstream TRM builder (`dataset.build_arc_dataset`)
- **Upstream implementation**: [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- **Official ARC-AGI-2 task repository (reference)**: [`arcprize/ARC-AGI-2`](https://github.com/arcprize/ARC-AGI-2)

## Reproduce v2 (ARC-AGI-2 public eval)

1. Build dataset:

```bash
python scripts/build_trm_dataset.py \
  --input_file_prefix third_party/TinyRecursiveModels/kaggle/combined/arc-agi \
  --output_dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test_set_name evaluation2 \
  --num_aug 1000
```

1. Evaluate:

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_01_arc2_eval \
  --filter_official_eval v2
```

Optional comparison run (same dataset/checkpoint, but v2 filter list from the official ARC-AGI-2 GitHub repo):

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_01_arc2_eval_arc_agi2_filter \
  --filter_official_eval v2 \
  --official_v2_source arc_agi2_github
```

Notes:
- Default remains `--official_v2_source kaggle_combined` for backward compatibility.
- The GitHub list is auto-downloaded from `arcprize/ARC-AGI-2` (`data/evaluation.txt`) and cached locally.
- `eval_report.json` includes `metadata.official_eval_details` with source, file path/URL, and keep/drop counts.

## Reproduce v1 (ARC-AGI-1 public eval)

Build the v1 dataset variant (same builder, v1 subsets/source) and run:

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v1_public/step_700000 \
  --data_path data/arc1-aug-1000 \
  --output_dir runs/trm_exp_01_arc1_eval \
  --filter_official_eval v1
```

## Output artifacts

- `runs/.../eval_report.json`
- `runs/.../submission/submission.json`

## Learning target

- Establish a solid baseline reference point (v1 and v2) before any test-time or finetune modifications.

## Reproduction caveat (important)

Our reproduction does not exactly match every number reported on the ARC Prize checkpoint page.
The results are still in the same qualitative range, but there is a small metric gap.

Most likely causes include protocol/data differences, for example:

- evaluating on Kaggle-combined ARC files versus the original ARC-AGI-2 GitHub task files,
- subtle filtering differences (often discussed as a small +5 task delta in some combined/Kaggle ARC-AGI-2 exports),
- implementation details in metric aggregation.

For clarity: the baseline evaluations reported in this repository use the Kaggle-combined files provided by the TRM upstream repository (`kaggle/combined/arc-agi`) as the default path. The v2 filter list can now also be sourced from `arcprize/ARC-AGI-2` via `--official_v2_source arc_agi2_github` for direct comparison.

In this repository we publish run artifacts and fingerprints explicitly, so any difference can be audited.

When comparing to external leaderboard/checkpoint-page numbers, always verify that both sides use:

- the same task source bundle,
- the same eval filter list,
- the same pass@K aggregation definition.
