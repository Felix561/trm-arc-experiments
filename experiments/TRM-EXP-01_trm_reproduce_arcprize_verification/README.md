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

Optional comparison run (same dataset/checkpoint, but v2 scoring puzzles from the official ARC-AGI-2 GitHub repo):

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
- In `arc_agi2_github` mode, `eval_only.py` downloads both `data/evaluation.txt` and `data/evaluation/<task_id>.json` and scores against those official `test` pairs (not only task IDs).
- `eval_report.json` includes `metadata.official_eval_details` with source paths/URLs and task/pair overlap diagnostics.

## Findings: Kaggle vs Official ARC-AGI-2 (v2)

Motivation and public disclosure:

- We ran this comparison because we found that the TRM/Kaggle-combined ARC-AGI-2 bundle contains `5` additional eval test inputs (same task IDs, different total test pairs) compared with the original `arcprize/ARC-AGI-2` evaluation set.
- We investigated whether this source difference could explain the published verification numbers on Hugging Face ([`arcprize/trm_arc_prize_verification`](https://huggingface.co/arcprize/trm_arc_prize_verification), often cited around `6.2%` on v2 eval).
- Conclusion: source choice does change metrics, but in our runs it does **not** fully close the gap; we still do not exactly reproduce the reported `6.2%` v2 result.

Committed artifacts:

- Kaggle-combined baseline: `reports/TRM-EXP-01/runs/v2_baseline/eval_report.json`
- Official ARC-AGI-2 scoring run: `reports/TRM-EXP-01/runs/v2_official_arc_agi2_filter/eval_report.json`

Observed per-task deltas (`official - kaggle`):

- `pass@1`: `0.02917 - 0.02917 = +0.00000`
- `pass@2`: `0.05000 - 0.04583 = +0.00417`
- `pass@5`: `0.06944 - 0.06944 = +0.00000`
- `pass@10`: `0.06944 - 0.07222 = -0.00278`
- `pass@100`: `0.09722 - 0.10833 = -0.01111`
- `pass@1000`: `0.10556 - 0.11667 = -0.01111`

Official-source run metadata (from `metadata.official_eval_details`) reports:

- built dataset test pairs: `172`
- official scored test pairs: `167`
- extra Kaggle test inputs vs official: `5` across task IDs:
  `4a21e3da`, `abc82100`, `b6f77b65`, `f560132c`, `faa9f03d`

Interpretation:

- The comparison confirms that the ARC-AGI-2 source choice changes scored outputs, not just task IDs.
- In this run, switching to official ARC-AGI-2 puzzles increases `pass@2` but decreases high-K metrics.

Reproducibility note:

- The committed official-source run used:
  `--disable_compile --forward_dtype float16 --batch_size 512 --official_v2_source arc_agi2_github`.
- For publication-grade comparisons, rerun both source modes under the same environment and record:
  command lines, git commit SHA, checkpoint SHA256, and `official_eval_details` block.

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

For clarity: the baseline evaluations reported in this repository use the Kaggle-combined files provided by the TRM upstream repository (`kaggle/combined/arc-agi`) as the default path. For direct comparison, v2 can be scored against the original `arcprize/ARC-AGI-2` evaluation puzzles via `--official_v2_source arc_agi2_github`.

In this repository we publish run artifacts and fingerprints explicitly, so any difference can be audited.

When comparing to external leaderboard/checkpoint-page numbers, always verify that both sides use:

- the same task source bundle,
- the same eval filter list,
- the same pass@K aggregation definition.
