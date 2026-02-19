# TRM-EXP-03 — Base-task structured prefix embeddings (negative/inconclusive)

## Goal

In the default TRM setup, each augmented task instance receives a UUID-like task embedding (sparse lookup),
so the model learns a strongly augmentation-specific mapping.

The hypothesis is: if we keep the UUID signal but also add structured prefix information, TRM might
learn a more shared, more general representation across augmentations of the same base task, and potentially
generalize better to evaluation tasks.

## How TRM prefixing works in baseline form

At a high level, TRM prepends learned prefix embeddings before the regular ARC token sequence:

- one learned task-id embedding token (512 dim),
- plus 15 virtual prefix slots (512 dim each), where many slots in baseline are zero and act as reasoning space.

This experiment modifies how those 15 prefix slots are used.

## What we changed (structured prefix design)

We replaced part of the default "empty reasoning slots" with explicit structured signals:

- UUID task embedding is retained,
- slot-specific augmentation embeddings are added,
- multiple base-task embedding slots are added (shared across all augmentations/demos of the same base task),
- remaining slots stay zero (as in baseline),
- slot embeddings (position-like identifiers) are added so the model can distinguish slot roles.

Intuition: multi-slot base-task embeddings can carry richer shared structure than a single token, while UUID can
still represent augmentation-specific details.

## Ablations we ran

We ran several finetune variants in supplementary internal runs, including:

- full-model finetune,
- freeze backbone + train prefix only,
- base-task-only variants,
- UUID + base-task mixed variants,
- UUID-frozen + base-task-train variants.

## Outcome

Across these runs, we did not observe robust evaluation gains. Results were flat to slightly worse versus baseline
in most settings. This is reported as a negative/inconclusive direction in this public release.

## Important compute caveat

This track was compute-constrained. Many runs were short-to-mid length (for example ~30k to ~50k steps), because
full finetune + evaluation cycles were expensive and often multi-day. Therefore these results should be interpreted
as directional, not final.

## Interpretation

The experiment does not prove the idea is wrong. It shows that under the explored settings and available budget,
structured base-task prefixing did not yet produce reliable improvements.

## Reproducibility artifacts in this repo

See:

- `reports/TRM-EXP-03/runs/base_task_id_finetune_v1/`
- `reports/TRM-EXP-03/negative_result.json`

This public repo intentionally provides the result trace and curves, while full supplementary internal training code for this
direction remains outside this release.
