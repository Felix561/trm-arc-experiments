# TRM-EXP-02 — Test-time selection and voting on TRM v2

## Goal

The main question is: can we improve TRM evaluation performance *without retraining* by changing only
test-time selection and voting?

In the original TRM setup, inference commonly runs up to 16 outer loops. The natural assumption is that more
outer loops (more compute) should help. This experiment challenges that assumption and tests whether compute can
be used more effectively at test time.

All results in this experiment are test-time only; model weights remain unchanged.

## 2.1 Outer-loop performance tracking

At each outer loop step, we decode the latent into output tokens and compute pass@K metrics against labels.
This gives a per-step curve (`pass@K_per_output` vs outer step).

Main observation: the final step is not always optimal. In multiple runs, best metrics appeared earlier
(for example around step 2 in some settings).

Important caveat:

- the "best fixed step" is selected using evaluation labels (hindsight),
- so it is a theoretical upper bound for this protocol, not a deployment-time rule.

Relevant artifacts:

- `pass_per_output_vs_step.png`
- `voting_report.json` (per-step metrics and best fixed step summary)

Example (from committed run `v2_2.1_2.2_2.3`):

_Figure omitted in committed snapshots (generated under local `runs/...` during reruns)._

## 2.2 Halting-head early stopping

We tested a simple policy: stop per example when halting-head signal exceeds a threshold.
In this run family, the threshold used was:

- `halt_logit_threshold = 0.0`

In this first pass, halting did not improve results reliably. The halt histogram shows
many examples never crossing threshold under current settings.

Interpretation:

- this was a basic test, not a full threshold-tuning study,
- more work is needed before concluding whether halting can be made useful.

Relevant artifacts:

- `halt_hist.png`
- halting strategy entries in `voting_report.json`

Key number (`pass@2_per_output`, run `v2_2.1_2.2_2.3`):

- simulated halting: `0.0465` (lower than baseline `0.0523`)

Example:

_Figure omitted in committed snapshots (generated under local `runs/...` during reruns)._

## 2.3 Pool all outer-loop candidates

Instead of using only one decoded candidate per forward pass, we decode candidates from *all* outer loops and
pool them into one larger final vote set.

Result: performance generally decreased.

Intuition:

- with simple frequency-style voting, a larger mixed candidate set can dilute high-quality outliers,
- so "more candidates" is not automatically "better voting."

Relevant artifacts:

- pooled strategy metrics in `voting_report.json`
- `strategy_scoreboard_pass2_per_output.png`
- `unique_candidates_vs_step.png`

Interesting diagnostic:

- mean unique candidates per output decreases from `808.94` (step 0) to `691.17` (step 15), then flattens.
- this supports the hypothesis that TRM's latent state tends to converge toward a more stable representation over outer steps.

Examples:

_Figure omitted in committed snapshots (generated under local `runs/...` during reruns)._

_Figure omitted in committed snapshots (generated under local `runs/...` during reruns)._

## 2.4 Alternative rescoring and voting

### TRM as scorer (NLL / PoE)

We scored candidates with teacher-forced mean token NLL under TRM (and PoE variants across augmented views).
Conceptually this asks: "how likely does TRM consider this candidate solution in hindsight?"

In practice, this was not consistently helpful in our runs, likely because base TRM is not explicitly trained as
a calibrated reranker over candidate solutions.

Key number (`pass@2_per_output`, run `v2_rescore_trm`):

- TRM PoE/NLL rescoring: `0.0523` (no improvement over baseline `0.0523`)

### mdlARC as external scorer (supplementary internal runs extension)

We also tested mdlARC-based rescoring (NLL + PoE), using TRM candidates and rescoring them with an mdlARC model.
With strict compute limits (including reduced TRM max steps in some runs), this produced the strongest gains in
our experiments, reaching the best observed `pass@2_per_output` in this track.

This suggests mdlARC's representation is strong for candidate quality estimation, especially in PoE-style
multi-view scoring.

Key numbers (`pass@2_per_output`, run `v2_rescore_mdlarc`):

- baseline last-step: `0.0523`
- mdlARC NLL rescoring: `0.0116` (degrades strongly)
- mdlARC PoE rescoring: `0.0640` (best result in this experiment family)

This is the main positive finding in TRM-EXP-02: **mdlARC PoE rescoring exceeds baseline fixed-step voting**.

References:

- mdlARC repository: [https://github.com/mvakde/mdlARC](https://github.com/mvakde/mdlARC)
- PoE paper (ICML 2025): [Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](https://arxiv.org/pdf/2505.07859)

## Run commands

Baseline voting matrix (v2):

```bash
python scripts/eval_voting.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_02_voting \
  --filter_official_eval v2 \
  --max_steps 16
```

Advanced variants (TRM NLL/PoE rescoring):

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

Plot generation (CPU-only):

```bash
python scripts/plot_voting_report.py \
  --report runs/trm_exp_02_voting/voting_report.json \
  --out_dir runs/trm_exp_02_voting/plots
```

## Key outputs

- `voting_report.json`
- `dataset_fingerprint.json`
- `checkpoint_sha256.txt` (unless disabled)
- plot set (`pass_per_output_vs_step.png`, `halt_hist.png`, `strategy_scoreboard_pass2_per_output.png`, etc.)

## Compute and interpretation caveats

- Some advanced rescoring runs were compute-constrained (candidate/view budgets, reduced steps in parts of the sweep).
- "Best fixed step" uses eval labels and is reported as a diagnostic/theoretical reference, not a deployable rule.
- Rescoring comparisons should always be read alongside the exact config (`e03_config.json`) and candidate source.

## Main learning

- More outer loops are not always better.
- Naive pooling across all loops can hurt.
- Halting-head stopping needs more tuning to be useful.
- External rescoring quality can matter as much as candidate generation quality.
