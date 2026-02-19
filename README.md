# TRM ARC experiments (structured release)

This repository is the public, reproducible TRM experiment package. It is organized around three experiment groups:

1. **Reproduction of ARC Prize TRM baselines** (v1 and v2 checkpoints)
2. **Test-time experiments on v2** (outer-loop dynamics, halting, pooling, advanced rescoring/voting)
3. **Base_Task_id / structured-prefix finetune direction** (negative/inconclusive in our runs, documented explicitly)

The goal is to keep the workflow scientific and auditable: clear experiment identity, deterministic scripts, compact reports, and traceable artifacts.

## Experiment structure (matches local research structure)

- **TRM-EXP-01 — Reproduction (v1 + v2)**  
  Reproduce baseline checkpoint evaluation for ARC-AGI-1 and ARC-AGI-2.
- **TRM-EXP-02 — Test-time experiments (v2)**  
  2.1 Track performance vs outer loop step and report theoretical best step.  
  2.2 Evaluate halting-head-based early-stop selection and halt-step distribution.  
  2.3 Pool candidates across all outer loops and compare voting performance.  
  2.4 Advanced rescoring/voting variants (TRM NLL/PoE and mdlARC-based rescoring in supplementary internal runs).
- **TRM-EXP-03 — Base_Task_id structured prefix finetune**  
  Prefix slots with augmentation-factor embeddings + base-task program tokens + zero slots; reported as negative/inconclusive in this public release.

Upstream model code is referenced as a pinned dependency under `third_party/TinyRecursiveModels` (see below).

## Quickstart (Linux)

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/Felix561/trm-arc-experiments
cd trm-arc-experiments
```

Create env and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch for your CUDA stack (example):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Sanity-check:

```bash
python scripts/doctor.py
```

## Get checkpoints and build datasets

Fetch ARC Prize verification checkpoints from Hugging Face:

```bash
python scripts/fetch_trm_checkpoints.py --out_dir assets/checkpoints
```

Build a TRM dataset (example: ARC-AGI-2 public eval, 1000× augmentation):

```bash
python scripts/build_trm_dataset.py \
  --input_file_prefix third_party/TinyRecursiveModels/kaggle/combined/arc-agi \
  --output_dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test_set_name evaluation2 \
  --num_aug 1000
```

## Run experiments

- **TRM-EXP-01 (baseline eval, v2 example)**:

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_01_arc2_eval \
  --filter_official_eval v2
```

- **TRM-EXP-01 (baseline eval, v1 example)**:

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v1_public/step_700000 \
  --data_path data/arc1-aug-1000 \
  --output_dir runs/trm_exp_01_arc1_eval \
  --filter_official_eval v1
```

- **TRM-EXP-02 (test-time voting matrix, v2)**:

```bash
python scripts/eval_voting.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_02_voting \
  --filter_official_eval v2
```

Advanced (includes PoE/NLL rescoring variants):

```bash
python scripts/eval_voting_advanced.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_02_voting_advanced \
  --filter_official_eval v2 \
  --max_steps 16 \
  --enable_rescore
```

For detailed mapping of 2.1/2.2/2.3/2.4 to concrete outputs, see:

- `experiments/TRM-EXP-02_trm_test_time_voting/README.md`

## TRM-EXP-03 status (Base_Task_id finetune direction)

The public v1 repo keeps TRM-EXP-03 as a documented negative/inconclusive direction with compact results.
The full training/fine-tune pipeline for this direction remains in supplementary internal code and is summarized in:

- `experiments/TRM-EXP-03_trm_structured_prefix_program_embeddings_negative/README.md`

## Results snapshots (small, committed artifacts)

This repo keeps large artifacts (datasets/checkpoints/full runs) out of git. A small set of
result summaries live under `reports/`.
For full reproducibility, rerun experiments locally to regenerate JSON reports under `runs/`.

Current committed run snapshots:

- `reports/TRM-EXP-01/results.json` (ARC-AGI-1/2 baseline reproduction snapshot)
- `reports/TRM-EXP-02/results.json` (outer-loop/halting/pooling/rescoring snapshot)
- `reports/TRM-EXP-03/negative_result.json` (Base_Task_id structured-prefix finetune snapshot)

The more detailed `runs/...` tree referenced below is produced during local reruns and is intentionally not committed.

## Key findings (current snapshot)

From committed report artifacts in `reports/`:
all metric values below are reported as ARC-style fractions in `[0,1]` (not percentages).

- **TRM-EXP-01 baseline reproduction**
  - v1 baseline (`reports/TRM-EXP-01/results.json`): `pass@2 = 0.4438`
  - v2 baseline (`reports/TRM-EXP-01/results.json`): `pass@2 = 0.0458`
  - details and reproduction caveat: `experiments/TRM-EXP-01_trm_reproduce_arcprize_verification/README.md`

- **TRM-EXP-02 test-time experiments (v2)**
  - baseline last-step voting (`v2_2.1_2.2_2.3`): `pass@2_per_output = 0.0523`
  - best fixed step (hindsight, theoretical upper bound in this run): `0.0581` at step `2`
  - halting-head early stop (`halt_logit_threshold = 0.0`): `0.0465` (worse than baseline)
  - pooled all-steps voting: `0.0465` (worse than baseline)
  - TRM NLL/PoE rescoring (`v2_rescore_trm`): `0.0523` (no gain over baseline)
  - mdlARC rescoring (`v2_rescore_mdlarc`):
    - mdlARC NLL: `0.0116` (degrades),
    - mdlARC PoE: **`0.0640`** (best in this experiment family)

- **TRM-EXP-03 structured-prefix finetune**
  - promising internal training curves in some runs, but no robust evaluation gain in this compute budget,
  - currently documented as negative/inconclusive with explicit compute caveats.

## Notes on metrics

ARC numbers can differ depending on averaging definition and filtering.
See `docs/metrics.md` for the definitions used here.

## Sources used in this repository

- TRM upstream: [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT).
  This repo expects it under `third_party/TinyRecursiveModels/` (git submodule).
  Recommended pinned commit for v1 of this repo: `7de0d20c8f26df706e2c7b3a21ceaf0b3542c953`.
- Checkpoints: [`arcprize/trm_arc_prize_verification`](https://huggingface.co/arcprize/trm_arc_prize_verification).
- Official ARC-AGI-2 task repository (reference): [`arcprize/ARC-AGI-2`](https://github.com/arcprize/ARC-AGI-2).
- For the evaluations reported in this repository, task files are built from the upstream TRM `kaggle/combined/arc-agi` inputs.
- External rescoring source used in TRM-EXP-02: [`mvakde/mdlARC`](https://github.com/mvakde/mdlARC) (reference-only in this repo; no vendored code).
- PoE reference used for TRM-EXP-02 interpretation: [Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](https://arxiv.org/pdf/2505.07859).
- See also `THIRD_PARTY_NOTICES.md` for license/terms pointers.

## License

- Code/docs in this repo: MIT (see `LICENSE`).
- Third-party components: see `THIRD_PARTY_NOTICES.md`.
