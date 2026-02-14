## TRM ARC evaluation & test-time voting experiments

This repository is a small, reproducible research artifact around **Tiny Recursive Models (TRM)** on ARC.
It focuses on **measurement correctness** and **test-time aggregation** (no training), plus one explicitly
reported **negative result** (structured prefix embeddings did not reliably improve evaluation metrics in our runs).

### What this repo contains

- **TRM-EXP-01** (`trm_reproduce_arcprize_verification`): reproduce published TRM checkpoint evaluation behavior on ARC splits.
- **TRM-EXP-02** (`trm_test_time_voting`): test-time voting/selection variants (step selection, pooling, PoE/NLL rescoring).
- **TRM-EXP-03** (`trm_structured_prefix_program_embeddings_negative`): negative/inconclusive result writeup for structured prefix “program” embeddings.

Upstream model code is referenced as a pinned dependency under `third_party/TinyRecursiveModels` (see below).

### Quickstart (Linux)

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/Felix561/trm-arc-experiments
cd public_trm_experiments
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

### Get checkpoints and build datasets

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

### Run experiments

- **TRM-EXP-01 (baseline eval)**:

```bash
python scripts/eval_only.py \
  --checkpoint assets/checkpoints/arc_v2_public/step_723914 \
  --data_path data/arc2concept-aug-1000 \
  --output_dir runs/trm_exp_01_arc2_eval \
  --filter_official_eval v2
```

- **TRM-EXP-02 (test-time voting matrix)**:

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

### Results snapshots (small, committed artifacts)

This repo keeps large artifacts (datasets/checkpoints/full runs) out of git. A small set of
result summaries live under `reports/`.
For full reproducibility, rerun experiments locally to regenerate JSON reports under `runs/`.

### Notes on metrics

ARC numbers can differ depending on averaging definition and filtering.
See `docs/metrics.md` for the definitions used here.

### Upstream dependencies

- TRM upstream: [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (MIT).
  This repo expects it under `third_party/TinyRecursiveModels/` (git submodule).
  Recommended pinned commit for v1 of this repo: `7de0d20c8f26df706e2c7b3a21ceaf0b3542c953`.
- Checkpoints: [`arcprize/trm_arc_prize_verification`](https://huggingface.co/arcprize/trm_arc_prize_verification).

### License

- Code/docs in this repo: MIT (see `LICENSE`).
- Third-party components: see `THIRD_PARTY_NOTICES.md`.

