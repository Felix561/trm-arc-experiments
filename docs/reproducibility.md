## Reproducibility

### Environment

This repo is documented for **Linux**. GPU evaluation requires a CUDA-capable setup.

The recommended workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch matching your CUDA stack separately (see `requirements.txt` header).

### Determinism / seeds

The experiment scripts set Python, NumPy and PyTorch RNG seeds when a `--seed` argument is available.

Note: GPU kernels and compilation can introduce small non-determinism. When comparing methods, prefer:

- fixing seeds
- running the same dataset build and checkpoint
- reporting deltas with confidence intervals when possible (bootstrap from per-output/per-task records)

### Artifacts

Large artifacts are not committed:

- datasets (built under `data/`)
- checkpoints (downloaded under `assets/checkpoints/`)
- full run outputs (under `runs/`)

Small, human-auditable result snapshots are committed under `reports/`.

