# Third-party notices

This repository contains original code and documentation under the MIT License (see `LICENSE`).

It also depends on third-party projects and assets. Each third-party component remains under its
own license and terms. This file summarizes the major components.

## TinyRecursiveModels (TRM upstream)

- **Project**: TinyRecursiveModels
- **Upstream**: `https://github.com/SamsungSAILMontreal/TinyRecursiveModels`
- **Included as**: git submodule at `third_party/TinyRecursiveModels`
- **License**: MIT (see `third_party/TinyRecursiveModels/LICENSE`)

## ARC Prize verification checkpoints (TRM)

- **Source**: ARC Prize Foundation
- **Link**: `https://huggingface.co/arcprize/trm_arc_prize_verification`
- **How used**: downloaded on-demand by `scripts/fetch_trm_checkpoints.py`
- **License / terms**: see the model card and repository terms on Hugging Face

## ARC datasets

This repository does **not** commit large dataset files by default. Datasets are fetched and/or
built locally. You are responsible for complying with the dataset licenses/terms for any sources
you download.

