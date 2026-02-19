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

## External scorer/reference used in TRM-EXP-02

- **Project**: mdlARC
- **Upstream**: `https://github.com/mvakde/mdlARC`
- **How used**: referenced as an external rescoring backend for reported TRM candidate rescoring results.
- **Included in this repo**: no (reference only; no vendored mdlARC source code).

## Literature reference used in experiment interpretation

- **Paper**: Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective
- **Link**: `https://arxiv.org/pdf/2505.07859`
- **How used**: conceptual reference for PoE-style multi-view rescoring discussion.
