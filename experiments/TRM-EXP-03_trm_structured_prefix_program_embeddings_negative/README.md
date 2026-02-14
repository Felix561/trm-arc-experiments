## TRM-EXP-03 — Structured prefix “program” embeddings (negative result)

### Goal

Test whether factoring the TRM prefix / puzzle embedding into semantically meaningful slots (UUID + augmentation factors + base “program” tokens)
can improve evaluation metrics via prefix-only finetuning (and variants).

### Outcome (current)

In our runs, structured-prefix training produced strong training fits but **did not reliably improve** evaluation pass@K.
This experiment is published primarily as a **negative/inconclusive result** to avoid duplicated effort.

### Status

This repository intentionally does **not** include a full training pipeline for this idea, because the initial public release
focuses on evaluation correctness (TRM-EXP-01) and test-time aggregation (TRM-EXP-02).

If you want to pursue this direction, start from upstream TRM training (`third_party/TinyRecursiveModels/pretrain.py`) and use the
dataset builder flags documented in the original local writeup.

