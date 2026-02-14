#!/usr/bin/env python3
"""
Minimal environment check.

This script intentionally stays lightweight: it verifies that the upstream TRM submodule is present,
and that core Python dependencies can be imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    trm_root = root / "third_party" / "TinyRecursiveModels"
    print(f"[doctor] repo_root={root}")
    print(f"[doctor] python={sys.version}")

    missing = []
    if not trm_root.exists():
        missing.append(str(trm_root))
    if missing:
        print("[doctor] missing paths:")
        for p in missing:
            print(f"  - {p}")
        print("[doctor] hint: did you clone with --recurse-submodules (or run git submodule update --init)?")
        raise SystemExit(2)

    # Imports (keep minimal)
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import tqdm  # noqa: F401
    import huggingface_hub  # noqa: F401

    # Ensure TRM can be imported
    sys.path.insert(0, str(trm_root))
    import puzzle_dataset  # noqa: F401

    print("[doctor] OK")


if __name__ == "__main__":
    # Avoid surprising CUDA allocations
    os.environ.setdefault("DISABLE_COMPILE", "1")
    main()

