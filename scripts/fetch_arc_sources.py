#!/usr/bin/env python3
"""
Fetch ARC JSON sources (small) via upstream TRM submodule.

Policy:
- Keep the git repo small (do not commit datasets).
- Materialize sources under `assets/data_sources/` if you want a stable local copy.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="assets/data_sources/kaggle_combined", help="Where to copy JSON sources")
    ap.add_argument("--copy", action="store_true", help="Copy JSONs from TRM submodule into out_dir")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trm_combined = repo_root / "third_party" / "TinyRecursiveModels" / "kaggle" / "combined"
    if not trm_combined.exists():
        raise SystemExit(
            f"Missing {trm_combined}. Did you clone with --recurse-submodules "
            "(or run git submodule update --init --recursive)?"
        )

    jsons = sorted(trm_combined.glob("*.json"))
    print(f"[arc] found {len(jsons)} json files in: {trm_combined}")

    if args.copy:
        out_dir = (repo_root / args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in jsons:
            shutil.copy2(p, out_dir / p.name)
        print(f"[arc] copied to: {out_dir}")
    else:
        print("[arc] nothing copied (pass --copy to materialize into assets/)")


if __name__ == "__main__":
    main()

