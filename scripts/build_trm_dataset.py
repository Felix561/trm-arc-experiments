#!/usr/bin/env python3
"""
Thin wrapper around TRM's ARC dataset builder.

We keep this wrapper small so the public repo stays minimal and doesn't fork upstream logic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file_prefix", required=True, help="Prefix path like .../arc-agi (without _training*.json suffix)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--subsets", nargs="+", required=True)
    ap.add_argument("--test_set_name", required=True)
    ap.add_argument("--num_aug", type=int, default=1000)
    ap.add_argument("--no_use_base_ids", action="store_true")
    ap.add_argument("--structured_prefix_fields", action="store_true")
    ap.add_argument("--augment_retries_factor", type=int, default=None)
    ap.add_argument("--identifiers_json", default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trm_root = repo_root / "third_party" / "TinyRecursiveModels"
    if not trm_root.exists():
        raise SystemExit(f"Missing TRM submodule at: {trm_root} (clone with --recurse-submodules)")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "dataset.build_arc_dataset",
        "--input-file-prefix",
        str(args.input_file_prefix),
        "--output-dir",
        str(args.output_dir),
        "--subsets",
        *list(args.subsets),
        "--test-set-name",
        str(args.test_set_name),
        "--num-aug",
        str(int(args.num_aug)),
    ]
    if args.no_use_base_ids:
        cmd.append("--no-use-base-ids")
    if args.structured_prefix_fields:
        cmd.append("--structured-prefix-fields")
    if args.augment_retries_factor is not None:
        cmd.extend(["--augment-retries-factor", str(int(args.augment_retries_factor))])
    if args.identifiers_json:
        cmd.extend(["--identifiers-json", str(args.identifiers_json)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(trm_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print("[build] cwd:", repo_root)
    print("[build] PYTHONPATH includes:", trm_root)
    print("[build] cmd:\n ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()

