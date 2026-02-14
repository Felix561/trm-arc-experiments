#!/usr/bin/env python3
"""
Fetch ARC Prize TRM verification checkpoints from Hugging Face.

Default source:
  https://huggingface.co/arcprize/trm_arc_prize_verification

This script keeps the git repo small by downloading weights into `assets/checkpoints/`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional

from huggingface_hub import snapshot_download


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_sha256_manifest(files: Iterable[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for fp in sorted(files):
        if fp.is_file():
            lines.append(f"{_sha256_file(fp)}  {fp.as_posix()}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="arcprize/trm_arc_prize_verification")
    ap.add_argument("--out_dir", default="assets/checkpoints", help="Download destination directory")
    ap.add_argument(
        "--subdirs",
        nargs="*",
        default=["arc_v1_public", "arc_v2_public"],
        help="Subdirectories to download from the HF repo (default: arc_v1_public arc_v2_public)",
    )
    ap.add_argument("--revision", default=None, help="Optional git revision/tag in the HF repo")
    ap.add_argument("--no_sha256", action="store_true", help="Skip sha256 manifest (faster)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns: Optional[List[str]] = None
    if args.subdirs:
        allow_patterns = []
        for sd in args.subdirs:
            sd = sd.strip("/").strip()
            allow_patterns.append(f"{sd}/**")
        # Also grab top-level metadata files if present (model card)
        allow_patterns.extend(["README.*", "LICENSE*", "*.md"])

    print(f"[fetch] repo_id={args.repo_id}")
    print(f"[fetch] out_dir={out_dir}")
    if allow_patterns:
        print(f"[fetch] allow_patterns={allow_patterns}")

    local_dir = snapshot_download(
        repo_id=str(args.repo_id),
        revision=args.revision,
        allow_patterns=allow_patterns,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[fetch] done: {local_dir}")

    if not args.no_sha256:
        files = [p for p in out_dir.rglob("*") if p.is_file()]
        _write_sha256_manifest(files, out_dir / "SHA256SUMS.txt")
        print(f"[fetch] wrote: {out_dir / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    # keep deterministic output paths
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()

