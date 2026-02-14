#!/usr/bin/env python3
"""
CPU-only plotting for TRM-EXP-01 eval reports.

Usage:
  python scripts/plot_eval_report.py --report runs/trm_exp_01_arc2_eval/eval_report.json --out_dir runs/trm_exp_01_arc2_eval/plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def _maybe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Paper-ish defaults (portable, no seaborn dependency)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
        }
    )
    return plt


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _save_both(plt, out_dir: str, name: str) -> None:
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to eval_report.json")
    ap.add_argument("--out_dir", required=True, help="Directory to write plots")
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    plt = _maybe_import_matplotlib()

    per_step: Dict[str, Dict[str, Any]] = rep.get("per_step", {}) or {}
    if per_step:
        steps = sorted(per_step.keys(), key=lambda x: int(x))
        xs = np.array([int(s) for s in steps], dtype=int)
        loss = np.array([float(per_step[s].get("loss_mean", 0.0) or 0.0) for s in steps], dtype=float)
        tok = np.array([float(per_step[s].get("tokenacc_mean", 0.0) or 0.0) for s in steps], dtype=float) * 100.0
        seq = np.array([float(per_step[s].get("seqem_mean", 0.0) or 0.0) for s in steps], dtype=float) * 100.0

        # Loss
        plt.figure(figsize=(7.0, 4.0))
        plt.plot(xs, loss, marker="o", markersize=4)
        plt.xlabel("Outer step")
        plt.ylabel("Loss (mean)")
        plt.title("TRM-EXP-01: loss vs outer step")
        _save_both(plt, args.out_dir, "loss_vs_step")
        plt.close()

        # TokenAcc + SeqEM
        plt.figure(figsize=(7.0, 4.0))
        plt.plot(xs, tok, marker="o", markersize=4, label="TokenAcc (%)")
        plt.plot(xs, seq, marker="o", markersize=4, label="SeqEM (%)")
        plt.xlabel("Outer step")
        plt.ylabel("Accuracy (%)")
        plt.title("TRM-EXP-01: token accuracy / exact match vs outer step")
        plt.legend()
        _save_both(plt, args.out_dir, "acc_vs_step")
        plt.close()

    # Pass@K bar chart (per-output is ARC Prize-style)
    metrics: Dict[str, Any] = rep.get("metrics", {}) or {}
    # Support both schemas:
    # - this repo: "ARC/pass@K_per_output"
    # - legacy: "pass@K_per_output"
    keys = ["ARC/pass@1_per_output", "ARC/pass@2_per_output", "ARC/pass@5_per_output", "ARC/pass@10_per_output"]
    vals: List[Tuple[str, float]] = []
    for k in keys:
        v = metrics.get(k)
        if v is None and k.startswith("ARC/"):
            v = metrics.get(k[len("ARC/") :])
        if v is None:
            continue
        label = k.replace("ARC/", "").replace("_per_output", "")
        vals.append((label, float(v) * 100.0))

    if vals:
        names = [a for (a, _v) in vals]
        ys = [v for (_a, v) in vals]
        plt.figure(figsize=(7.0, 3.6))
        plt.bar(names, ys)
        plt.ylabel("Accuracy (%)")
        plt.title("TRM-EXP-01: ARC Prize-style pass@K per output")
        _save_both(plt, args.out_dir, "passk_per_output")
        plt.close()

    print(f"Wrote plots to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

