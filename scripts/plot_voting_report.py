#!/usr/bin/env python3
"""
CPU-only plotting for TRM-EXP-02 voting reports.

Usage:
  python scripts/plot_voting_report.py --report runs/trm_exp_02_voting/voting_report.json --out_dir runs/trm_exp_02_voting/plots
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to voting_report.json")
    ap.add_argument("--out_dir", required=True, help="Directory to write PNGs")
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    plt = _maybe_import_matplotlib()

    def _save_both(name: str) -> None:
        png = os.path.join(args.out_dir, f"{name}.png")
        pdf = os.path.join(args.out_dir, f"{name}.pdf")
        plt.savefig(png, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")

    # Step curves
    step_curves = _get(rep, "curves.step_curves", {}) or {}
    p1 = step_curves.get("pass1_per_output", [])
    p2 = step_curves.get("pass2_per_output", [])
    best_step = int(_get(rep, "curves.best_step_by_pass2_per_output", 0) or 0)

    if p1 and p2:
        xs = np.arange(len(p2))
        plt.figure(figsize=(7, 4))
        plt.plot(xs, np.array(p2) * 100.0, label="pass@2_per_output")
        plt.plot(xs, np.array(p1) * 100.0, label="pass@1_per_output")
        plt.axvline(best_step, linestyle="--", linewidth=1)
        plt.xlabel("Outer step")
        plt.ylabel("Accuracy (%)")
        plt.title("TRM-EXP-02: pass@K_per_output vs outer step")
        plt.legend()
        _save_both("pass_per_output_vs_step")
        plt.close()

    # Strategy scoreboard (pass@2_per_output)
    strategies: Dict[str, Dict[str, Any]] = rep.get("strategies", {}) or {}
    if strategies:
        rows: List[Tuple[str, float, float]] = []
        for name, m in strategies.items():
            p2o = float(m.get("ARC/pass@2_per_output", 0.0) or 0.0)
            p1o = float(m.get("ARC/pass@1_per_output", 0.0) or 0.0)
            rows.append((name, p2o, p1o))
        rows.sort(key=lambda t: t[1], reverse=True)
        names = [r[0] for r in rows]
        vals = [r[1] * 100.0 for r in rows]

        plt.figure(figsize=(10, max(3.5, 0.4 * len(names))))
        y = np.arange(len(names))
        plt.barh(y, vals)
        plt.yticks(y, names)
        plt.gca().invert_yaxis()
        plt.xlabel("pass@2_per_output (%)")
        plt.title("TRM-EXP-02: strategy scoreboard")
        _save_both("strategy_scoreboard_pass2_per_output")
        plt.close()

    # Candidate diversity
    uniq = _get(rep, "diagnostics.unique_candidates_mean_by_step", []) or []
    if uniq:
        xs = np.arange(len(uniq))
        plt.figure(figsize=(7, 4))
        plt.plot(xs, np.array(uniq), marker="o", markersize=3, linewidth=1)
        plt.xlabel("Outer step")
        plt.ylabel("Mean unique candidates per output")
        plt.title("TRM-EXP-02: candidate diversity vs step")
        _save_both("unique_candidates_vs_step")
        plt.close()

    # Halt histogram
    halt_hist = _get(rep, "diagnostics.halt_hist", {}) or {}
    if halt_hist:
        ks = sorted(halt_hist.keys(), key=lambda x: int(x))
        vs = [int(halt_hist[k]) for k in ks]
        plt.figure(figsize=(9, 3.5))
        plt.bar([str(k) for k in ks], vs)
        plt.xlabel("Theoretical halt step (-1 means never)")
        plt.ylabel("Count")
        plt.title("TRM-EXP-02: theoretical halting distribution")
        plt.yscale("log")
        _save_both("halt_hist_logy")
        plt.close()

    print(f"Wrote plots to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

