#!/usr/bin/env python3
"""
TRM-EXP-02: Intelligent test-time voting (no training).

This script runs one inference pass over the augmented test set for multiple TRM outer steps,
then computes several selection strategies offline.

Outputs:
- voting_report.json (machine-readable summary)
- dataset_fingerprint.json (hashes for traceability)
- checkpoint_sha256.txt (if enabled)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PuzzleIdSeparator = "|||"
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0:
        return arr
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[int(tid)])


def arc_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    arr = np.array(grid)
    return arr.astype(np.uint8)


def grid_hash(grid: np.ndarray) -> str:
    assert grid.ndim == 2 and grid.dtype == np.uint8
    buffer = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_json_bytes(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _sigmoid_float(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def inverse_aug(name: str):
    """
    Inverse of TRM augmentation naming:
      <base>|||t{dihedral_id}|||{perm_digits}
    Returns (base_name, inverse_fn) where inverse_fn maps view-grid -> canonical-grid.
    """
    if PuzzleIdSeparator not in name:
        return name, lambda x: x
    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # drop leading 't'
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map_grid


def _arc_crop_fallback(tokens_flat_900: np.ndarray) -> np.ndarray:
    """Numpy-only fallback for cropping 900 TRM tokens into a grid (0..9)."""
    grid = np.asarray(tokens_flat_900).reshape(30, 30)
    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) or (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return (grid[: max_size[0], : max_size[1]] - 2).astype(np.uint8)


def _rank_candidates(stats_map: Dict[str, Tuple[int, float]], rank_by: str) -> List[str]:
    items: List[Tuple[str, int, float, float]] = []
    for h, (cnt, sumq) in stats_map.items():
        avgq = float(sumq) / max(int(cnt), 1)
        items.append((h, int(cnt), float(sumq), avgq))
    if rank_by == "count_avgq":
        items.sort(key=lambda t: (t[1], t[3]), reverse=True)
    elif rank_by == "sumq_count":
        items.sort(key=lambda t: (t[2], t[1], t[3]), reverse=True)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")
    return [h for (h, _cnt, _sumq, _avgq) in items]


def _compute_pass_metrics_from_votes(
    test_puzzles: Dict[str, Any],
    gt_outputs: List[Tuple[str, str, str]],
    votes: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]],
    pass_Ks: Sequence[int],
) -> Dict[str, float]:
    correct_task = [0.0 for _ in range(len(pass_Ks))]
    correct_outputs = [0 for _ in range(len(pass_Ks))]
    total_outputs = max(len(gt_outputs), 1)

    for task_name, puzzle in test_puzzles.items():
        num_test_correct = [0 for _ in range(len(pass_Ks))]
        for pair in puzzle.get("test", []):
            inp_h = grid_hash(arc_grid_to_np(pair["input"]))
            lab_h = grid_hash(arc_grid_to_np(pair["output"]))
            p_map = votes.get((task_name, inp_h))
            if not p_map:
                continue
            ranked = _rank_candidates(p_map, rank_by="count_avgq")
            for i, k in enumerate(pass_Ks):
                ok = lab_h in set(ranked[:k])
                num_test_correct[i] += int(ok)
                correct_outputs[i] += int(ok)
        denom = max(len(puzzle.get("test", [])), 1)
        for i in range(len(pass_Ks)):
            correct_task[i] += num_test_correct[i] / denom

    metrics = {f"ARC/pass@{k}": correct_task[i] / max(len(test_puzzles), 1) for i, k in enumerate(pass_Ks)}
    metrics.update({f"ARC/pass@{k}_per_output": correct_outputs[i] / total_outputs for i, k in enumerate(pass_Ks)})
    return metrics


def _pool_votes_across_steps(
    votes_by_step: Sequence[Dict[Tuple[str, str], Dict[str, Tuple[int, float]]]],
    steps: Sequence[int],
) -> Dict[Tuple[str, str], Dict[str, Tuple[int, float]]]:
    pooled: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = {}
    for s in steps:
        for key, p_map in votes_by_step[s].items():
            out_map = pooled.setdefault(key, {})
            for h, (cnt, sumq) in p_map.items():
                prev = out_map.get(h)
                if prev is None:
                    out_map[h] = (int(cnt), float(sumq))
                else:
                    out_map[h] = (int(prev[0] + int(cnt)), float(prev[1] + float(sumq)))
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", default="runs/trm_exp_02_voting")
    ap.add_argument("--batch_size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--forward_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--filter_official_eval", choices=["v1", "v2", "concept"], default="v2")
    ap.add_argument("--halt_logit_threshold", type=float, default=0.0)
    ap.add_argument("--no_sha256", action="store_true")
    ap.add_argument("--disable_compile", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trm_root = repo_root / "third_party" / "TinyRecursiveModels"
    if not trm_root.exists():
        raise SystemExit(f"Missing TRM submodule at: {trm_root} (clone with --recurse-submodules)")
    sys.path.insert(0, str(trm_root))

    # TRM imports
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # type: ignore
    from utils.functions import load_model_class  # type: ignore

    # Crop: prefer upstream if available, else fallback
    try:  # pragma: no cover
        from evaluators.arc import _crop as arc_crop  # type: ignore
    except Exception:  # pragma: no cover
        arc_crop = _arc_crop_fallback  # type: ignore

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    # Fingerprints (label-free; stored for traceability)
    dataset_json_path = Path(args.data_path) / "test" / "dataset.json"
    identifiers_json_path = Path(args.data_path) / "identifiers.json"
    test_puzzles_json_path = Path(args.data_path) / "test_puzzles.json"
    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    identifiers = json.loads(identifiers_json_path.read_text(encoding="utf-8"))
    test_puzzles = json.loads(test_puzzles_json_path.read_text(encoding="utf-8"))

    dataset_fingerprint = {
        "dataset_json_path": str(dataset_json_path),
        "identifiers_json_path": str(identifiers_json_path),
        "test_puzzles_json_path": str(test_puzzles_json_path),
        "dataset_json_sha256": _sha256_json_bytes(dataset_json),
        "identifiers_json_sha256": _sha256_json_bytes(identifiers),
        "test_puzzles_json_sha256": _sha256_json_bytes(test_puzzles),
        "metadata": dataset_json,
    }
    (out_dir / "dataset_fingerprint.json").write_text(json.dumps(dataset_fingerprint, indent=2), encoding="utf-8")

    ckpt_path = Path(args.checkpoint)
    if not args.no_sha256:
        (out_dir / "checkpoint_sha256.txt").write_text(_sha256_file(ckpt_path) + "\n", encoding="utf-8")

    # Filter tasks to official list (default v2)
    if args.filter_official_eval:
        official = _load_official_eval_task_ids(trm_root=trm_root, eval_version=str(args.filter_official_eval))
        if official is not None:
            test_puzzles = {k: v for k, v in test_puzzles.items() if k in official}

    # Build gt list for per-output averaging
    gt_outputs: List[Tuple[str, str, str]] = []
    for task_name, puzzle in test_puzzles.items():
        for pair in puzzle.get("test", []):
            inp_h = grid_hash(arc_grid_to_np(pair["input"]))
            lab_h = grid_hash(arc_grid_to_np(pair["output"]))
            gt_outputs.append((task_name, inp_h, lab_h))

    # Dataset
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=int(args.seed),
            dataset_paths=[str(args.data_path)],
            rank=0,
            num_replicas=1,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=int(args.batch_size),
        ),
        split="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    metadata = dataset.metadata

    # Model (same minimal defaults as eval_only.py)
    model_cfg = dict(
        batch_size=int(args.batch_size),
        vocab_size=int(metadata.vocab_size),
        seq_len=int(metadata.seq_len),
        num_puzzle_identifiers=int(metadata.num_puzzle_identifiers),
        causal=False,
        halt_exploration_prob=0.1,
        halt_max_steps=int(args.max_steps),
        H_cycles=3,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        puzzle_emb_ndim=512,
        pos_encodings="rope",
        forward_dtype=str(args.forward_dtype),
        mlp_t=False,
        puzzle_emb_len=16,
        no_ACT_continue=True,
    )
    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    loss_cls = load_model_class("losses@ACTLossHead")
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_cls(model, loss_type="stablemax_cross_entropy")

    sd = torch.load(str(ckpt_path), map_location="cuda")
    if isinstance(sd, dict):
        sd = {(k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, assign=True)
    model = model.cuda().eval()

    if (not args.disable_compile) and ("DISABLE_COMPILE" not in os.environ):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Vote stats: step -> (task, input_hash) -> pred_hash -> (count, sum_qsig)
    max_steps = int(args.max_steps)
    votes_by_step: List[Dict[Tuple[str, str], Dict[str, Tuple[int, float]]]] = [defaultdict(dict) for _ in range(max_steps)]
    votes_by_halt: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
    halt_hist: DefaultDict[int, int] = defaultdict(int)  # -1 means never

    # Aggregate per-step metrics (small; no per-task dumps)
    per_step = {str(s): {"tokenacc_sum": 0.0, "seqem_exact": 0, "n": 0} for s in range(max_steps)}

    def _tokenacc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target >= 2
        denom = mask.sum(dim=1).clamp_min(1)
        num = ((pred == target) & mask).sum(dim=1)
        return (num.to(torch.float32) / denom.to(torch.float32))

    # Inference
    t0 = time.time()
    processed_batches = 0
    processed_examples = 0

    with torch.no_grad():
        for _set_name, batch_cpu, _gb in loader:
            # Filter by selected tasks (base task id)
            if args.filter_official_eval:
                mask_list = []
                pids_cpu = batch_cpu["puzzle_identifiers"].numpy()
                for pid_int in pids_cpu:
                    name = identifiers[int(pid_int)]
                    base_name, _ = inverse_aug(name)
                    mask_list.append(base_name in test_puzzles)
                mask = torch.tensor(mask_list, dtype=torch.bool)
                if not mask.any():
                    continue
                batch_cpu = {k: v[mask] for k, v in batch_cpu.items()}

            processed_batches += 1
            processed_examples += int(batch_cpu["inputs"].shape[0])

            batch = {k: v.cuda(non_blocking=True) for k, v in batch_cpu.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            all_preds: List[Dict[str, torch.Tensor]] = []
            for _s in range(max_steps):
                carry, _loss, _metrics, preds, _all_finish = model(carry=carry, batch=batch, return_keys=["preds", "q_halt_logits"])
                all_preds.append({"preds": preds["preds"].clone(), "q_halt_logits": preds["q_halt_logits"].clone()})

            inputs_np = batch_cpu["inputs"].numpy()
            pids_np = batch_cpu["puzzle_identifiers"].numpy()
            labels_cpu = batch_cpu.get("labels")
            if labels_cpu is None:
                raise RuntimeError("Dataset batch missing labels; cannot compute TokenAcc/SeqEM.")
            labels = labels_cpu.cuda(non_blocking=True)
            B = int(inputs_np.shape[0])
            for b_idx in range(B):
                pid_int = int(pids_np[b_idx])
                name = identifiers[pid_int]
                base_name, inv_fn = inverse_aug(name)

                # Canonical input hash (for matching to ground truth)
                try:
                    inp_grid = inv_fn(arc_crop(inputs_np[b_idx]))
                    inp_hash = grid_hash(inp_grid.astype(np.uint8))
                except Exception:
                    continue

                key = (base_name, inp_hash)

                theoretical_halt_step = -1
                halt_pred_hash = None
                halt_sumq = 0.0

                for s in range(max_steps):
                    q_halt = float(all_preds[s]["q_halt_logits"][b_idx].item())
                    qsig = _sigmoid_float(q_halt)
                    pred_tokens = all_preds[s]["preds"][b_idx].detach().cpu().numpy()
                    try:
                        pred_grid = inv_fn(arc_crop(pred_tokens))
                        pred_hash = grid_hash(pred_grid.astype(np.uint8))
                    except Exception:
                        continue

                    step_map = votes_by_step[s].setdefault(key, {})
                    prev = step_map.get(pred_hash)
                    if prev is None:
                        step_map[pred_hash] = (1, qsig)
                    else:
                        step_map[pred_hash] = (int(prev[0] + 1), float(prev[1] + qsig))

                    if theoretical_halt_step == -1 and q_halt > float(args.halt_logit_threshold):
                        theoretical_halt_step = s
                        halt_pred_hash = pred_hash
                        halt_sumq = qsig

                halt_hist[theoretical_halt_step] += 1
                if halt_pred_hash is not None:
                    out_map = votes_by_halt.setdefault(key, {})
                    prev = out_map.get(halt_pred_hash)
                    if prev is None:
                        out_map[halt_pred_hash] = (1, halt_sumq)
                    else:
                        out_map[halt_pred_hash] = (int(prev[0] + 1), float(prev[1] + halt_sumq))

            # Per-step TokenAcc/SeqEM aggregated (vectorized; keep JSON small)
            for s in range(max_steps):
                pred_s = all_preds[s]["preds"]
                exact = (pred_s == labels).all(dim=1)
                tokenacc = _tokenacc(pred_s, labels)
                rec = per_step[str(s)]
                n = int(pred_s.shape[0])
                rec["tokenacc_sum"] += float(tokenacc.sum().detach().item())
                rec["seqem_exact"] += int(exact.sum().detach().item())
                rec["n"] += n

            del batch, batch_cpu, carry, all_preds
            if processed_batches % 100 == 0:
                torch.cuda.empty_cache()

    wall_infer = time.time() - t0

    # Offline strategy matrix
    pass_Ks = (1, 2, 5, 10, 100, 1000)
    strategies: Dict[str, Dict[str, float]] = {}

    baseline_votes = votes_by_step[max_steps - 1]
    strategies["baseline_last_step_count_avgq"] = _compute_pass_metrics_from_votes(test_puzzles, gt_outputs, baseline_votes, pass_Ks)

    # Per-step curve + best fixed step by pass@2_per_output (NOTE: uses eval labels; for clean protocol pick step on a dev set)
    step_curves = {"pass1_per_output": [], "pass2_per_output": []}
    best_step = 0
    best_score = -1.0
    per_step_metrics: List[Dict[str, float]] = []
    for s in range(max_steps):
        m = _compute_pass_metrics_from_votes(test_puzzles, gt_outputs, votes_by_step[s], pass_Ks)
        per_step_metrics.append(m)
        step_curves["pass1_per_output"].append(float(m.get("ARC/pass@1_per_output", 0.0)))
        step_curves["pass2_per_output"].append(float(m.get("ARC/pass@2_per_output", 0.0)))
        v = float(m.get("ARC/pass@2_per_output", -1.0))
        if v > best_score:
            best_score = v
            best_step = s

    strategies["best_fixed_step_count_avgq"] = dict(per_step_metrics[best_step])
    strategies["best_fixed_step_count_avgq"]["_best_step"] = float(best_step)

    pooled_all = _pool_votes_across_steps(votes_by_step, steps=list(range(max_steps)))
    strategies["pooled_all_steps_count_avgq"] = _compute_pass_metrics_from_votes(test_puzzles, gt_outputs, pooled_all, pass_Ks)
    strategies["simulated_halting_count_avgq"] = _compute_pass_metrics_from_votes(test_puzzles, gt_outputs, votes_by_halt, pass_Ks)

    # Candidate diversity diagnostics (mean unique candidates per output)
    unique_cands_mean_by_step: List[float] = []
    for s in range(max_steps):
        counts = [len(votes_by_step[s].get((t, inp_h), {})) for (t, inp_h, _lab_h) in gt_outputs]
        unique_cands_mean_by_step.append(float(np.mean(counts)) if counts else 0.0)

    # Pick best strategy by pass@2_per_output
    best_strategy = None
    best_pass2 = -1.0
    for name, m in strategies.items():
        v = float(m.get("ARC/pass@2_per_output", -1.0))
        if v > best_pass2:
            best_pass2 = v
            best_strategy = name

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint": str(ckpt_path),
            "data_path": str(args.data_path),
            "batch_size": int(args.batch_size),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "filter_official_eval": str(args.filter_official_eval),
            "halt_logit_threshold": float(args.halt_logit_threshold),
        },
        "timing": {
            "wall_inference_s": float(wall_infer),
            "wall_total_s": float(wall_infer),  # no rescoring stage in minimal runner
        },
        "summary": {
            "best_strategy_by_pass2_per_output": best_strategy,
            "best_pass2_per_output": float(best_pass2),
            "best_step_by_pass2_per_output": int(best_step),
            "num_tasks": int(len(test_puzzles)),
            "num_outputs": int(len(gt_outputs)),
            "processed_batches": int(processed_batches),
            "processed_examples": int(processed_examples),
        },
        "strategies": strategies,
        "curves": {"step_curves": step_curves, "best_step_by_pass2_per_output": int(best_step)},
        "diagnostics": {
            "unique_candidates_mean_by_step": unique_cands_mean_by_step,
            "halt_hist": {str(k): int(v) for k, v in sorted(halt_hist.items(), key=lambda kv: int(kv[0]))},
        },
        "per_step": {
            s: {
                "tokenacc_mean": (float(rec["tokenacc_sum"]) / max(int(rec["n"]), 1)),
                "seqem_mean": (float(rec["seqem_exact"]) / max(int(rec["n"]), 1)),
                "n": float(max(int(rec["n"]), 1)),
            }
            for s, rec in per_step.items()
        },
    }

    (out_dir / "voting_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_dir / 'voting_report.json'}")


def _load_official_eval_task_ids(trm_root: Path, eval_version: str) -> Optional[set]:
    combined = trm_root / "kaggle" / "combined"
    if eval_version == "v2":
        p = combined / "arc-agi_evaluation2_challenges.json"
    elif eval_version == "v1":
        p = combined / "arc-agi_evaluation_challenges.json"
    elif eval_version == "concept":
        p = combined / "arc-agi_concept_challenges.json"
    else:
        raise ValueError(f"Unknown eval_version: {eval_version}")
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return set(d.keys())


if __name__ == "__main__":
    main()

