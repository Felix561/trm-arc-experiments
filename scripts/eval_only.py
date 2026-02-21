#!/usr/bin/env python3
"""
TRM-EXP-01 baseline evaluator (no training).

This is a minimal, self-contained evaluation driver that:
- loads a TRM checkpoint (state_dict)
- runs recursive refinement for a fixed number of outer steps
- aggregates predictions using the upstream ARC evaluator
- writes a machine-readable JSON result artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib import request

import numpy as np
import torch
from torch.utils.data import DataLoader


def arc_grid_to_np(grid):
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


def _download_text_with_cache(url: str, cache_path: Path, force_refresh: bool) -> Tuple[Path, bool]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_refresh:
        return cache_path, False

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with request.urlopen(url, timeout=60) as resp:
            body = resp.read()
        tmp_path.write_bytes(body)
        tmp_path.replace(cache_path)
        return cache_path, True
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        if cache_path.exists():
            print(f"[filter] download failed, using cached file at: {cache_path}")
            return cache_path, False
        raise


def _load_task_ids_from_json(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data)}")
    return set(str(k) for k in data.keys())


def _load_task_ids_from_text(path: Path) -> Set[str]:
    out: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and (not line.startswith("#")):
                out.add(line)
    return out


def _load_official_eval_task_ids(
    trm_root: Path,
    eval_version: str,
    official_v2_source: str,
    arc_agi2_eval_list: Optional[Path],
    arc_agi2_ref: str,
    arc_agi2_cache_dir: Path,
    refresh_arc_agi2_eval: bool,
) -> Tuple[Optional[Set[str]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "eval_version": str(eval_version),
        "source": None,
        "task_list_path": None,
        "task_list_url": None,
        "downloaded_now": False,
    }

    combined = trm_root / "kaggle" / "combined"
    if eval_version == "v2" and official_v2_source == "arc_agi2_github":
        meta["source"] = "arc_agi2_github"
        if arc_agi2_eval_list is not None:
            p = arc_agi2_eval_list
            meta["task_list_path"] = str(p)
            if not p.exists():
                meta["error"] = f"File does not exist: {p}"
                return None, meta
            return _load_task_ids_from_text(p), meta

        safe_ref = arc_agi2_ref.replace("/", "__")
        p = arc_agi2_cache_dir / f"evaluation_{safe_ref}.txt"
        url = f"https://raw.githubusercontent.com/arcprize/ARC-AGI-2/{arc_agi2_ref}/data/evaluation.txt"
        meta["task_list_url"] = url
        meta["task_list_path"] = str(p)
        try:
            p, downloaded_now = _download_text_with_cache(
                url=url,
                cache_path=p,
                force_refresh=bool(refresh_arc_agi2_eval),
            )
            meta["task_list_path"] = str(p)
            meta["downloaded_now"] = bool(downloaded_now)
            return _load_task_ids_from_text(p), meta
        except Exception as e:
            meta["error"] = str(e)
            return None, meta

    meta["source"] = "kaggle_combined"
    if eval_version == "v2":
        p = combined / "arc-agi_evaluation2_challenges.json"
    elif eval_version == "v1":
        p = combined / "arc-agi_evaluation_challenges.json"
    elif eval_version == "concept":
        p = combined / "arc-agi_concept_challenges.json"
    else:
        raise ValueError(f"Unknown eval_version: {eval_version}")
    meta["task_list_path"] = str(p)
    if not p.exists():
        meta["error"] = f"File does not exist: {p}"
        return None, meta
    return _load_task_ids_from_json(p), meta


def _strip_orig_mod_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            out[k[len("_orig_mod.") :]] = v
        else:
            out[k] = v
    return out


def _checkpoint_num_puzzle_identifiers(sd: Dict[str, Any]) -> Optional[int]:
    key = "model.inner.puzzle_emb.weights"
    w = sd.get(key)
    if w is None or not hasattr(w, "shape") or len(w.shape) < 1:
        return None
    return int(w.shape[0])


def _compute_arc_metrics_and_submission_local(evaluator, save_path: Optional[str]) -> Dict[str, float]:
    """
    Single-process equivalent of upstream ARC.result().
    Upstream implementation always uses torch.distributed.gather_object, which fails
    when no process group is initialized (common in local eval runs).
    """
    submission = {}
    pass_ks = list(getattr(evaluator, "pass_Ks", (1, 2, 5, 10, 100, 1000)))
    submission_k = int(getattr(evaluator, "submission_K", 2))
    test_puzzles = getattr(evaluator, "test_puzzles")
    local_hmap = getattr(evaluator, "_local_hmap")
    local_preds = getattr(evaluator, "_local_preds")

    correct = [0.0 for _ in range(len(pass_ks))]

    for task_name, puzzle in test_puzzles.items():
        submission[task_name] = []
        num_test_correct = [0 for _ in range(len(pass_ks))]

        for pair in puzzle.get("test", []):
            input_hash = grid_hash(arc_grid_to_np(pair["input"]))
            label_hash = grid_hash(arc_grid_to_np(pair["output"]))

            p_map = {}
            for h, q in local_preds.get(task_name, {}).get(input_hash, []):
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += float(q)

            if not p_map:
                print(f"Puzzle {task_name} has no predictions.")
                continue

            # Convert sum_q to avg_q and sort by (count desc, avg_q desc).
            ranked = []
            for h, (cnt, sumq) in p_map.items():
                ranked.append((h, int(cnt), float(sumq) / max(int(cnt), 1)))
            ranked.sort(key=lambda t: (t[1], t[2]), reverse=True)

            for i, k in enumerate(pass_ks):
                topk = {h for (h, _cnt, _avgq) in ranked[: int(k)]}
                num_test_correct[i] += int(label_hash in topk)

            pred_grids = []
            for h, _cnt, _avgq in ranked[:submission_k]:
                if h in local_hmap:
                    pred_grids.append(local_hmap[h])

            while len(pred_grids) < submission_k and len(pred_grids) > 0:
                pred_grids.append(pred_grids[0])

            if len(pred_grids) == 0:
                continue

            submission[task_name].append(
                {f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)}
            )

        tests = max(len(puzzle.get("test", [])), 1)
        for i in range(len(pass_ks)):
            correct[i] += num_test_correct[i] / tests

    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        (Path(save_path) / "submission.json").write_text(json.dumps(submission), encoding="utf-8")

    denom = max(len(test_puzzles), 1)
    return {f"ARC/pass@{k}": correct[i] / denom for i, k in enumerate(pass_ks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to TRM checkpoint file (state_dict)")
    ap.add_argument("--data_path", required=True, help="Path to built TRM dataset directory (contains test/)")
    ap.add_argument("--output_dir", default="runs/trm_exp_01", help="Output directory for JSON + submission")
    ap.add_argument("--batch_size", type=int, default=768, help="Global batch size (TRM-specific)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=16, help="Outer refinement steps")
    ap.add_argument("--max_batches", type=int, default=0, help="Process at most N batches (0 = all)")
    ap.add_argument("--forward_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--filter_official_eval", choices=["v1", "v2", "concept"], default=None)
    ap.add_argument(
        "--official_v2_source",
        choices=["kaggle_combined", "arc_agi2_github"],
        default="kaggle_combined",
        help="Task ID source for --filter_official_eval v2 (default: kaggle_combined)",
    )
    ap.add_argument(
        "--arc_agi2_eval_list",
        default=None,
        help="Optional local path to ARC-AGI-2 data/evaluation.txt (only for --official_v2_source arc_agi2_github)",
    )
    ap.add_argument(
        "--arc_agi2_ref",
        default="main",
        help="Git ref for ARC-AGI-2 raw download (only for --official_v2_source arc_agi2_github)",
    )
    ap.add_argument(
        "--arc_agi2_cache_dir",
        default=".cache/arc-agi-2",
        help="Cache dir for ARC-AGI-2 evaluation list download (repo-relative if not absolute)",
    )
    ap.add_argument(
        "--refresh_arc_agi2_eval",
        action="store_true",
        help="Force refresh of downloaded ARC-AGI-2 evaluation list",
    )
    ap.add_argument("--disable_compile", action="store_true", help="Disable torch.compile")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for model inference")
    ap.add_argument("--no_sha256", action="store_true", help="Skip checkpoint sha256 computation")
    ap.add_argument("--no_submission", action="store_true", help="Do not write submission.json (faster, smaller)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trm_root = repo_root / "third_party" / "TinyRecursiveModels"
    if not trm_root.exists():
        raise SystemExit(f"Missing TRM submodule at: {trm_root} (clone with --recurse-submodules)")
    sys.path.insert(0, str(trm_root))

    arc_agi2_cache_dir = Path(args.arc_agi2_cache_dir).expanduser()
    if not arc_agi2_cache_dir.is_absolute():
        arc_agi2_cache_dir = repo_root / arc_agi2_cache_dir

    arc_agi2_eval_list: Optional[Path] = None
    if args.arc_agi2_eval_list:
        arc_agi2_eval_list = Path(str(args.arc_agi2_eval_list)).expanduser()
        if not arc_agi2_eval_list.is_absolute():
            arc_agi2_eval_list = repo_root / arc_agi2_eval_list

    # Imports from TRM
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # type: ignore
    from utils.functions import load_model_class  # type: ignore
    from evaluators.arc import ARC as UpstreamARC  # type: ignore

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device=cuda requested but CUDA is not available. Use --device=cpu on this machine.")

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

    # Load checkpoint early so we can adapt embedding table size if needed.
    ckpt_path = Path(args.checkpoint)
    sd = torch.load(str(ckpt_path), map_location=device)
    if isinstance(sd, dict):
        sd = _strip_orig_mod_prefix(sd)

    num_pids = int(metadata.num_puzzle_identifiers)
    ckpt_num_pids = _checkpoint_num_puzzle_identifiers(sd) if isinstance(sd, dict) else None
    if ckpt_num_pids is not None and ckpt_num_pids != num_pids:
        print(
            f"[compat] dataset num_puzzle_identifiers={num_pids}, "
            f"checkpoint num_puzzle_identifiers={ckpt_num_pids}; using checkpoint size for model init"
        )
        num_pids = int(ckpt_num_pids)

    # Model (defaults match TRM-att common settings; checkpoint must be compatible with dataset meta)
    model_cfg = dict(
        batch_size=int(args.batch_size),
        vocab_size=int(metadata.vocab_size),
        seq_len=int(metadata.seq_len),
        num_puzzle_identifiers=num_pids,
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
    model = model_cls(model_cfg)
    model = loss_cls(model, loss_type="stablemax_cross_entropy")

    # Load checkpoint weights
    model.load_state_dict(sd, assign=True)
    model = model.to(device).eval()

    if (not args.disable_compile) and ("DISABLE_COMPILE" not in os.environ):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile failed; continuing without compile. error={e}")

    # Evaluator
    evaluator = UpstreamARC(
        data_path=str(args.data_path),
        eval_metadata=metadata,
        submission_K=2,
        pass_Ks=(1, 2, 5, 10, 100, 1000),
        aggregated_voting=True,
    )

    # Optional official filtering
    filter_details: Dict[str, Any] = {}
    if args.filter_official_eval is not None:
        official, filter_details = _load_official_eval_task_ids(
            trm_root=trm_root,
            eval_version=str(args.filter_official_eval),
            official_v2_source=str(args.official_v2_source),
            arc_agi2_eval_list=arc_agi2_eval_list,
            arc_agi2_ref=str(args.arc_agi2_ref),
            arc_agi2_cache_dir=arc_agi2_cache_dir,
            refresh_arc_agi2_eval=bool(args.refresh_arc_agi2_eval),
        )
        if official is not None:
            dataset_task_ids = set(str(k) for k in evaluator.test_puzzles.keys())
            kept_ids = dataset_task_ids & official
            dropped_ids = dataset_task_ids - official
            missing_from_dataset = official - dataset_task_ids
            evaluator.test_puzzles = {k: v for k, v in evaluator.test_puzzles.items() if k in official}

            filter_details.update(
                {
                    "official_task_count": int(len(official)),
                    "dataset_task_count_before_filter": int(len(dataset_task_ids)),
                    "kept_task_count": int(len(kept_ids)),
                    "dropped_task_count": int(len(dropped_ids)),
                    "missing_official_ids_in_dataset_count": int(len(missing_from_dataset)),
                    "dropped_task_sample": sorted(list(dropped_ids))[:5],
                    "missing_official_id_sample": sorted(list(missing_from_dataset))[:5],
                }
            )
            print(
                "[filter] "
                f"source={filter_details.get('source')} "
                f"official={len(official)} "
                f"dataset_before={len(dataset_task_ids)} "
                f"kept={len(kept_ids)} "
                f"dropped={len(dropped_ids)} "
                f"missing_official_in_dataset={len(missing_from_dataset)}"
            )
            if filter_details.get("downloaded_now"):
                print(f"[filter] downloaded official v2 list to {filter_details.get('task_list_path')}")
        else:
            err = filter_details.get("error")
            if err:
                print(f"[filter] official list unavailable ({err}); continuing without filtering")
            else:
                print("[filter] official list unavailable; continuing without filtering")

    evaluator.begin_eval()

    # Inference loop
    t0 = time.time()
    num_batches = 0
    num_examples = 0
    return_keys = {"preds", "q_halt_logits"}

    # Aggregate per-step metrics (small + reproducible)
    max_steps = int(args.max_steps)
    per_step = {
        str(s): {"loss_sum": 0.0, "tokenacc_sum": 0.0, "seqem_exact": 0, "n": 0}
        for s in range(max_steps)
    }

    def _tokenacc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # token accuracy over ARC pixels only (labels >= 2)
        mask = target >= 2
        denom = mask.sum(dim=1).clamp_min(1)
        num = ((pred == target) & mask).sum(dim=1)
        return (num.to(torch.float32) / denom.to(torch.float32))

    with torch.no_grad():
        for _set_name, batch_cpu, _gb in loader:
            num_batches += 1
            num_examples += int(batch_cpu["inputs"].shape[0])
            batch = {k: v.to(device, non_blocking=(device == "cuda")) for k, v in batch_cpu.items()}
            carry = model.initial_carry(batch)
            preds_last = None
            labels = batch.get("labels")
            if labels is None:
                raise RuntimeError("Dataset batch is missing 'labels'; cannot compute TokenAcc/SeqEM.")

            for s in range(max_steps):
                carry, loss, _metrics, preds, _all_finish = model(carry=carry, batch=batch, return_keys=return_keys)
                preds_last = preds

                pred_tokens = preds["preds"]
                # SeqEM: exact match on full token sequence
                exact = (pred_tokens == labels).all(dim=1)
                tokenacc = _tokenacc(pred_tokens, labels)

                rec = per_step[str(s)]
                n = int(pred_tokens.shape[0])
                rec["loss_sum"] += float(loss.detach().item()) * n
                rec["tokenacc_sum"] += float(tokenacc.sum().detach().item())
                rec["seqem_exact"] += int(exact.sum().detach().item())
                rec["n"] += n

            assert preds_last is not None
            evaluator.update_batch(batch, preds_last)
            del batch, batch_cpu, carry, preds_last, labels
            if device == "cuda" and num_batches % 100 == 0:
                torch.cuda.empty_cache()
            if int(args.max_batches) > 0 and num_batches >= int(args.max_batches):
                print(f"[limit] stopping early at max_batches={args.max_batches}")
                break

    wall = time.time() - t0

    # Results
    submission_dir = out_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    metrics = _compute_arc_metrics_and_submission_local(
        evaluator=evaluator,
        save_path=None if args.no_submission else str(submission_dir),
    )

    # Add ARC Prize-style per-output metrics (2 attempts per output => pass@2_per_output).
    # Upstream TRM evaluator only returns per-task averages; compute per-output averages here.
    def _compute_per_output_metrics() -> Dict[str, float]:
        pass_Ks = list(getattr(evaluator, "pass_Ks", (1, 2, 5, 10, 100, 1000)))
        total_outputs = 0
        correct_outputs = [0 for _ in pass_Ks]

        # evaluator.test_puzzles: task -> puzzle dict with "test" pairs including ground-truth outputs
        for task_name, puzzle in evaluator.test_puzzles.items():
            for pair in puzzle.get("test", []):
                total_outputs += 1
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                # p_map: pred_hash -> [count, sum_q]
                p_map: Dict[str, list] = {}
                for h, q in evaluator._local_preds.get(task_name, {}).get(input_hash, []):  # type: ignore[attr-defined]
                    p_map.setdefault(h, [0, 0.0])
                    p_map[h][0] += 1
                    p_map[h][1] += float(q)
                if not p_map:
                    continue

                # avg q for tie-break; sort by (count desc, avg_q desc)
                items = []
                for h, (cnt, sumq) in p_map.items():
                    avgq = float(sumq) / max(int(cnt), 1)
                    items.append((h, int(cnt), avgq))
                items.sort(key=lambda t: (t[1], t[2]), reverse=True)
                ranked = [h for (h, _cnt, _avgq) in items]

                for i, k in enumerate(pass_Ks):
                    correct_outputs[i] += int(label_hash in set(ranked[: int(k)]))

        denom = max(int(total_outputs), 1)
        return {f"ARC/pass@{k}_per_output": float(correct_outputs[i]) / denom for i, k in enumerate(pass_Ks)}

    try:
        metrics.update(_compute_per_output_metrics())
    except Exception as e:
        # Keep the run usable even if this computation fails (should be rare).
        metrics["ARC/per_output_metrics_error"] = str(e)

    # Reduce per-step stats to means (keep JSON small)
    per_step_means: Dict[str, Dict[str, float]] = {}
    for s, rec in per_step.items():
        n = max(int(rec["n"]), 1)
        per_step_means[s] = {
            "loss_mean": float(rec["loss_sum"]) / n,
            "tokenacc_mean": float(rec["tokenacc_sum"]) / n,
            "seqem_mean": float(rec["seqem_exact"]) / n,
            "n": float(n),
        }

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": None if bool(args.no_sha256) else _sha256_file(ckpt_path),
            "data_path": str(args.data_path),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "device": device,
            "filter_official_eval": args.filter_official_eval,
            "official_v2_source": str(args.official_v2_source),
            "official_eval_details": filter_details or None,
            "num_batches": int(num_batches),
            "num_examples": int(num_examples),
            "wall_time_s": float(wall),
        },
        "per_step": per_step_means,
        "metrics": metrics,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
