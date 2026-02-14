#!/usr/bin/env python3
"""
TRM-EXP-02 (advanced): Intelligent voting at test time (no training).

This is the "full variants" runner ported from the private research code:
- One heavy inference pass over augmented eval (multiple outer steps)
- Offline strategy matrix
- Optional teacher-forced NLL / PoE rescoring (budgeted)
- Professional, machine-readable artifacts

IMPORTANT:
- This script is GPU-first.
- JSON outputs are kept manageable by default (no giant per-task dumps).
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional dependency: allow running in minimal envs.
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore


# -----------------------------------------------------------------------------
# Minimal ARC augmentation + hashing helpers (copied from TRM dataset builder)
# -----------------------------------------------------------------------------

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
    assert arr.ndim == 2
    return arr.astype(np.uint8)


def grid_hash(grid: np.ndarray) -> str:
    assert grid.ndim == 2 and grid.dtype == np.uint8
    buffer = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def inverse_aug(name: str):
    if PuzzleIdSeparator not in name:
        return name, lambda x: x
    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # remove leading 't'
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map_grid


def _sigmoid_float(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_json_bytes(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _load_official_eval_task_ids(trm_root: Path, eval_version: str) -> Optional[set]:
    combined_dir = trm_root / "kaggle" / "combined"
    if eval_version == "v2":
        path = combined_dir / "arc-agi_evaluation2_challenges.json"
    elif eval_version == "v1":
        path = combined_dir / "arc-agi_evaluation_challenges.json"
    elif eval_version == "concept":
        path = combined_dir / "arc-agi_concept_challenges.json"
    else:
        raise ValueError(f"Unknown eval_version: {eval_version}")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


def _try_load_checkpoint_config(checkpoint_path: str) -> Optional[dict]:
    if OmegaConf is None:
        return None
    ckpt_dir = os.path.dirname(checkpoint_path)
    cfg_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except Exception:
        return None


def _arch_from_checkpoint_config(cfg: Optional[dict]) -> Optional[dict]:
    if not cfg:
        return None
    arch = cfg.get("arch")
    if not isinstance(arch, dict):
        return None
    return arch


def _arc_crop_fallback(tokens_flat_900: np.ndarray) -> np.ndarray:
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
    submission_K: int,
    hmap: Dict[str, np.ndarray],
    save_submission_path: Optional[str],
    rank_by: str,
) -> Dict[str, float]:
    correct_task = [0.0 for _ in range(len(pass_Ks))]
    correct_outputs = [0 for _ in range(len(pass_Ks))]
    total_outputs = max(len(gt_outputs), 1)
    submission: Dict[str, List[Dict[str, Any]]] = {}

    for task_name, puzzle in test_puzzles.items():
        submission[task_name] = []
        num_test_correct = [0 for _ in range(len(pass_Ks))]
        for pair in puzzle.get("test", []):
            inp_h = grid_hash(arc_grid_to_np(pair["input"]))
            lab_h = grid_hash(arc_grid_to_np(pair["output"]))
            p_map = votes.get((task_name, inp_h))
            if not p_map:
                submission[task_name].append({f"attempt_{i+1}": [[0]] for i in range(submission_K)})
                continue
            ranked = _rank_candidates(p_map, rank_by=rank_by)
            for i, k in enumerate(pass_Ks):
                ok = lab_h in set(ranked[:k])
                num_test_correct[i] += int(ok)
                correct_outputs[i] += int(ok)

            pred_grids: List[np.ndarray] = []
            for h in ranked[:submission_K]:
                g = hmap.get(h)
                if g is not None:
                    pred_grids.append(g)
            while len(pred_grids) < submission_K:
                pred_grids.append(pred_grids[0] if pred_grids else np.zeros((1, 1), dtype=np.uint8))
            submission[task_name].append({f"attempt_{i+1}": g.tolist() for i, g in enumerate(pred_grids)})

        denom = max(len(puzzle.get("test", [])), 1)
        for i in range(len(pass_Ks)):
            correct_task[i] += num_test_correct[i] / denom

    if save_submission_path is not None:
        os.makedirs(save_submission_path, exist_ok=True)
        with open(os.path.join(save_submission_path, "submission.json"), "w", encoding="utf-8") as f:
            json.dump(submission, f)

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


@dataclass
class ViewRecord:
    pid_int: int
    name: str
    inputs_tokens: np.ndarray


def _parse_view_aug(name: str) -> Tuple[str, int, np.ndarray]:
    if PuzzleIdSeparator not in name:
        return name, 0, np.arange(10, dtype=np.uint8)
    parts = name.split(PuzzleIdSeparator)
    base = parts[0]
    if len(parts) < 3:
        return base, 0, np.arange(10, dtype=np.uint8)
    t_part = parts[-2]
    perm_part = parts[-1]
    dihedral_id = int(t_part[1:]) if t_part.startswith("t") else 0
    dihedral_id = int(np.clip(dihedral_id, 0, 7))
    mapping = np.arange(10, dtype=np.uint8)
    try:
        digits = [int(ch) for ch in perm_part]
        if len(digits) >= 10:
            mapping = np.array(digits[:10], dtype=np.uint8)
    except Exception:
        pass
    return base, dihedral_id, mapping


def _apply_view_aug_to_grid(grid_0_9: np.ndarray, dihedral_id: int, mapping_0_9: np.ndarray) -> np.ndarray:
    g = mapping_0_9[grid_0_9]
    g = dihedral_transform(g, dihedral_id)
    return g.astype(np.uint8)


def _grid_to_seq_tokens_no_translation(grid_0_9: np.ndarray) -> np.ndarray:
    nrow, ncol = grid_0_9.shape
    arr = np.zeros((30, 30), dtype=np.int32)
    arr[:nrow, :ncol] = grid_0_9.astype(np.int32) + 2
    eos_row, eos_col = nrow, ncol
    if eos_row < 30:
        arr[eos_row, :eos_col] = 1
    if eos_col < 30:
        arr[:eos_row, eos_col] = 1
    return arr.reshape(-1)


def _score_candidates_nll_for_view(
    model,
    IGNORE_LABEL_ID: int,
    stablemax_cross_entropy,
    softmax_cross_entropy,
    inputs_tokens_1x900: np.ndarray,
    pid_int: int,
    cand_label_seqs_mx900: np.ndarray,
    score_steps: int,
    nll_type: str,
    batch_size: int,
) -> np.ndarray:
    assert cand_label_seqs_mx900.ndim == 2 and cand_label_seqs_mx900.shape[1] == 900
    M = cand_label_seqs_mx900.shape[0]
    device = torch.device("cuda")

    # Chunked scoring to avoid OOM when M is large (e.g. 1000).
    bs = int(batch_size) if int(batch_size) > 0 else M
    out = np.zeros((M,), dtype=np.float32)

    inp_1 = torch.from_numpy(inputs_tokens_1x900.astype(np.int32)).to(device).view(1, -1)
    for i0 in range(0, M, bs):
        i1 = min(i0 + bs, M)
        m = i1 - i0
        inp = inp_1.repeat(m, 1)
        labels = torch.from_numpy(cand_label_seqs_mx900[i0:i1].astype(np.int32)).to(device)
        labels = labels.clone()
        labels[labels == 0] = int(IGNORE_LABEL_ID)
        pids = torch.full((m,), int(pid_int), dtype=torch.int32, device=device)
        batch = {"inputs": inp, "labels": labels, "puzzle_identifiers": pids}

        with torch.no_grad():
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            logits = None
            for _ in range(max(int(score_steps), 1)):
                carry, _loss, _metrics, outputs, _all_finish = model(carry=carry, batch=batch, return_keys=["logits"])
                logits = outputs["logits"]
            assert logits is not None
            if nll_type == "stablemax":
                token_nll = stablemax_cross_entropy(logits, labels, ignore_index=int(IGNORE_LABEL_ID))
            elif nll_type == "softmax":
                token_nll = softmax_cross_entropy(logits, labels, ignore_index=int(IGNORE_LABEL_ID))
            else:
                raise ValueError(f"Unknown nll_type: {nll_type}")
            mask = labels != int(IGNORE_LABEL_ID)
            counts = mask.sum(-1).clamp_min(1).to(torch.float32)
            mean_nll = token_nll.sum(-1).to(torch.float32) / counts
            out[i0:i1] = mean_nll.detach().cpu().numpy().astype(np.float32, copy=False)

        # Reduce fragmentation risk
        del batch, labels, inp, pids, carry, logits, token_nll, mask, counts, mean_nll

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="TRM-EXP-02 advanced voting runner (no training)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", default="runs/trm_exp_02_voting_advanced")
    ap.add_argument("--batch_size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--filter_official_eval", choices=["v1", "v2", "concept"], default="v2")
    ap.add_argument("--halt_logit_threshold", type=float, default=0.0)

    # Advanced rescoring
    ap.add_argument("--enable_rescore", action="store_true", help="Enable TRM teacher-forced NLL / PoE rescoring (expensive)")
    ap.add_argument("--nll_type", choices=["stablemax", "softmax"], default="stablemax")
    ap.add_argument("--score_steps", type=int, default=1, help="TRM outer steps to use when scoring candidates")
    ap.add_argument("--poe_num_views", type=int, default=8)
    ap.add_argument("--poe_max_candidates", type=int, default=32)
    ap.add_argument("--poe_gate_top1_frac", type=float, default=0.4)
    ap.add_argument("--poe_batch_size", type=int, default=128, help="Chunk size for PoE/NLL rescoring (prevents OOM for large candidate sets)")
    ap.add_argument("--view_reservoir_size", type=int, default=64)
    ap.add_argument("--rescore_candidate_source", choices=["pooled_all_steps", "last_step"], default="pooled_all_steps")

    # Artifacts controls
    ap.add_argument("--skip_sha256", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--write_per_output_jsonl", action="store_true", help="Write per-output JSONL (can be large)")
    ap.add_argument("--no_submissions", action="store_true", help="Skip writing submission.json files")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    trm_root = repo_root / "third_party" / "TinyRecursiveModels"
    if not trm_root.exists():
        raise SystemExit(f"Missing TRM submodule at: {trm_root} (clone with --recurse-submodules)")
    sys.path.insert(0, str(trm_root))

    # TRM imports
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # type: ignore
    from utils.functions import load_model_class  # type: ignore
    from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy, softmax_cross_entropy  # type: ignore

    # Prefer upstream crop if available (numba), else fallback
    try:  # pragma: no cover
        from evaluators.arc import _crop as arc_crop  # type: ignore
    except Exception:  # pragma: no cover
        arc_crop = _arc_crop_fallback  # type: ignore

    os.makedirs(args.output_dir, exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    # Fingerprints
    dataset_json_path = os.path.join(args.data_path, "test", "dataset.json")
    identifiers_json_path = os.path.join(args.data_path, "identifiers.json")
    test_puzzles_json_path = os.path.join(args.data_path, "test_puzzles.json")
    dataset_json = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
    identifiers = json.loads(Path(identifiers_json_path).read_text(encoding="utf-8"))
    test_puzzles = json.loads(Path(test_puzzles_json_path).read_text(encoding="utf-8"))

    dataset_fingerprint = {
        "dataset_json_path": dataset_json_path,
        "identifiers_json_path": identifiers_json_path,
        "test_puzzles_json_path": test_puzzles_json_path,
        "dataset_json_sha256": _sha256_json_bytes(dataset_json),
        "identifiers_json_sha256": _sha256_json_bytes(identifiers),
        "test_puzzles_json_sha256": _sha256_json_bytes(test_puzzles),
        "metadata": dataset_json,
    }
    Path(os.path.join(args.output_dir, "dataset_fingerprint.json")).write_text(
        json.dumps(dataset_fingerprint, indent=2), encoding="utf-8"
    )

    if not args.skip_sha256:
        try:
            ckpt_sha = _sha256_file(args.checkpoint)
            Path(os.path.join(args.output_dir, "checkpoint_sha256.txt")).write_text(ckpt_sha + "\n", encoding="utf-8")
        except Exception as e:
            Path(os.path.join(args.output_dir, "checkpoint_sha256.txt")).write_text(f"error: {e}\n", encoding="utf-8")

    # Filter tasks to official list
    if args.filter_official_eval:
        official = _load_official_eval_task_ids(trm_root=trm_root, eval_version=str(args.filter_official_eval))
        if official is not None:
            test_puzzles = {k: v for k, v in test_puzzles.items() if k in official}

    # GT outputs list
    gt_outputs: List[Tuple[str, str, str]] = []
    for task_name, puzzle in test_puzzles.items():
        for pair in puzzle.get("test", []):
            inp_h = grid_hash(arc_grid_to_np(pair["input"]))
            lab_h = grid_hash(arc_grid_to_np(pair["output"]))
            gt_outputs.append((task_name, inp_h, lab_h))

    # Dataloader
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=args.seed,
            dataset_paths=[args.data_path],
            rank=0,
            num_replicas=1,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=args.batch_size,
        ),
        split="test",
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=1, prefetch_factor=8, pin_memory=True, persistent_workers=True)
    metadata = dataset.metadata

    # Model
    ckpt_cfg = _try_load_checkpoint_config(args.checkpoint)
    ckpt_arch = _arch_from_checkpoint_config(ckpt_cfg)
    model_cfg = dict(
        batch_size=args.batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
        halt_exploration_prob=0.1,
        halt_max_steps=args.max_steps,
        H_cycles=3,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        num_heads=8,
        expansion=4,
        puzzle_emb_ndim=512,
        pos_encodings="rope",
        forward_dtype="bfloat16",
        mlp_t=False,
        puzzle_emb_len=16,
        no_ACT_continue=True,
    )
    if ckpt_arch:
        allowed = set(model_cfg.keys())
        for k, v in dict(ckpt_arch).items():
            if k in allowed and v is not None:
                model_cfg[k] = v
    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    loss_cls = load_model_class("losses@ACTLossHead")
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_cls(model, loss_type="stablemax_cross_entropy")
    sd = torch.load(args.checkpoint, map_location="cuda")
    if isinstance(sd, dict):
        sd = {(k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, assign=True)
    model = model.cuda().eval()

    if "DISABLE_COMPILE" not in os.environ:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Vote stats
    max_steps = int(args.max_steps)
    votes_by_step: List[Dict[Tuple[str, str], Dict[str, Tuple[int, float]]]] = [defaultdict(dict) for _ in range(max_steps)]
    votes_by_halt: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
    hmap: Dict[str, np.ndarray] = {}
    halt_hist: DefaultDict[int, int] = defaultdict(int)

    # View reservoir for PoE rescoring
    view_reservoir: Dict[Tuple[str, str], List[ViewRecord]] = defaultdict(list)
    view_seen: Dict[Tuple[str, str], set] = defaultdict(set)

    def _reservoir_add(reservoir: List[ViewRecord], seen: set, item: ViewRecord, max_size: int) -> None:
        if item.pid_int in seen:
            return
        if len(reservoir) >= max_size:
            return
        reservoir.append(item)
        seen.add(item.pid_int)

    # Inference
    start_wall = time.time()
    processed_batches = 0
    processed_examples = 0
    for _set_name, batch_cpu, _gb in loader:
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
        with torch.no_grad():
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            all_preds: List[Dict[str, torch.Tensor]] = []
            for _s in range(max_steps):
                carry, _loss, _metrics, preds, _all_finish = model(carry=carry, batch=batch, return_keys=["preds", "q_halt_logits"])
                all_preds.append({"preds": preds["preds"].clone(), "q_halt_logits": preds["q_halt_logits"].clone()})

        inputs_np = batch_cpu["inputs"].numpy()
        pids_np = batch_cpu["puzzle_identifiers"].numpy()
        B = int(inputs_np.shape[0])
        for b_idx in range(B):
            pid_int = int(pids_np[b_idx])
            name = identifiers[pid_int]
            base_name, inv_fn = inverse_aug(name)
            try:
                inp_grid = inv_fn(arc_crop(inputs_np[b_idx]))
                if inp_grid.size == 0:
                    continue
                inp_hash = grid_hash(inp_grid.astype(np.uint8))
            except Exception:
                continue

            key = (base_name, inp_hash)

            if args.view_reservoir_size > 0:
                vr = ViewRecord(pid_int=pid_int, name=name, inputs_tokens=inputs_np[b_idx].astype(np.int16, copy=True))
                _reservoir_add(view_reservoir[key], view_seen[key], vr, int(args.view_reservoir_size))

            theoretical_halt_step = -1
            halt_pred_hash = None
            halt_sumq = 0.0

            for s in range(max_steps):
                q_halt = float(all_preds[s]["q_halt_logits"][b_idx].item())
                qsig = _sigmoid_float(q_halt)
                pred_tokens = all_preds[s]["preds"][b_idx].detach().cpu().numpy()
                try:
                    pred_grid = inv_fn(arc_crop(pred_tokens))
                    if pred_grid.size == 0:
                        continue
                    pred_hash = grid_hash(pred_grid.astype(np.uint8))
                    hmap[pred_hash] = pred_grid.astype(np.uint8)
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

        del batch, batch_cpu, carry, all_preds
        if processed_batches % 100 == 0:
            torch.cuda.empty_cache()

    wall_infer = time.time() - start_wall

    # Offline strategy matrix
    pass_Ks = (1, 2, 5, 10, 100, 1000)
    strategies: Dict[str, Dict[str, float]] = {}

    baseline_votes = votes_by_step[max_steps - 1]
    if not args.no_submissions:
        sub_dir = os.path.join(args.output_dir, "submission_baseline_last_step")
    else:
        sub_dir = None
    strategies["baseline_last_step_count_avgq"] = _compute_pass_metrics_from_votes(
        test_puzzles=test_puzzles,
        gt_outputs=gt_outputs,
        votes=baseline_votes,
        pass_Ks=pass_Ks,
        submission_K=2,
        hmap=hmap,
        save_submission_path=sub_dir,
        rank_by="count_avgq",
    )

    # Per-step curves + best step
    step_curves: Dict[str, List[float]] = {"pass1_per_output": [], "pass2_per_output": []}
    best_step = 0
    best_score = -1.0
    per_step_metrics: List[Dict[str, float]] = []
    for s in range(max_steps):
        m = _compute_pass_metrics_from_votes(
            test_puzzles=test_puzzles,
            gt_outputs=gt_outputs,
            votes=votes_by_step[s],
            pass_Ks=pass_Ks,
            submission_K=2,
            hmap=hmap,
            save_submission_path=None,
            rank_by="count_avgq",
        )
        per_step_metrics.append(m)
        step_curves["pass1_per_output"].append(float(m.get("ARC/pass@1_per_output", 0.0)))
        step_curves["pass2_per_output"].append(float(m.get("ARC/pass@2_per_output", 0.0)))
        if float(m.get("ARC/pass@2_per_output", -1.0)) > best_score:
            best_score = float(m.get("ARC/pass@2_per_output", -1.0))
            best_step = s

    if not args.no_submissions:
        sub_dir = os.path.join(args.output_dir, f"submission_best_fixed_step_{best_step}")
    else:
        sub_dir = None
    strategies["best_fixed_step_count_avgq"] = _compute_pass_metrics_from_votes(
        test_puzzles=test_puzzles,
        gt_outputs=gt_outputs,
        votes=votes_by_step[best_step],
        pass_Ks=pass_Ks,
        submission_K=2,
        hmap=hmap,
        save_submission_path=sub_dir,
        rank_by="count_avgq",
    )
    strategies["best_fixed_step_count_avgq"]["_best_step"] = float(best_step)

    pooled_all = _pool_votes_across_steps(votes_by_step, steps=list(range(max_steps)))
    strategies["pooled_all_steps_count_avgq"] = _compute_pass_metrics_from_votes(
        test_puzzles=test_puzzles,
        gt_outputs=gt_outputs,
        votes=pooled_all,
        pass_Ks=pass_Ks,
        submission_K=2,
        hmap=hmap,
        save_submission_path=None if args.no_submissions else os.path.join(args.output_dir, "submission_pooled_all_steps"),
        rank_by="count_avgq",
    )
    strategies["pooled_all_steps_sumq_count"] = _compute_pass_metrics_from_votes(
        test_puzzles=test_puzzles,
        gt_outputs=gt_outputs,
        votes=pooled_all,
        pass_Ks=pass_Ks,
        submission_K=2,
        hmap=hmap,
        save_submission_path=None if args.no_submissions else os.path.join(args.output_dir, "submission_pooled_all_steps_sumq"),
        rank_by="sumq_count",
    )
    strategies["simulated_halting_count_avgq"] = _compute_pass_metrics_from_votes(
        test_puzzles=test_puzzles,
        gt_outputs=gt_outputs,
        votes=votes_by_halt,
        pass_Ks=pass_Ks,
        submission_K=2,
        hmap=hmap,
        save_submission_path=None if args.no_submissions else os.path.join(args.output_dir, "submission_simulated_halting"),
        rank_by="count_avgq",
    )

    # Optional PoE/NLL rescoring (TRM-as-scorer)
    rescoring_summary = None
    if args.enable_rescore:
        t_rescore_start = time.time()
        poe_votes = pooled_all if args.rescore_candidate_source == "pooled_all_steps" else baseline_votes
        M = int(args.poe_max_candidates)
        V = int(args.poe_num_views)
        gate = float(args.poe_gate_top1_frac)

        poe_selected: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
        num_rescored = 0
        num_skipped_gate = 0
        num_missing_views = 0

        for (task_name, inp_h, _lab_h) in gt_outputs:
            key = (task_name, inp_h)
            p_map = poe_votes.get(key)
            if not p_map:
                continue
            ranked = _rank_candidates(p_map, rank_by="count_avgq")[: max(M, 2)]
            if len(ranked) == 0:
                continue

            if gate > 0:
                total_votes = sum(int(cnt) for (cnt, _sumq) in p_map.values())
                top1_votes = int(p_map.get(ranked[0], (0, 0.0))[0])
                top1_frac = (top1_votes / max(total_votes, 1)) if total_votes > 0 else 0.0
                if top1_frac >= gate:
                    out = poe_selected.setdefault(key, {})
                    for h in ranked[:2]:
                        out[h] = (1, 1.0)
                    num_skipped_gate += 1
                    continue

            views = view_reservoir.get(key, [])
            if len(views) == 0:
                num_missing_views += 1
                continue
            if len(views) > V:
                views = rng.sample(views, V)

            cand_grids = []
            for h in ranked:
                g = hmap.get(h)
                if g is not None:
                    cand_grids.append((h, g))
            if len(cand_grids) == 0:
                continue

            scores = np.zeros((len(cand_grids),), dtype=np.float32)
            for vr in views:
                _base, dihedral_id, mapping = _parse_view_aug(vr.name)
                labels_m = []
                for _h, g in cand_grids:
                    gv = _apply_view_aug_to_grid(g, dihedral_id=dihedral_id, mapping_0_9=mapping)
                    labels_m.append(_grid_to_seq_tokens_no_translation(gv))
                labels_mx900 = np.stack(labels_m, 0).astype(np.int32)
                nll = _score_candidates_nll_for_view(
                    model=model,
                    IGNORE_LABEL_ID=IGNORE_LABEL_ID,
                    stablemax_cross_entropy=stablemax_cross_entropy,
                    softmax_cross_entropy=softmax_cross_entropy,
                    inputs_tokens_1x900=vr.inputs_tokens.astype(np.int32),
                    pid_int=int(vr.pid_int),
                    cand_label_seqs_mx900=labels_mx900,
                    score_steps=int(args.score_steps),
                    nll_type=str(args.nll_type),
                    batch_size=int(args.poe_batch_size),
                )
                scores += nll.astype(np.float32)

            order = np.argsort(scores)
            top = [cand_grids[i][0] for i in order[:2]]
            out = poe_selected.setdefault(key, {})
            for h in top:
                out[h] = (1, 1.0)
            num_rescored += 1

        strategies["poe_rescore_nll"] = _compute_pass_metrics_from_votes(
            test_puzzles=test_puzzles,
            gt_outputs=gt_outputs,
            votes=poe_selected,
            pass_Ks=pass_Ks,
            submission_K=2,
            hmap=hmap,
            save_submission_path=None if args.no_submissions else os.path.join(args.output_dir, "submission_poe_rescore_nll"),
            rank_by="count_avgq",
        )

        rescoring_summary = {
            "num_outputs_total": int(max(len(gt_outputs), 1)),
            "num_outputs_rescored": int(num_rescored),
            "num_outputs_skipped_gate": int(num_skipped_gate),
            "num_outputs_missing_views": int(num_missing_views),
            "poe_num_views": int(V),
            "poe_max_candidates": int(M),
            "poe_gate_top1_frac": float(gate),
            "rescore_candidate_source": str(args.rescore_candidate_source),
            "score_steps": int(args.score_steps),
            "nll_type": str(args.nll_type),
            "wall_rescore_s": float(time.time() - t_rescore_start),
        }

    # Choose best by pass@2_per_output
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
            "checkpoint": str(args.checkpoint),
            "data_path": str(args.data_path),
            "batch_size": int(args.batch_size),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "filter_official_eval": str(args.filter_official_eval),
            "halt_logit_threshold": float(args.halt_logit_threshold),
            "enable_rescore": bool(args.enable_rescore),
            "rescore_candidate_source": str(args.rescore_candidate_source),
            "poe_num_views": int(args.poe_num_views),
            "poe_max_candidates": int(args.poe_max_candidates),
            "poe_gate_top1_frac": float(args.poe_gate_top1_frac),
            "score_steps": int(args.score_steps),
            "nll_type": str(args.nll_type),
        },
        "timing": {
            "wall_inference_s": float(wall_infer),
            "wall_total_s": float(time.time() - start_wall),
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
            "halt_hist": {str(k): int(v) for k, v in sorted(halt_hist.items(), key=lambda kv: int(kv[0]))},
        },
        "rescoring": rescoring_summary,
    }

    Path(os.path.join(args.output_dir, "voting_report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Optional per-output JSONL (can be large; off by default)
    if args.write_per_output_jsonl:
        per_out_path = os.path.join(args.output_dir, "per_output.jsonl")
        with open(per_out_path, "w", encoding="utf-8") as f:
            baseline_votes = votes_by_step[max_steps - 1]
            pooled_all = _pool_votes_across_steps(votes_by_step, steps=list(range(max_steps)))

            def _topk(p_map: Dict[str, Tuple[int, float]], k: int = 5):
                ranked = _rank_candidates(p_map, rank_by="count_avgq")[:k] if p_map else []
                out = []
                for h in ranked:
                    cnt, sumq = p_map[h]
                    out.append({"pred_hash": h, "count": int(cnt), "avg_q": float(sumq) / max(int(cnt), 1)})
                return out

            for (task_name, inp_h, lab_h) in gt_outputs:
                rec = {"task": task_name, "input_hash": inp_h, "label_hash": lab_h}
                rec["top5_last_step"] = _topk(baseline_votes.get((task_name, inp_h), {}))
                rec["top5_pooled_all"] = _topk(pooled_all.get((task_name, inp_h), {}))
                f.write(json.dumps(rec) + "\n")

    # Optional plots: run the plotting script separately (CPU-only).
    if args.plots:
        print("[plots] Run manually (CPU): python scripts/plot_voting_report.py --report .../voting_report.json --out_dir .../plots")


if __name__ == "__main__":
    main()

