import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight torch stub so helper-only imports work in CPU-only test envs.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = object
    torch_stub.utils = torch_utils
    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data

from scripts.eval_voting import _compute_pass_metrics_from_votes, _rank_candidates, grid_hash


def test_grid_hash_changes_on_shape_and_content():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    b = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    c = np.array([[1, 2, 0], [3, 4, 0]], dtype=np.uint8)
    d = np.array([[1, 2], [3, 5]], dtype=np.uint8)

    assert grid_hash(a) == grid_hash(b)
    assert grid_hash(a) != grid_hash(c)
    assert grid_hash(a) != grid_hash(d)


def test_rank_candidates_tie_breakers_are_deterministic():
    stats_map = {
        "h1": (2, 0.2),
        "h2": (2, 1.8),
        "h3": (1, 3.0),
    }

    assert _rank_candidates(stats_map, rank_by="count_avgq") == ["h2", "h1", "h3"]
    assert _rank_candidates(stats_map, rank_by="sumq_count") == ["h3", "h2", "h1"]


def test_compute_pass_metrics_per_task_vs_per_output():
    task1_input1 = [[0]]
    task1_output1 = [[1]]
    task1_input2 = [[2]]
    task1_output2 = [[3]]
    task2_input1 = [[4]]
    task2_output1 = [[5]]

    test_puzzles = {
        "task1": {"test": [{"input": task1_input1, "output": task1_output1}, {"input": task1_input2, "output": task1_output2}]},
        "task2": {"test": [{"input": task2_input1, "output": task2_output1}]},
    }

    in1_h = grid_hash(np.array(task1_input1, dtype=np.uint8))
    out1_h = grid_hash(np.array(task1_output1, dtype=np.uint8))
    in2_h = grid_hash(np.array(task1_input2, dtype=np.uint8))
    out2_h = grid_hash(np.array(task1_output2, dtype=np.uint8))
    in3_h = grid_hash(np.array(task2_input1, dtype=np.uint8))
    out3_h = grid_hash(np.array(task2_output1, dtype=np.uint8))

    votes = {
        ("task1", in1_h): {out1_h: (2, 1.0)},
        ("task1", in2_h): {"wrong": (2, 1.0)},
        ("task2", in3_h): {out3_h: (2, 1.0)},
    }

    metrics = _compute_pass_metrics_from_votes(
        test_puzzles=test_puzzles,
        gt_outputs=[("task1", in1_h, out1_h), ("task1", in2_h, out2_h), ("task2", in3_h, out3_h)],
        votes=votes,
        pass_Ks=(1,),
    )

    assert metrics["ARC/pass@1"] == 0.75
    assert metrics["ARC/pass@1_per_output"] == (2.0 / 3.0)
