"""Stage 5: game-level train/val/test split, seeded and persisted for reproducibility.

Splits by game_id, not by row/position -- positions within a game are
correlated (they share context), so splitting at the position level would leak
information between train and eval and inflate reported metrics. This is a
known failure mode the spec explicitly calls out; the split is constructed at
the game level in every case here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def split_games(game_ids: list[str], seed: int = SEED) -> dict[str, str]:
    """Deterministically assign each game_id to train/val/test, ~80/10/10."""
    ids = sorted(set(game_ids))  # sort first so the result doesn't depend on input order
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    assignment: dict[str, str] = {}
    for i, game_id in enumerate(ids):
        if i < n_train:
            assignment[game_id] = "train"
        elif i < n_train + n_val:
            assignment[game_id] = "val"
        else:
            assignment[game_id] = "test"
    return assignment


def build_and_save(name: str) -> dict:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=["game_id"])
    assignment = split_games(df["game_id"].tolist())

    out_path = PROCESSED_DIR / f"{name}_splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(assignment, f)

    game_counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        game_counts[split] += 1

    row_counts = df["game_id"].map(assignment).value_counts().to_dict()

    return {
        "games": game_counts,
        "rows": row_counts,
        "output_file": str(out_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    summary = {}
    for name in ["personal", "pretrain"]:
        print(f"Splitting {name}...")
        result = build_and_save(name)
        summary[name] = result
        print(json.dumps(result, indent=2))

    with open(PROCESSED_DIR / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
