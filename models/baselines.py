"""Stage 7: baselines to beat before training the real model.

Random-legal-move baseline (sanity check, expected near-zero accuracy) and
Stockfish-always-best-move baseline: how often does the objectively best move
match what Aman actually played? This isn't just a bar to beat -- it's itself
an interesting stat (it measures how "engine-like" vs. "human" Aman's bullet
play is), and the spec asks for it to be reported regardless of the number.

Evaluated on a fixed-seed sample of the personal test split, not the full
~118k-row test set -- at even 50ms/move, Stockfish over the full test set
would take ~1.5 hours, which buys nothing here: Stage 7's job is a quick
sanity gate before training starts, not the final ablation. The full,
rigorous four-way comparison (random / Stockfish / pretrained / fine-tuned) on
the complete held-out test set is Stage 10's job.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SAMPLE_SIZE = 2000
SEED = 42
STOCKFISH_TIME_LIMIT_SECONDS = 0.05


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_split(name: str = "personal") -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    with open(PROCESSED_DIR / f"{name}_splits.json", "r", encoding="utf-8") as f:
        assignment = json.load(f)
    df["split"] = df["game_id"].map(assignment)
    return df[df["split"] == "test"].reset_index(drop=True)


def random_move_accuracy(rows: pd.DataFrame, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    correct = 0
    for fen, actual_uci in zip(rows["fen"], rows["move_uci"], strict=True):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        predicted = legal_moves[rng.integers(len(legal_moves))]
        if predicted.uci() == actual_uci:
            correct += 1
    return correct / len(rows)


def stockfish_best_move_accuracy(
    rows: pd.DataFrame, engine_path: str, time_limit: float = STOCKFISH_TIME_LIMIT_SECONDS
) -> float:
    correct = 0
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for fen, actual_uci in zip(rows["fen"], rows["move_uci"], strict=True):
            board = chess.Board(fen)
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            if result.move is not None and result.move.uci() == actual_uci:
                correct += 1
    return correct / len(rows)


def main() -> None:
    config = load_config()
    test_df = load_test_split("personal")

    rng = np.random.default_rng(SEED)
    sample_size = min(SAMPLE_SIZE, len(test_df))
    sample = test_df.iloc[rng.choice(len(test_df), size=sample_size, replace=False)].reset_index(drop=True)
    print(f"Evaluating baselines on a {sample_size:,}-row sample of the personal test split ({len(test_df):,} total rows)...")

    random_acc = random_move_accuracy(sample)
    print(f"Random-legal-move accuracy: {random_acc:.4f}")

    engine_path = config["engine"]["stockfish_path"]
    sf_acc = stockfish_best_move_accuracy(sample, engine_path)
    print(f"Stockfish-best-move accuracy: {sf_acc:.4f}")

    summary = {
        "sample_size": sample_size,
        "test_split_total_rows": len(test_df),
        "random_baseline_accuracy": random_acc,
        "stockfish_baseline_accuracy": sf_acc,
        "stockfish_time_limit_seconds": STOCKFISH_TIME_LIMIT_SECONDS,
    }
    with open(PROCESSED_DIR / "baselines_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
