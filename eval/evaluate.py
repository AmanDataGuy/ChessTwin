"""Stage 10: the headline ablation table.

Top-1/top-3 move-prediction accuracy for all four comparison points -- random
baseline, Stockfish-best-move baseline, pretrained-only model, fine-tuned
model -- on a fixed-seed sample of the held-out personal test split. This
table is the project's core deliverable (spec Section 6, Stage 10): does
fine-tuning on personal data actually help, and by how much?

Evaluated on a sample, not the full ~118k-row test set, for the same reason
as Stage 7's baselines: Stockfish at 50ms/move over the full set would take
~1.6 hours. The sample here (5,000 rows) is larger than Stage 7's quick
sanity-check sample (2,000) since this is the number that goes in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch

from models.baselines import (
    load_config,
    load_test_split,
    random_move_accuracy,
    stockfish_best_move_accuracy,
)
from models.encoding import MoveVocabulary, board_to_tensor
from models.network import PolicyNetwork

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

EVAL_SAMPLE_SIZE = 5000
SEED = 42


def model_topk_accuracy(model: PolicyNetwork, vocab: MoveVocabulary, rows, device: torch.device) -> dict:
    """Top-1/top-3 accuracy with legal-move masking applied before ranking."""
    model.eval()
    correct_top1 = correct_top3 = 0
    with torch.no_grad():
        for fen, actual_uci in zip(rows["fen"], rows["move_uci"], strict=True):
            board = chess.Board(fen)
            tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
            logits = model(tensor).squeeze(0).cpu()
            mask = torch.from_numpy(vocab.legal_move_mask(board))
            logits = logits.masked_fill(~mask, float("-inf"))

            actual_idx = vocab.encode(chess.Move.from_uci(actual_uci))
            top3 = logits.topk(3).indices.tolist()
            if actual_idx == top3[0]:
                correct_top1 += 1
            if actual_idx in top3:
                correct_top3 += 1
    n = len(rows)
    return {"top1_acc": correct_top1 / n, "top3_acc": correct_top3 / n}


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PolicyNetwork, MoveVocabulary]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = MoveVocabulary.load_or_build()
    model = PolicyNetwork.from_config(checkpoint["config"], vocab_size=checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab


def main() -> None:
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = load_test_split("personal")
    rng = np.random.default_rng(SEED)
    sample_size = min(EVAL_SAMPLE_SIZE, len(test_df))
    sample = test_df.iloc[rng.choice(len(test_df), size=sample_size, replace=False)].reset_index(drop=True)
    print(f"Evaluating on {sample_size:,} test positions (of {len(test_df):,} total)")

    results = {}

    print("Random baseline...")
    results["random"] = {"top1_acc": random_move_accuracy(sample)}

    print("Stockfish-best-move baseline...")
    engine_path = str(REPO_ROOT / config["engine"]["stockfish_path"])
    results["stockfish_best_move"] = {"top1_acc": stockfish_best_move_accuracy(sample, engine_path)}

    pretrained_path = CHECKPOINT_DIR / "pretrained.pt"
    if pretrained_path.exists():
        print("Pretrained-only model...")
        model, vocab = load_model(pretrained_path, device)
        results["pretrained_only"] = model_topk_accuracy(model, vocab, sample, device)
    else:
        print(f"Skipping pretrained-only: {pretrained_path} not found.")
        results["pretrained_only"] = None

    finetuned_path = CHECKPOINT_DIR / "finetuned.pt"
    if finetuned_path.exists():
        print("Fine-tuned model...")
        model, vocab = load_model(finetuned_path, device)
        results["finetuned"] = model_topk_accuracy(model, vocab, sample, device)
    else:
        print(f"Skipping fine-tuned: {finetuned_path} not found.")
        results["finetuned"] = None

    print(json.dumps(results, indent=2))
    with open(PROCESSED_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({"sample_size": sample_size, "test_split_total_rows": len(test_df), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
