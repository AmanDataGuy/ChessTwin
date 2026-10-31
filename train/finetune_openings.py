"""Targeted opening specialization -- a further, focused pass on top of the
main fine-tuned checkpoint, training only on early-game positions
(move_number < EARLY_MOVE_THRESHOLD) instead of the full mixed dataset.

Rationale (see technical_report.md "Run 4"): two different full-retrain
hyperparameter changes each moved Aman's specific opening probabilities by
2-3x in uncorrelated directions while barely touching the aggregate
move-matching metric -- broad retraining reshuffles which opening "wins" more
than it reliably improves personalization for a *specific* named opening.
This script targets the problem directly instead: a few gentle-LR epochs
exposing the model to nothing but its own early-game training distribution,
starting from the already-competent fine-tuned model rather than from
scratch, so midgame/endgame competence (learned from the full dataset) isn't
overwritten -- only nudged on top of by additional focused exposure to
openings specifically.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.encoding import MoveVocabulary, board_to_tensor
from models.network import PolicyNetwork
from train.pretrain import evaluate, load_config, top_k_accuracy

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

EARLY_MOVE_THRESHOLD = 10
EPOCHS = 4
LR = 5e-5  # gentler than the main finetune_lr (1e-4) -- this is a nudge on an already fine-tuned model, not a restart


class EarlyMoveDataset(Dataset):
    def __init__(self, split: str, vocab: MoveVocabulary):
        df = pd.read_parquet(
            PROCESSED_DIR / "personal.parquet", columns=["game_id", "fen", "move_uci", "move_number"]
        )
        with open(PROCESSED_DIR / "personal_splits.json", "r", encoding="utf-8") as f:
            assignment = json.load(f)
        df["split"] = df["game_id"].map(assignment)
        df = df[(df["split"] == split) & (df["move_number"] < EARLY_MOVE_THRESHOLD)].reset_index(drop=True)
        self.df = df
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        board = chess.Board(row["fen"])
        tensor = board_to_tensor(board)
        label = self.vocab.encode(chess.Move.from_uci(row["move_uci"]))
        return torch.from_numpy(tensor), label


def main() -> None:
    config = load_config()
    vocab = MoveVocabulary.load_or_build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, vocab size: {len(vocab)}")

    base_path = CHECKPOINT_DIR / "finetuned.pt"
    checkpoint = torch.load(base_path, map_location=device)
    model = PolicyNetwork.from_config(checkpoint["config"], vocab_size=checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded base checkpoint from {base_path}")

    train_ds = EarlyMoveDataset("train", vocab)
    val_ds = EarlyMoveDataset("val", vocab)
    print(f"Train: {len(train_ds):,} early-move positions, Val: {len(val_ds):,}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    step = 0
    for epoch in range(EPOCHS):
        for boards, labels in train_loader:
            boards, labels = boards.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(boards)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % 100 == 0:
                top1 = top_k_accuracy(logits, labels, 1)
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} top1 {top1:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val (early-moves only): {val_metrics}")

    out_path = CHECKPOINT_DIR / "finetuned_openings.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "vocab_size": len(vocab), "config": checkpoint["config"]},
        out_path,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
