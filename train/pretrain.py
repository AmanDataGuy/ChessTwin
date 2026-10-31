"""Stage 8: pretrain the policy network from scratch on the general Lichess dataset.

Trains the ResNet policy head with cross-entropy loss (supervised behavioral
cloning, not RL, per spec Section 2) to predict human moves. Tracks loss and
top-1/top-3 accuracy on the pretrain val split via wandb (offline mode by
default -- no account needed). Saves the resulting checkpoint as the base
model that finetune.py continues training from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import chess
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

import wandb
from models.encoding import MoveVocabulary, board_to_tensor
from models.network import PolicyNetwork

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PositionDataset(Dataset):
    """FEN + move UCI -> (board tensor, vocab index), computed on the fly.

    Re-deriving the tensor/index from the stored FEN/UCI at load time (instead
    of a training-time cache) is cheap relative to GPU compute and keeps the
    dataset decoupled from any one vocabulary ordering (see data/process_pgn.py).
    """

    def __init__(
        self,
        name: str,
        split: str,
        vocab: MoveVocabulary,
        max_rows: int | None = None,
        oversample_early_moves_factor: int = 1,
        early_move_threshold: int = 10,
    ):
        columns = ["game_id", "fen", "move_uci"]
        if oversample_early_moves_factor > 1:
            columns.append("move_number")
        df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns)
        with open(PROCESSED_DIR / f"{name}_splits.json", "r", encoding="utf-8") as f:
            assignment = json.load(f)
        df["split"] = df["game_id"].map(assignment)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_rows is not None and len(df) > max_rows:
            # Deterministic sample, not a truncation, so we're not just training on
            # the earliest games -- a fixed seed keeps it reproducible.
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

        if oversample_early_moves_factor > 1:
            # Opening/early-middlegame positions are the ones a specific player's
            # style is most recognizable in, but they're a small fraction of total
            # rows (most positions in a dataset are mid/endgame). Duplicating them
            # amplifies that signal during fine-tuning without touching val (which
            # is built without oversampling, so it stays an unbiased measure).
            early = df[df["move_number"] < early_move_threshold]
            extra_copies = [early] * (oversample_early_moves_factor - 1)
            df = pd.concat([df, *extra_copies], ignore_index=True)

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


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total_top1 = total_top3 = 0.0
    n_batches = 0
    with torch.no_grad():
        for boards, labels in loader:
            boards, labels = boards.to(device), labels.to(device)
            logits = model(boards)
            total_loss += criterion(logits, labels).item()
            total_top1 += top_k_accuracy(logits, labels, 1)
            total_top3 += top_k_accuracy(logits, labels, 3)
            n_batches += 1
    model.train()
    return {"loss": total_loss / n_batches, "top1_acc": total_top1 / n_batches, "top3_acc": total_top3 / n_batches}


def main() -> None:
    config = load_config()
    vocab = MoveVocabulary.load_or_build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, vocab size: {len(vocab)}")

    # Windows' spawn-based multiprocessing re-pickles the whole loaded dataframe to
    # each DataLoader worker; on this machine's available RAM that duplication
    # reliably OOMs at num_workers>0 (confirmed by testing), so training runs
    # single-process. To still finish in a reasonable wall-clock time on a single
    # CPU core doing FEN->tensor encoding, train/val are capped to a fixed sample
    # size rather than the full 4.17M/517k rows -- a deliberate, logged time-budget
    # tradeoff, not a silent shortcut (see README's limitations section).
    max_train = config["training"].get("pretrain_max_train_positions")
    max_val = config["training"].get("pretrain_max_val_positions")
    train_ds = PositionDataset("pretrain", "train", vocab, max_rows=max_train)
    val_ds = PositionDataset("pretrain", "val", vocab, max_rows=max_val)
    print(f"Train: {len(train_ds):,} positions, Val: {len(val_ds):,} positions")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PolicyNetwork.from_config(config, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["pretrain_lr"])
    criterion = nn.CrossEntropyLoss()
    use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "pretrained.pt"
    resume_path = CHECKPOINT_DIR / "pretrained_resume.pt"
    start_epoch = 0
    # This machine has hit multiple unrelated crashes mid-training this session
    # (a wandb IPC failure, a transient CUDA driver error) -- resuming from the
    # last completed epoch instead of restarting from scratch each time turns a
    # crash into a minor interruption instead of hours of lost progress.
    if resume_path.exists():
        resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = resume_checkpoint["epoch"] + 1
        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    wandb.init(project="chess-persona", name="pretrain", mode=config["training"]["wandb_mode"], config=config)

    epochs = config["training"]["pretrain_epochs"]
    step = 0
    for epoch in range(start_epoch, epochs):
        start = time.time()
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
            if step % 200 == 0:
                top1 = top_k_accuracy(logits, labels, 1)
                elapsed = time.time() - start
                rate = step * batch_size / elapsed
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} top1 {top1:.4f} ({rate:.0f} pos/s)")
                wandb.log({"train/loss": loss.item(), "train/top1": top1, "step": step})

        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val: {val_metrics}")
        wandb.log({**{f"val/{k}": v for k, v in val_metrics.items()}, "epoch": epoch})

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "vocab_size": len(vocab),
                "config": config,
            },
            resume_path,
        )
        print(f"Saved resume checkpoint ({epoch=}) to {resume_path}")

    torch.save(
        {"model_state_dict": model.state_dict(), "vocab_size": len(vocab), "config": config},
        checkpoint_path,
    )
    resume_path.unlink(missing_ok=True)  # training finished normally; the resume file has served its purpose
    print(f"Saved checkpoint to {checkpoint_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
