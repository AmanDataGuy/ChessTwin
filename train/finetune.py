"""Stage 9: fine-tune the pretrained base model on Aman's personal games only.

Loads the Stage 8 checkpoint and continues training on the personal dataset at
a lower learning rate than pretraining, same cross-entropy objective. This is
the step that turns a generic-human policy into a personalized one -- the
spec's core hypothesis (chess_persona_project_spec.md Section 1) is that this
step measurably improves move-prediction accuracy on Aman's own held-out games
over both the pretrained-only model and a raw Stockfish baseline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb
from models.encoding import MoveVocabulary
from models.network import PolicyNetwork
from train.pretrain import PositionDataset, evaluate, load_config, top_k_accuracy

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def main() -> None:
    config = load_config()
    vocab = MoveVocabulary.load_or_build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, vocab size: {len(vocab)}")

    base_checkpoint_path = CHECKPOINT_DIR / "pretrained.pt"
    checkpoint = torch.load(base_checkpoint_path, map_location=device)
    model = PolicyNetwork.from_config(config, vocab_size=checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded base checkpoint from {base_checkpoint_path}")

    max_train = config["training"].get("finetune_max_train_positions")
    max_val = config["training"].get("finetune_max_val_positions")
    oversample_factor = config["training"].get("finetune_oversample_early_moves_factor", 1)
    train_ds = PositionDataset(
        "personal", "train", vocab, max_rows=max_train, oversample_early_moves_factor=oversample_factor
    )
    val_ds = PositionDataset("personal", "val", vocab, max_rows=max_val)  # no oversampling: val stays unbiased
    print(f"Train: {len(train_ds):,} positions (oversample factor {oversample_factor}), Val: {len(val_ds):,} positions")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["finetune_lr"])
    criterion = nn.CrossEntropyLoss()
    use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "finetuned.pt"
    resume_path = CHECKPOINT_DIR / "finetuned_resume.pt"
    start_epoch = 0
    if resume_path.exists():
        resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = resume_checkpoint["epoch"] + 1
        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    wandb.init(project="chess-persona", name="finetune", mode=config["training"]["wandb_mode"], config=config)

    epochs = config["training"]["finetune_epochs"]
    step = 0
    for epoch in range(start_epoch, epochs):
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
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} top1 {top1:.4f}")
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
    resume_path.unlink(missing_ok=True)
    print(f"Saved checkpoint to {checkpoint_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
