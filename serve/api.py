"""Stage 11: FastAPI serving layer -- FEN in, ranked moves + confidence out.

Loads the fine-tuned checkpoint if present, falling back to the pretrained-only
checkpoint (useful for the pretrained-vs-finetuned comparison and for testing
before finetune.py has run). This module is the internal layer serve/bot.py
calls into for its Lichess Bot API integration; it also stands alone as a
plain HTTP API.
"""

from __future__ import annotations

from pathlib import Path

import chess
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.encoding import MoveVocabulary, board_to_tensor
from models.network import PolicyNetwork

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

app = FastAPI(title="ChessPersona")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: PolicyNetwork | None = None
_vocab: MoveVocabulary | None = None
_checkpoint_name: str | None = None


class MoveRequest(BaseModel):
    fen: str
    top_k: int = 5


class RankedMove(BaseModel):
    move: str
    confidence: float


class MoveResponse(BaseModel):
    moves: list[RankedMove]
    model_checkpoint: str


def _load_model() -> tuple[PolicyNetwork, MoveVocabulary, str]:
    global _model, _vocab, _checkpoint_name
    if _model is not None:
        return _model, _vocab, _checkpoint_name

    for name in ("finetuned.pt", "pretrained.pt"):
        path = CHECKPOINT_DIR / name
        if path.exists():
            checkpoint = torch.load(path, map_location=_device)
            vocab = MoveVocabulary.load_or_build()
            model = PolicyNetwork.from_config(checkpoint["config"], vocab_size=checkpoint["vocab_size"]).to(_device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            _model, _vocab, _checkpoint_name = model, vocab, name
            return _model, _vocab, _checkpoint_name

    raise HTTPException(status_code=503, detail="No trained checkpoint found in checkpoints/. Run train/pretrain.py first.")


@app.post("/predict", response_model=MoveResponse)
def predict(request: MoveRequest) -> MoveResponse:
    model, vocab, checkpoint_name = _load_model()

    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc

    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0).cpu()

    mask = torch.from_numpy(vocab.legal_move_mask(board))
    if not mask.any():
        raise HTTPException(status_code=400, detail="No legal moves in this position (game over).")
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=0)

    k = min(request.top_k, int(mask.sum().item()))
    top_probs, top_indices = probs.topk(k)
    moves = [
        RankedMove(move=vocab.decode(idx.item(), board).uci(), confidence=prob.item())
        for prob, idx in zip(top_probs, top_indices, strict=True)
    ]
    return MoveResponse(moves=moves, model_checkpoint=checkpoint_name)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(_device)}
