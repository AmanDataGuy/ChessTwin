import chess
import torch
from fastapi.testclient import TestClient

import serve.api as api_module
from models.encoding import MoveVocabulary
from models.network import PolicyNetwork


def _write_fake_checkpoint(path, vocab_size: int) -> None:
    """A tiny, untrained checkpoint -- enough to exercise the API's plumbing
    without depending on a real (multi-hour) training run having finished."""
    config = {"model": {"residual_blocks": 1, "channels": 8, "board_planes": 19}}
    model = PolicyNetwork.from_config(config, vocab_size=vocab_size)
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size, "config": config}, path)


def _reset_api_state(monkeypatch, checkpoint_dir) -> None:
    monkeypatch.setattr(api_module, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(api_module, "_model", None)
    monkeypatch.setattr(api_module, "_vocab", None)
    monkeypatch.setattr(api_module, "_checkpoint_name", None)


def test_predict_returns_ranked_legal_moves(tmp_path, monkeypatch):
    vocab = MoveVocabulary.load_or_build()
    _write_fake_checkpoint(tmp_path / "finetuned.pt", len(vocab))
    _reset_api_state(monkeypatch, tmp_path)

    client = TestClient(api_module.app)
    board = chess.Board()
    resp = client.post("/predict", json={"fen": board.fen(), "top_k": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["moves"]) == 5
    legal_ucis = {m.uci() for m in board.legal_moves}
    for move in data["moves"]:
        assert move["move"] in legal_ucis
    assert data["model_checkpoint"] == "finetuned.pt"


def test_predict_rejects_invalid_fen(tmp_path, monkeypatch):
    vocab = MoveVocabulary.load_or_build()
    _write_fake_checkpoint(tmp_path / "finetuned.pt", len(vocab))
    _reset_api_state(monkeypatch, tmp_path)

    client = TestClient(api_module.app)
    resp = client.post("/predict", json={"fen": "not-a-fen", "top_k": 5})

    assert resp.status_code == 400


def test_predict_without_any_checkpoint_returns_503(tmp_path, monkeypatch):
    _reset_api_state(monkeypatch, tmp_path)  # empty dir, no checkpoint files

    client = TestClient(api_module.app)
    board = chess.Board()
    resp = client.post("/predict", json={"fen": board.fen()})

    assert resp.status_code == 503


def test_health():
    client = TestClient(api_module.app)
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
