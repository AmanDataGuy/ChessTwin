import chess
import torch

from models.encoding import BOARD_PLANES, MoveVocabulary, board_to_tensor
from models.network import PolicyNetwork


def test_forward_pass_output_shape_random_batch():
    vocab = MoveVocabulary.build()
    model = PolicyNetwork(vocab_size=len(vocab))

    batch = torch.randn(4, BOARD_PLANES, 8, 8)
    logits = model(batch)

    assert logits.shape == (4, len(vocab))
    assert torch.isfinite(logits).all()


def test_forward_pass_real_position():
    vocab = MoveVocabulary.build()
    model = PolicyNetwork(vocab_size=len(vocab))
    model.eval()

    board = chess.Board()
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)

    assert logits.shape == (1, len(vocab))
    assert torch.isfinite(logits).all()


def test_from_config_matches_yaml_dimensions():
    config = {
        "model": {"residual_blocks": 3, "channels": 16, "board_planes": BOARD_PLANES},
    }
    vocab_size = 100
    model = PolicyNetwork.from_config(config, vocab_size)

    assert len(model.blocks) == 3
    assert model.stem_conv.out_channels == 16
    assert model.stem_conv.in_channels == BOARD_PLANES
    assert model.policy_fc.out_features == vocab_size


def test_legal_move_masking_integration():
    """End-to-end: model logits + Stage 3 legal-move mask -> valid probability distribution over legal moves only."""
    vocab = MoveVocabulary.build()
    model = PolicyNetwork(vocab_size=len(vocab))
    model.eval()

    board = chess.Board()
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor).squeeze(0)

    mask = torch.from_numpy(vocab.legal_move_mask(board))
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked_logits, dim=0)

    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert probs[~mask].sum().item() < 1e-6  # no probability mass on illegal moves
