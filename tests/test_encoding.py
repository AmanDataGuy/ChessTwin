import chess
import numpy as np

from models.encoding import BOARD_PLANES, MoveVocabulary, board_to_tensor


def test_board_tensor_shape_and_values_starting_position():
    board = chess.Board()
    tensor = board_to_tensor(board)

    assert tensor.shape == (BOARD_PLANES, 8, 8)
    # 32 pieces on the board at the start, one 1.0 per piece across the 12 piece planes.
    assert tensor[:12].sum() == 32
    # White to move.
    assert np.all(tensor[12] == 1.0)
    # All four castling rights available at the start.
    assert np.all(tensor[13] == 1.0)
    assert np.all(tensor[14] == 1.0)
    assert np.all(tensor[15] == 1.0)
    assert np.all(tensor[16] == 1.0)
    # No en passant target at the start.
    assert tensor[17].sum() == 0
    # Halfmove clock is 0 at the start.
    assert np.all(tensor[18] == 0.0)


def test_vocab_size_in_expected_range():
    vocab = MoveVocabulary.build()
    # Spec expects ~1900-2000 for a from/to/promotion encoding; allow some slack
    # since the exact count depends on the enumeration's edge-case handling.
    assert 1800 <= len(vocab) <= 2100


def test_round_trip_all_legal_moves_starting_position():
    vocab = MoveVocabulary.build()
    board = chess.Board()

    for move in board.legal_moves:
        index = vocab.encode(move)
        decoded = vocab.decode(index, board)
        assert decoded == move


def test_round_trip_promotion_moves():
    vocab = MoveVocabulary.build()
    # White pawn on a7, black rook on b8: a7a8=Q, a7b8=Qx are both available,
    # plus underpromotions.
    board = chess.Board("1r6/P7/8/8/8/8/8/4K2k w - - 0 1")

    for move in board.legal_moves:
        index = vocab.encode(move)
        decoded = vocab.decode(index, board)
        assert decoded == move


def test_round_trip_en_passant():
    vocab = MoveVocabulary.build()
    board = chess.Board()
    for uci in ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.push(chess.Move.from_uci(uci))
    # White pawn on e5 can capture en passant on d6.
    ep_move = chess.Move.from_uci("e5d6")
    assert ep_move in board.legal_moves

    index = vocab.encode(ep_move)
    decoded = vocab.decode(index, board)
    assert decoded == ep_move


def test_legal_move_mask_matches_legal_moves_starting_position():
    vocab = MoveVocabulary.build()
    board = chess.Board()

    mask = vocab.legal_move_mask(board)
    assert mask.sum() == 20  # 20 legal opening moves

    legal_indices = {vocab.encode(m) for m in board.legal_moves}
    masked_indices = set(np.nonzero(mask)[0].tolist())
    assert legal_indices == masked_indices


def test_legal_move_mask_excludes_illegal_moves():
    vocab = MoveVocabulary.build()
    board = chess.Board()
    mask = vocab.legal_move_mask(board)

    # e2e5 (pawn moving 3 squares) is not a legal opening move.
    illegal_move = chess.Move.from_uci("e2e5")
    assert illegal_move not in board.legal_moves
    illegal_index = vocab.encode(illegal_move)
    assert mask[illegal_index] == False
