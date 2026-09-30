"""Stage 3: board tensor encoding, move vocabulary, and legal-move masking.

Board tensor: 19 planes of 8x8 -- 12 piece/color planes, 1 side-to-move plane,
4 castling-rights planes, 1 en-passant-target plane, 1 halfmove-clock plane
(normalized). Board orientation is absolute (not flipped for black to move),
matching the spec's literal plane list rather than AlphaZero-style canonicalization.

Move vocabulary: a fixed index over every (from_square, to_square) reachable by
a queen-or-knight movement pattern from any square (this superset covers every
piece's actual moves, since a rook/bishop/queen/king/pawn move is always a
special case of a queen-pattern move, and a knight move is a knight-pattern
move), plus 3 underpromotion variants (knight/bishop/rook) for every pawn move
landing on the back rank. A promotion move with no suffix is treated as an
implicit queen promotion, matching Leela/Maia-style compact move encodings.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = REPO_ROOT / "models" / "move_vocab.json"

# Plane layout.
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
BOARD_PLANES = 19
PLANE_SIDE_TO_MOVE = 12
PLANE_CASTLING_WK = 13
PLANE_CASTLING_WQ = 14
PLANE_CASTLING_BK = 15
PLANE_CASTLING_BQ = 16
PLANE_EN_PASSANT = 17
PLANE_HALFMOVE_CLOCK = 18

QUEEN_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_DELTAS = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
UNDERPROMOTION_PIECES = ["n", "b", "r"]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a board position as a (BOARD_PLANES, 8, 8) float32 tensor."""
    tensor = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = PIECE_TYPES.index(piece.piece_type) + color_offset
        tensor[plane, rank, file] = 1.0

    tensor[PLANE_SIDE_TO_MOVE, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[PLANE_CASTLING_WK, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[PLANE_CASTLING_WQ, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[PLANE_CASTLING_BK, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[PLANE_CASTLING_BQ, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        tensor[PLANE_EN_PASSANT, rank, file] = 1.0

    tensor[PLANE_HALFMOVE_CLOCK, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    return tensor


def _generate_vocab() -> list[str]:
    """Enumerate the fixed move vocabulary as sorted UCI strings (no promotion
    suffix for queen promotions, by convention)."""
    moves: set[str] = set()

    for from_sq in chess.SQUARES:
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)

        for delta_rank, delta_file in QUEEN_DIRECTIONS:
            for dist in range(1, 8):
                to_rank, to_file = from_rank + delta_rank * dist, from_file + delta_file * dist
                if not (0 <= to_rank < 8 and 0 <= to_file < 8):
                    break
                to_sq = chess.square(to_file, to_rank)
                moves.add(chess.Move(from_sq, to_sq).uci())

        for delta_rank, delta_file in KNIGHT_DELTAS:
            to_rank, to_file = from_rank + delta_rank, from_file + delta_file
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = chess.square(to_file, to_rank)
                moves.add(chess.Move(from_sq, to_sq).uci())

    # Underpromotions: genuine pawn-shaped moves only (one square forward or
    # diagonal, landing on the back rank) -- not every queen/knight move that
    # happens to land on rank 1 or 8.
    for from_file in range(8):
        for delta_file in (-1, 0, 1):
            to_file = from_file + delta_file
            if not (0 <= to_file < 8):
                continue
            # White: rank 7 (index 6) -> rank 8 (index 7).
            white_uci = chess.Move(chess.square(from_file, 6), chess.square(to_file, 7)).uci()
            # Black: rank 2 (index 1) -> rank 1 (index 0).
            black_uci = chess.Move(chess.square(from_file, 1), chess.square(to_file, 0)).uci()
            for base_uci in (white_uci, black_uci):
                for piece in UNDERPROMOTION_PIECES:
                    moves.add(base_uci + piece)

    return sorted(moves)


class MoveVocabulary:
    def __init__(self, entries: list[str]):
        self.entries = entries
        self.index_of = {uci: i for i, uci in enumerate(entries)}

    @classmethod
    def build(cls) -> MoveVocabulary:
        return cls(_generate_vocab())

    @classmethod
    def load_or_build(cls, path: Path = VOCAB_PATH) -> MoveVocabulary:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return cls(json.load(f))
        vocab = cls.build()
        vocab.save(path)
        return vocab

    def save(self, path: Path = VOCAB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _canonical_uci(move: chess.Move) -> str:
        """Queen promotions are stored without a promotion suffix; everything else keeps its UCI form."""
        uci = move.uci()
        if move.promotion == chess.QUEEN:
            return uci[:-1]
        return uci

    def encode(self, move: chess.Move) -> int:
        return self.index_of[self._canonical_uci(move)]

    def decode(self, index: int, board: chess.Board) -> chess.Move:
        """Reconstruct a chess.Move from a vocab index, given the board it applies to
        (needed to resolve the implicit-queen-promotion convention)."""
        uci = self.entries[index]
        from_sq = chess.parse_square(uci[0:2])
        to_sq = chess.parse_square(uci[2:4])
        promotion = None
        if len(uci) == 5:
            promotion = chess.Piece.from_symbol(uci[4]).piece_type
        else:
            piece = board.piece_at(from_sq)
            to_rank = chess.square_rank(to_sq)
            if piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
                promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promotion)

    def legal_move_mask(self, board: chess.Board) -> np.ndarray:
        """Boolean mask over the vocabulary: True at indices matching a currently legal move."""
        mask = np.zeros(len(self.entries), dtype=bool)
        for move in board.legal_moves:
            mask[self.encode(move)] = True
        return mask
