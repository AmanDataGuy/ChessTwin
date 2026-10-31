"""Stage 4: replay every game and emit (position, move, metadata) rows to parquet.

Stores each row as a FEN string (the position *before* the move) plus the move
played as a UCI string, not a dense board tensor or a vocabulary index --
board_to_tensor() and MoveVocabulary.encode() are both cheap to compute at
train time from a FEN/UCI pair, and storing the raw ~19*8*8 float tensor per
row would make the processed dataset roughly 60x larger on disk for no benefit
(it also couples the dataset to one specific vocabulary ordering, which storing
UCI strings avoids).

Personal dataset: only positions where it was Aman's turn to move (matched by
username against configs/default.yaml, case-insensitive).
Pretrain dataset: every position, either player's turn (per spec Stage 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

CHUNK_SIZE = 50_000

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
PERSONAL_PGN = REPO_ROOT / "data" / "raw" / "personal" / "all_games.pgn"
PRETRAIN_DIR = REPO_ROOT / "data" / "raw" / "pretrain"
OUT_DIR = REPO_ROOT / "data" / "processed"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _personal_usernames(config: dict) -> set[str]:
    chesscom = config["data"].get("chesscom_usernames", config["data"].get("chesscom_username"))
    chesscom = [chesscom] if isinstance(chesscom, str) else list(chesscom or [])
    lichess = config["data"].get("lichess_username")
    return {u.lower() for u in (*chesscom, lichess) if u}


def _personal_result(headers: chess.pgn.Headers, aman_is_white: bool) -> str:
    result = headers.get("Result", "*")
    if result == "1-0":
        return "win" if aman_is_white else "loss"
    if result == "0-1":
        return "loss" if aman_is_white else "win"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def _parse_elo(value: str | None) -> int | None:
    if value and value.isdigit():
        return int(value)
    return None


def iter_pgn_rows(pgn_path: Path, mode: str, usernames: set[str] | None = None):
    """Yield row dicts one at a time (mode: 'personal' or 'pretrain') -- a generator,
    not a list, so memory use stays bounded regardless of dataset size."""
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        game_index = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_index += 1
            headers = game.headers

            aman_is_white = None
            if mode == "personal":
                white = headers.get("White", "").lower()
                black = headers.get("Black", "").lower()
                if white in usernames:
                    aman_is_white = True
                elif black in usernames:
                    aman_is_white = False
                else:
                    continue  # defensive: shouldn't happen given how these PGNs were fetched

            white_elo = _parse_elo(headers.get("WhiteElo"))
            black_elo = _parse_elo(headers.get("BlackElo"))
            time_control = headers.get("TimeControl", "unknown")
            game_id = f"{pgn_path.stem}_{game_index}"

            board = game.board()
            for move_number, move in enumerate(game.mainline_moves()):
                mover_is_white = board.turn == chess.WHITE
                is_target_turn = mode == "pretrain" or (mover_is_white == aman_is_white)

                if is_target_turn:
                    yield {
                        "game_id": game_id,
                        "move_number": move_number,
                        "fen": board.fen(),
                        "move_uci": move.uci(),
                        "time_control": time_control,
                        "rating": white_elo if mover_is_white else black_elo,
                        "result": _personal_result(headers, aman_is_white) if mode == "personal" else None,
                    }

                board.push(move)


def process_pgn_file(pgn_path: Path, mode: str, usernames: set[str] | None = None) -> list[dict]:
    """Small-dataset convenience wrapper: materializes iter_pgn_rows() as a list.
    Kept for tests and one-off scripts; main() below uses iter_pgn_rows() directly
    with chunked writes to avoid holding a full dataset in memory."""
    return list(iter_pgn_rows(pgn_path, mode, usernames))


def write_parquet_chunked(pgn_path: Path, mode: str, out_path: Path, usernames: set[str] | None = None) -> dict:
    """Stream rows from a PGN file straight to parquet in fixed-size chunks,
    so peak memory is bounded by CHUNK_SIZE rows, not the whole dataset."""
    writer: pq.ParquetWriter | None = None
    buffer: list[dict] = []
    total_rows = 0
    game_ids: set[str] = set()

    def flush():
        nonlocal writer, buffer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        buffer = []

    for row in iter_pgn_rows(pgn_path, mode, usernames):
        buffer.append(row)
        game_ids.add(row["game_id"])
        total_rows += 1
        if len(buffer) >= CHUNK_SIZE:
            flush()
    flush()

    if writer is not None:
        writer.close()

    return {"rows": total_rows, "games": len(game_ids)}


def main() -> None:
    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    usernames = _personal_usernames(config)
    print(f"Processing personal dataset ({PERSONAL_PGN}), Aman-usernames={usernames}...")
    personal_stats = write_parquet_chunked(PERSONAL_PGN, "personal", OUT_DIR / "personal.parquet", usernames=usernames)
    print(f"  -> {personal_stats['rows']:,} rows, {personal_stats['games']:,} games")

    pretrain_pgns = sorted(PRETRAIN_DIR.glob("*.pgn"))
    if not pretrain_pgns:
        raise FileNotFoundError(f"No pretrain PGN found in {PRETRAIN_DIR}; run fetch_pretrain_games.py first.")
    pretrain_pgn = pretrain_pgns[0]
    print(f"Processing pretrain dataset ({pretrain_pgn})...")
    pretrain_stats = write_parquet_chunked(pretrain_pgn, "pretrain", OUT_DIR / "pretrain.parquet")
    print(f"  -> {pretrain_stats['rows']:,} rows, {pretrain_stats['games']:,} games")

    summary = {
        "personal_rows": personal_stats["rows"],
        "personal_games": personal_stats["games"],
        "pretrain_rows": pretrain_stats["rows"],
        "pretrain_games": pretrain_stats["games"],
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
