"""Stage 11: Lichess Bot API integration -- the actual public-facing demo.

Requires a Lichess account upgraded to BOT status (one-way -- disables normal
rated play on that account) and a personal API token with the bot:play scope,
set as LICHESS_BOT_TOKEN in .env.

Move selection is model-preference-first with a Stockfish blunder veto, not
pure greedy argmax: the model's own top-1 pick is played unless a shallow
Stockfish check shows it loses significantly more material than the best of
the model's other plausible candidates, in which case the highest-probability
*non-blundering* candidate is played instead. This was added after live
testing showed pure top-1 selection hangs pieces the raw move-matching
accuracy metric doesn't penalize (matching *what* Aman usually plays doesn't
guarantee avoiding the ~50% of positions where the model doesn't guess right --
some of those misses are actual blunders, not just "a different reasonable
move"). eval/evaluate.py's ablation table is unaffected -- it measures pure
model-only move-matching, which is what the spec's headline metric is about;
this veto is a serving-time-only addition for actual play quality.

Live at lichess.org/@/ChessTwinBot. The outer event loop and each in-progress
game both reconnect on a dropped stream (Lichess's long-lived HTTP connections
drop periodically under normal operation) rather than crashing the process --
added after an unattended run died on exactly that.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import berserk
import chess
import chess.engine
import torch
import yaml

from data.fetch_personal_games import load_dotenv
from models.encoding import board_to_tensor
from serve.api import _load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
RECONNECT_DELAY_SECONDS = 5

# How many of the model's top candidate moves to safety-check, how long to spend
# evaluating each one, and how much worse (in centipawns, from the mover's own
# perspective) a candidate can be than the best-scoring one before it's vetoed
# as a blunder. 250cp catches "hangs a piece"-class blunders without demanding
# engine-perfect play, which would defeat the point of a personalized model.
SAFETY_TOP_K = 4
SAFETY_MOVETIME_SECONDS = 0.05
BLUNDER_THRESHOLD_CP = 250


def pick_move(board: chess.Board, engine: chess.engine.SimpleEngine) -> chess.Move:
    """Model's top candidates, ranked by probability, with the first
    non-blundering one played (see module docstring)."""
    model, vocab, _ = _load_model()
    device = next(model.parameters()).device
    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).squeeze(0).cpu()
    mask = torch.from_numpy(vocab.legal_move_mask(board))
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=0)

    k = min(SAFETY_TOP_K, int(mask.sum().item()))
    if k <= 1:
        return vocab.decode(int(probs.argmax().item()), board)

    top_indices = probs.topk(k).indices
    candidates = [vocab.decode(idx.item(), board) for idx in top_indices]

    scores = []
    for move in candidates:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=SAFETY_MOVETIME_SECONDS))
        # Score from the perspective of the player who just moved (us).
        score = info["score"].pov(not board.turn)
        scores.append(score.score(mate_score=100_000))
        board.pop()

    best_cp = max(s for s in scores if s is not None)
    for move, cp in zip(candidates, scores, strict=True):
        if cp is not None and best_cp - cp <= BLUNDER_THRESHOLD_CP:
            return move
    return candidates[0]  # defensive fallback; the loop above always returns for a well-formed position


def _board_from_moves(moves_str: str) -> chess.Board:
    board = chess.Board()
    for uci in moves_str.split():
        board.push(chess.Move.from_uci(uci))
    return board


def play_game(client: berserk.Client, game_id: str, engine: chess.engine.SimpleEngine) -> None:
    """Play one game. A dropped stream mid-game (Lichess's long-lived HTTP
    connections do drop periodically -- this is normal, not an error condition)
    ends this function, not the whole bot process; the outer loop in main()
    keeps running and can still accept new challenges."""
    try:
        my_color = None
        for event in client.bots.stream_game_state(game_id):
            if event["type"] == "gameFull":
                my_id = client.account.get()["id"]
                my_color = "white" if event["white"].get("id") == my_id else "black"
                board = _board_from_moves(event["state"]["moves"])
            elif event["type"] == "gameState":
                board = _board_from_moves(event["moves"])
            else:
                continue

            if board.is_game_over():
                break

            is_my_turn = (board.turn == chess.WHITE) == (my_color == "white")
            if is_my_turn:
                move = pick_move(board, engine)
                client.bots.make_move(game_id, move.uci())
    except Exception:  # noqa: BLE001 -- daemon loop: log and keep the bot alive, never crash on a dropped stream
        print(f"Game {game_id} stream dropped:")
        traceback.print_exc()


def main() -> None:
    load_dotenv()
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set LICHESS_BOT_TOKEN in .env (a Lichess personal token with the bot:play scope).")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    engine_path = config["engine"]["stockfish_path"]

    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    print("Bot online, waiting for challenges...")
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        while True:
            try:
                for event in client.bots.stream_incoming_events():
                    if event["type"] == "challenge":
                        client.bots.accept_challenge(event["challenge"]["id"])
                    elif event["type"] == "gameStart":
                        play_game(client, event["game"]["id"], engine)
            except Exception:  # noqa: BLE001 -- daemon loop: log and reconnect, never crash on a dropped stream
                print("Incoming-events stream dropped, reconnecting:")
                traceback.print_exc()
                time.sleep(RECONNECT_DELAY_SECONDS)


if __name__ == "__main__":
    main()
