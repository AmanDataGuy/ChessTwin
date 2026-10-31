"""Actively challenges other online Lichess bots to rated bullet games.

serve/bot.py only reacts to incoming challenges -- a real player has to find
and challenge it for it to ever play a game. This script instead issues
outgoing rated-bullet challenges to other currently-online bot accounts
(fetched from the public /api/bot/online list) on a timer, so the account
accumulates real rated games and a real rating. Runs as a separate process
from serve/bot.py -- it only makes HTTP calls (no torch/CUDA), so it's safe to
run alongside the bot process; once a challenge is accepted, Lichess sends the
usual gameStart event and the already-running bot.py plays it exactly like any
other game.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import requests

from serve.bot import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADERS_UA = {"User-Agent": "ChessPersona research project (contact: ankit.work2026@gmail.com)"}
CHALLENGE_INTERVAL_SECONDS = 180
BOT_USERNAME = "ChessTwinBot"


def get_online_bots(exclude: str) -> list[str]:
    resp = requests.get("https://lichess.org/api/bot/online", params={"nb": 50}, headers=HEADERS_UA, timeout=30)
    resp.raise_for_status()
    bots = []
    for line in resp.text.strip().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if data["username"].lower() != exclude.lower():
            bots.append(data["username"])
    return bots


def challenge_bot(token: str, opponent: str) -> tuple[int, str]:
    resp = requests.post(
        f"https://lichess.org/api/challenge/{opponent}",
        headers={**HEADERS_UA, "Authorization": f"Bearer {token}"},
        data={"rated": "true", "clock.limit": 60, "clock.increment": 0, "color": "random", "variant": "standard"},
        timeout=30,
    )
    return resp.status_code, resp.text


def main() -> None:
    load_dotenv()
    token = os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set LICHESS_BOT_TOKEN in .env (a Lichess personal token with the bot:play scope).")

    print("Matchmaking loop started -- challenging other online bots to rated 1+0 bullet games.")
    while True:
        try:
            bots = get_online_bots(exclude=BOT_USERNAME)
            if bots:
                opponent = random.choice(bots)
                status, body = challenge_bot(token, opponent)
                print(f"Challenged {opponent}: HTTP {status} {body[:200]}")
            else:
                print("No other bots currently online.")
        except Exception as exc:  # noqa: BLE001 -- daemon loop: log and keep retrying, one bad request shouldn't kill it
            print(f"Matchmaking error: {exc}")
        time.sleep(CHALLENGE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
