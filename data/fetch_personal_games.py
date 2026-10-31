"""Stage 1: pull Aman's personal games from Chess.com and/or Lichess.

Reads usernames from CHESSCOM_USERNAME / LICHESS_USERNAME env vars, falling back
to configs/default.yaml. Writes raw PGNs to data/raw/personal/ and logs a summary
(game counts by time control and color) to data/raw/personal/summary.json.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import chess.pgn
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
OUT_DIR = REPO_ROOT / "data" / "raw" / "personal"

# Chess.com requires a descriptive User-Agent or it returns 403.
CHESSCOM_HEADERS = {"User-Agent": "ChessPersona research project (contact: ankit.work2026@gmail.com)"}

# Lichess's edge WAF silently serves the SPA's HTML 404 page to requests without a
# browser-like User-Agent, instead of routing them to the API (discovered by testing
# live against api/games/user/{username} — a bare-curl request 404s, adding a UA fixes it).
LICHESS_HEADERS = {
    "User-Agent": "ChessPersona research project (contact: ankit.work2026@gmail.com)",
    "Accept": "application/x-chess-pgn",
}


def load_dotenv() -> None:
    """Populate os.environ from a local .env file, if present, without overriding real env vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_chesscom_games(username: str) -> str:
    """Return all of a Chess.com user's games concatenated as one PGN blob."""
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    resp = requests.get(archives_url, headers=CHESSCOM_HEADERS, timeout=30)
    resp.raise_for_status()
    archive_urls = resp.json()["archives"]

    pgns = []
    for url in archive_urls:
        r = requests.get(url, headers=CHESSCOM_HEADERS, timeout=30)
        r.raise_for_status()
        for game in r.json().get("games", []):
            pgn = game.get("pgn")
            if pgn:
                pgns.append(pgn)
    return "\n\n".join(pgns)


def fetch_lichess_games(username: str) -> str:
    """Return all of a Lichess user's games as one PGN blob (with clocks/evals/opening)."""
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "rated": "true",
        "evals": "true",
        "clocks": "true",
        "opening": "true",
        "format": "pgn",
    }
    headers = dict(LICHESS_HEADERS)
    token = os.environ.get("LICHESS_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(5):
        resp = requests.get(url, params=params, headers=headers, timeout=300, stream=True)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"  Lichess rate-limited (429), waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.text
    resp.raise_for_status()
    return resp.text


def summarize(pgn_text: str, usernames: list[str]) -> dict:
    """Log total games, date range, time-control breakdown, and color breakdown."""
    usernames_lower = {u.lower() for u in usernames if u}
    game_count = 0
    dates = []
    time_controls: Counter[str] = Counter()
    colors: Counter[str] = Counter()

    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        game_count += 1
        headers = game.headers
        date = headers.get("UTCDate") or headers.get("Date")
        if date and date != "????.??.??":
            dates.append(date)
        time_controls[headers.get("TimeControl", "unknown")] += 1

        white = headers.get("White", "").lower()
        black = headers.get("Black", "").lower()
        if white in usernames_lower:
            colors["white"] += 1
        elif black in usernames_lower:
            colors["black"] += 1
        else:
            colors["unknown"] += 1

    return {
        "total_games": game_count,
        "date_range": {"earliest": min(dates) if dates else None, "latest": max(dates) if dates else None},
        "time_control_breakdown": dict(time_controls),
        "color_breakdown": dict(colors),
    }


def main() -> None:
    load_dotenv()
    config = load_config()

    env_chesscom = os.environ.get("CHESSCOM_USERNAME")
    if env_chesscom:
        chesscom_usernames = [u.strip() for u in env_chesscom.split(",") if u.strip()]
    else:
        configured = config["data"].get("chesscom_usernames", config["data"].get("chesscom_username"))
        chesscom_usernames = [configured] if isinstance(configured, str) else list(configured or [])

    lichess_username = os.environ.get("LICHESS_USERNAME") or config["data"]["lichess_username"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pgn_parts = []
    per_platform_summaries = {}
    all_usernames = list(chesscom_usernames) + [lichess_username]

    if chesscom_usernames:
        chesscom_pgn_parts = []
        for username in chesscom_usernames:
            print(f"Fetching Chess.com games for '{username}'...")
            pgn = fetch_chesscom_games(username)
            chesscom_pgn_parts.append(pgn)
            print(f"  -> {summarize(pgn, [username])['total_games']} games")
        chesscom_pgn = "\n\n".join(chesscom_pgn_parts)
        (OUT_DIR / "chesscom_games.pgn").write_text(chesscom_pgn, encoding="utf-8")
        all_pgn_parts.append(chesscom_pgn)
        per_platform_summaries["chesscom"] = summarize(chesscom_pgn, chesscom_usernames)

    if lichess_username:
        print(f"Fetching Lichess games for '{lichess_username}'...")
        lichess_pgn = fetch_lichess_games(lichess_username)
        (OUT_DIR / "lichess_games.pgn").write_text(lichess_pgn, encoding="utf-8")
        all_pgn_parts.append(lichess_pgn)
        per_platform_summaries["lichess"] = summarize(lichess_pgn, [lichess_username])
        print(f"  -> {per_platform_summaries['lichess']['total_games']} games")

    if not all_pgn_parts:
        print("No usernames configured (CHESSCOM_USERNAME / LICHESS_USERNAME / configs/default.yaml). Nothing to do.")
        sys.exit(1)

    merged_pgn = "\n\n".join(all_pgn_parts)
    (OUT_DIR / "all_games.pgn").write_text(merged_pgn, encoding="utf-8")

    merged_summary = summarize(merged_pgn, all_usernames)
    summary = {"per_platform": per_platform_summaries, "merged": merged_summary}

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
