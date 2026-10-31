"""Stage 2: build the pretraining set by streaming a Lichess monthly database dump
and filtering to games matching Aman's time control and rating band.

Streams and decompresses the (tens-of-GB) .pgn.zst dump on the fly rather than
downloading it whole, and stops as soon as enough matching games are collected.
Filters at the game level via a lightweight line-based header scan (not a full
chess.pgn parse, which is ~80 games/sec and far too slow to scan millions of
games — this is a pure text scan that only inspects PGN tag lines and copies
matching raw game blocks through untouched). Filter: Event contains the target
time-control class (e.g. "Bullet"), and both players' Elo fall within
configs/default.yaml's rating_band. Writes matched games to data/raw/pretrain/
and logs the exact thresholds used and the resulting dataset size.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import requests
import yaml
import zstandard

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
OUT_DIR = REPO_ROOT / "data" / "raw" / "pretrain"

HEADERS = {"User-Agent": "ChessPersona research project (contact: ankit.work2026@gmail.com)"}
TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def headers_match(headers: dict, time_control_class: str, rating_band: tuple[int, int]) -> bool:
    event = headers.get("Event", "")
    if time_control_class.lower() not in event.lower():
        return False
    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except ValueError:
        return False
    lo, hi = rating_band
    return lo <= white_elo <= hi and lo <= black_elo <= hi


def iter_games(text_stream):
    """Yield (headers, raw_game_text) for every game in a PGN text stream.

    Line-based state machine: a PGN dump is a sequence of games, each a block of
    '[Tag "value"]' header lines, a blank line, movetext lines, and a trailing
    blank line separating it from the next game's headers.
    """
    headers: dict = {}
    game_lines: list[str] = []
    in_movetext = False

    for line in text_stream:
        if line.startswith("["):
            m = TAG_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
            game_lines.append(line)
        elif line.strip() == "":
            game_lines.append(line)
            if in_movetext:
                # Blank line after movetext: end of this game.
                yield headers, "".join(game_lines)
                headers = {}
                game_lines = []
                in_movetext = False
            else:
                # Blank line after headers: movetext starts next.
                in_movetext = True
        else:
            game_lines.append(line)


def main() -> None:
    config = load_config()
    rating_band = tuple(config["data"]["rating_band"])
    time_control_class = "Bullet" if config["data"]["main_time_control"] == "bullet" else config["data"]["main_time_control"]
    month = config["pretrain"]["database_month"]
    target_games = config["pretrain"]["target_games"]

    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
    print(f"Streaming {url}")
    print(f"Filter: Event contains '{time_control_class}', both players' Elo in {rating_band}")
    print(f"Target: {target_games} matching games")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pretrain_{month}_{time_control_class.lower()}_{rating_band[0]}-{rating_band[1]}.pgn"

    resp = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    resp.raise_for_status()

    decompressor = zstandard.ZstdDecompressor(max_window_size=2**31)
    stream_reader = decompressor.stream_reader(resp.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")

    matched = 0
    scanned = 0
    start = time.time()

    with open(out_path, "w", encoding="utf-8") as out_f:
        for headers, raw_game_text in iter_games(text_stream):
            scanned += 1
            if headers_match(headers, time_control_class, rating_band):
                out_f.write(raw_game_text)
                out_f.write("\n")
                matched += 1

            if scanned % 50_000 == 0:
                elapsed = time.time() - start
                rate = scanned / elapsed if elapsed > 0 else 0
                print(f"  scanned {scanned:,}, matched {matched:,} ({elapsed:.0f}s elapsed, {rate:.0f} scanned/s)")

            if matched >= target_games:
                break

    resp.close()

    summary = {
        "month": month,
        "filter": {"event_contains": time_control_class, "rating_band": list(rating_band)},
        "games_scanned": scanned,
        "games_matched": matched,
        "output_file": str(out_path.relative_to(REPO_ROOT)),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
