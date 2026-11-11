# ChessPersona

A model that predicts the move *Aman* would play in a given chess position. It's behavioral cloning on personal chess games: pretrained on general human play, then fine-tuned on Aman's own games. This is the same pattern as Maia Chess, applied at the individual level.

It plays live as **[ChessTwinBot](https://lichess.org/@/ChessTwinBot)** on Lichess.

[`technical_report.md`](technical_report.md) is the full record: architecture, every real number produced, and a log of what was tried, what failed, and why.

## Status (2026-11-12): finished, packed up

| Stage | What | Status |
|---|---|---|
| 0 | Scaffolding, CI, venv | Done. GitHub Actions runs `ruff check` + `pytest` on every push/PR |
| 1-2 | Personal + pretrain data fetch | Done |
| 3 | Board/move encoding | Done, 7/7 tests |
| 4-5 | PGN → parquet, game-level split | Done, 5/5 split tests |
| 6 | Model architecture | Done, 4/4 tests |
| 7 | Baselines (random, Stockfish) | Done |
| 8-9 | Pretrain + fine-tune | Done. 5 training rounds; Run 3 is deployed |
| 10 | Evaluation / ablation | Done. Re-run after every retrain |
| 11 | Serving | FastAPI `/predict` + live Lichess bot + auto-matchmaking against other bots |
| 12 | Tests / docs | **20/20 tests passing, ruff clean** |

Further blind training experiments have stopped: Runs 4 and 5 didn't beat Run 3 (see Training history below). Planned but not built: blunder-rate fidelity and calibration analysis.

## Result

The core hypothesis holds. Fine-tuning a pretrained general-human model on Aman's games predicts his moves better than a raw engine or the generic model alone. The test set is held-out games of his that were never seen in training. The numbers below are for the deployed checkpoint (Run 3):

| Model | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| Random legal move | 7.08% | — |
| Stockfish (always plays the objectively best move) | 37.68% | — |
| Pretrained-only (general human bullet play) | 41.84% | 70.38% |
| **Fine-tuned (personalized to Aman)** | **47.68%** | **75.42%** |

- Fine-tuning adds **+5.84 pp** top-1 over the pretrained-only model.
- Both learned models beat "always play Stockfish's best move." Even the generic model predicts Aman better than the objectively best move does.
- Evaluated on a 5,000-row sample of the 133,956-row personal test split. `python -m eval.evaluate` writes the numbers to `data/processed/ablation_results.json` (a copy is in the data repo, see below).

## Training history

| Run | Pretrain | Fine-tune | Fine-tuned top-1 / top-3 (test) | French* | Catalan* | Outcome |
|---|---|---|---|---|---|---|
| 1 | 1M pos, 1 ep | 400k pos, 2 ep | 39.88% / 65.68% | 10.3% | 8.4% | Capped one-day run, superseded |
| 2 | full, 4 ep | full, 6 ep | 47.72% / 75.26% | 17.7% | 17.8% | Superseded |
| **3** | **full, 8 ep** | **full, 12 ep, 3x early-move oversample, lr 1e-4** | **47.68% / 75.42%** | **8.3%** | **24.6%** | **Deployed** |
| 4 | (Run 3) | 5 ep, 2x oversample, lr 2e-4 | val 48.1% / 75.1% | 10.4% | 15.6% | Reverted: Catalan regressed |
| 5 | (Run 3) | +4 ep on openings only (moves < 10), lr 5e-5 | 45.86% / — | 12.6% | 16.8% | Reverted: −1.8 pp overall, Catalan regressed |

\* The model's probability of playing Aman's repertoire move: 1…e6 after 1.e4 (French), and 2.g3 after 1.d4 Nf6 2.c4 e6 (Catalan). Run 1's French/Catalan figures are the pre-Run-2 baselines.

Takeaway: once the model was trained at full scale, hyperparameter changes moved aggregate accuracy by under 1 pp but swung specific opening probabilities 2-3x in uncorrelated directions. Improving further likely needs more personal data or a different objective (e.g. weighted loss on repertoire positions), not more tuning.

## Architecture

- **Board encoding**: 19-plane 8x8 tensor (12 piece/color planes, side-to-move, 4 castling-rights planes, en passant, halfmove clock).
- **Move vocabulary**: 1,924 entries, built by enumerating queen/knight movement patterns from every square, plus pawn-shaped underpromotions. Not hardcoded.
- **Model**: a 6-residual-block, 64-channel ResNet (Maia Chess's proven architecture) with an AlphaZero-style policy head over the move vocabulary. The caller applies legal-move masking.
- **Training**: supervised cross-entropy (behavioral cloning, not RL). Pretrain on general Lichess bullet games in Aman's rating band, then fine-tune on his own games at a lower learning rate. Both stages auto-resume from `*_resume.pt` if interrupted.
- **Serving-time move selection** (bot only, not part of the measured model): the bot takes the model's top-4 candidates and Stockfish checks each one for 50 ms. It plays the most probable candidate that isn't more than 250 cp worse than the best of the four.

## Data

| Dataset | Games | Positions | Train / val / test (games) | Source |
|---|---|---|---|---|
| Personal (fine-tune) | 34,546 | 1,331,760 | 27,636 / 3,454 / 3,456 | Chess.com `stronkkkest`, `desichessss`, `due4awinn` + Lichess `reacher2024`, 2022-04 → 2026-09 |
| Pretrain | 69,993 | 5,219,708 | 55,994 / 6,999 / 7,000 | Lichess 2026-07 dump, Bullet, Elo 2150-2650 (centered on Aman's 2409 Lichess bullet rating) |

Splits are 80/10/10 at the game level, never the position level. Positions within a game are correlated, so a position-level split would leak and inflate accuracy.

## Serving

- `serve/api.py`: FastAPI `POST /predict` (FEN → ranked moves + confidence, pure model output) and `GET /health`.
- `serve/bot.py`: the Lichess bot. It accepts challenges, plays with the blunder veto above, and reconnects on dropped streams.
- `serve/matchmaking.py`: challenges other online Lichess bots to rated bullet every 3 minutes, so the account builds a real rating. It makes HTTP calls only (no GPU), so it's safe to run alongside the bot.

## Limitations

- **Personalization ceiling.** At ~48% top-1, the model still doesn't play Aman's move more than half the time. The French and the Catalan are still not its top prediction in their key positions (see the table above).
- **The blunder veto adds strength but doesn't target Aman's ~2400 level.** It only blocks moves that are much worse than another top-4 candidate. It doesn't upgrade a mediocre move. This was a deliberate choice to stay faithful to Aman's style.
- **Single-process data loading.** On the original Windows dev machine, spawn-based DataLoader workers duplicated the dataframe per worker and ran out of memory, so all training uses `num_workers=0`.
- **Don't run the bot and training on the GPU at the same time.** Doing so caused repeated CUDA init crashes on this laptop (RTX 3050). Take the bot offline while training.
- **Evaluation uses a 5,000-row sample.** The Stockfish baseline at 50 ms/move over all 133,956 test rows would take ~1.6 h. Exact percentages would shift slightly on the full set.

## Where everything lives

| What | Where |
|---|---|
| Code, config, docs, move vocabulary | this GitHub repo |
| Checkpoints (`pretrained.pt`, `finetuned.pt` + Run 4/5 experiments) | [huggingface.co/amanDS5153/ChessTwin](https://huggingface.co/amanDS5153/ChessTwin) (public) |
| Raw PGNs, processed parquet, splits, eval outputs, original project spec | `amanDS5153/ChessTwin-data` HF dataset (private) |
| Stockfish | not stored: install it yourself |
| `.env` (Lichess tokens) | local only, never committed |

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
hf download amanDS5153/ChessTwin --local-dir checkpoints --include "*.pt"
```

`pip install -r requirements.txt` pulls whatever torch wheel your platform defaults to. For a specific CUDA build, first run `pip install torch --index-url https://download.pytorch.org/whl/cu128` (pick the index that matches your driver).

Stockfish: install it so `stockfish` is on your PATH (`apt install stockfish`, `brew install stockfish`, or the [download](https://stockfishchess.org/download/)), or set `engine.stockfish_path` in `configs/default.yaml` to the binary's path. Stages 7 and 10 and the bot use it. The bot needs `LICHESS_BOT_TOKEN` in `.env`. Fetching personal Lichess games uses `LICHESS_API_TOKEN`.

To skip Stages 1-5 and use the exact data behind these results (needs access to the private data repo):

```
hf download amanDS5153/ChessTwin-data --repo-type dataset --local-dir data/hf
mkdir -p data/processed && cp data/hf/processed/* data/processed/
```

## Reproducing the pipeline

```
python data/fetch_personal_games.py       # Stage 1: pull your own games (set usernames in configs/default.yaml)
python data/fetch_pretrain_games.py       # Stage 2: filter a Lichess database dump to your time control/rating band
python data/process_pgn.py                # Stage 4: PGN -> (FEN, move, metadata) parquet
python data/splits.py                     # Stage 5: game-level 80/10/10 split
python models/baselines.py                # Stage 7: random + Stockfish baselines
python -m train.pretrain                  # Stage 8: pretrain on general data
python -m train.finetune                  # Stage 9: fine-tune on your games
python -m eval.evaluate                   # Stage 10: the headline ablation table
uvicorn serve.api:app --reload            # Stage 11: FastAPI serving layer
python -m serve.bot                       # Stage 11: Lichess bot (stop before training)
python -m serve.matchmaking               # Stage 11: optional, auto-challenge other bots
```

`configs/default.yaml` holds the Run 3 settings, which produced the deployed checkpoint.

## Tests

```
pytest
```

20 tests: board/move encoding round-trips, legal-move masking, train/val/test split invariants, model forward passes, and the serving API. CI runs these plus `ruff check .` on every push.
