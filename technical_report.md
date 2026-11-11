# ChessPersona — Technical Report

Single, continuously-maintained record of the project: architecture, every real number produced, everything tried, what failed and why, and current status. Supersedes and consolidates the former `plan_ahead.md` (forward-looking plan) and `progress_report.md` (backward-looking log), which grew heavily redundant with each other as the project moved from planning into iterative training/deployment. This file is updated after every meaningful run, fix, or decision — do not let it go stale.

Last updated: 2026-11-12.

---

## 1. What this is

A model that predicts the move a specific person (Aman) would play in a given chess position — behavioral cloning, not RL (see `chess_persona_project_spec.md` §2 for why). Pretrained on general human bullet games in Aman's rating band, then fine-tuned on his own games. Deployed as a Lichess bot (`ChessTwinBot`) rather than a custom web app, following the same deployment pattern as Maia Chess's own bot accounts.

**Precedent this is built on** (established before writing any code):
- **Maia Chess** (McIlroy-Young et al., KDD 2020) — the architecture family (residual CNN, board-tensor encoding, human-move-prediction objective) is copied from here, not invented.
- **"Learning Models of Individual Behavior in Chess"** (McIlroy-Young et al., KDD 2022) — the closest direct precedent: fine-tunes a Maia base model on one player's games, validating this project's exact hypothesis (personalization beats both a raw engine and the population model at predicting that one player).
- Maia-2 (NeurIPS 2024) and "Generative Modeling of Individual Behavior at Scale" (2025) informed stretch-goal framing, not the baseline design.

## 2. Architecture

- **Board encoding**: 19-plane 8x8 tensor — 12 piece/color planes, 1 side-to-move plane, 4 castling-rights planes, 1 en-passant plane, 1 halfmove-clock plane. Absolute orientation (not flipped for Black), per the spec's literal plane list.
- **Move vocabulary**: 1,924 entries, built programmatically (not hardcoded) by enumerating queen-pattern and knight-pattern (from,to) pairs from every square, plus genuine pawn-shaped underpromotions. Queen promotions stored without a suffix (Leela/Maia convention).
- **Model**: 6-residual-block, 64-channel ResNet (`models/network.py`), stem conv+BN+ReLU, AlphaZero-style policy head (1x1 conv → 32 channels → flatten → linear to vocab size). Legal-move masking applied by the caller, not inside the network.
- **Training objective**: supervised cross-entropy (behavioral cloning). No RL, no search inside the network.
- **Serving-time addition** (not part of the trained model): a Stockfish-backed blunder veto — see §7.

## 3. Data

| Dataset | Games | Positions | Source |
|---|---|---|---|
| Personal (fine-tune) | 34,546 | 1,331,760 | Chess.com `stronkkkest` (4,926), `desichessss` (16,410), `due4awinn` (3,322) + Lichess `reacher2024` (9,990) |
| Pretrain | 69,993 | 5,219,708 | Lichess 2026-07 database dump, filtered to Bullet + Elo [2150, 2650] (centered on Aman's 2409 Lichess bullet rating) |

Split: game-level 80/10/10 (never position-level — positions within a game are correlated, so a position-level split would leak and inflate reported accuracy). Personal: 27,636/3,454/3,456 games → 1,062,664/135,140/133,956 rows. Pretrain: 55,994/6,999/7,000 games → 4,173,352/516,965/529,391 rows.

## 4. Pipeline stage status

| Stage | What | Status |
|---|---|---|
| 0 | Scaffolding, CI, venv | Done |
| 1 | Personal data fetch | Done |
| 2 | Pretrain data fetch/filter | Done |
| 3 | Board/move encoding | Done, 7/7 tests |
| 4 | Dataset → parquet | Done |
| 5 | Train/val/test split | Done, 5/5 tests |
| 6 | Model architecture | Done, 4/4 tests |
| 7 | Baselines (random, Stockfish) | Done |
| 8 | Pretraining | Done (3 full runs — see §6) |
| 9 | Fine-tuning | Done (3 full runs — see §6) |
| 10 | Evaluation/ablation | Done — re-run after every retrain |
| 11 | Serving (API + Lichess bot) | Live at lichess.org/@/ChessTwinBot, running Run 3 checkpoint |
| 12 | Tests/docs/polish | Done — 20/20 tests passing, lint clean |

Full test suite: `tests/test_{encoding,splits,model_forward,api}.py`, 20 tests.

## 5. Build-phase issues and fixes (Stages 0-7)

- **Lichess 404 on a real endpoint**: edge WAF silently served the SPA's HTML 404 page to requests without a browser-like `User-Agent`. Fixed by adding one.
- **Persistent Lichess 429 that outlasted backoff**: looked IP-level, not the documented per-second throttle. Fixed with an authenticated personal token (`LICHESS_API_TOKEN`) — no OAuth scope needed since the endpoint only requires *any* authenticated token for the higher tier.
- **Stage 2 filtering was ~200x too slow** (~80 games/sec using full `chess.pgn` parsing per candidate against a ~5% match rate, i.e. hours to scan enough games). Fixed with a hand-written line-based header-only scanner (no PGN parsing) — 2,150 games/sec, 27x speedup. Full parsing deferred to Stage 4, which only ever runs on the already-filtered set.
- **Move vocabulary bug**: first build produced 2,956 entries (spec expects ~1900-2000). Root cause: underpromotion generation was applied to *any* move landing on the back rank (including impossible ones like a rook sliding `a1a8`), not just genuine pawn-shaped moves. Caught by a unit test asserting the vocab size range. Fixed → 1,924 entries.
- **Stale config value**: `configs/default.yaml` said `board_planes: 17` (original Maia-scheme assumption) while the actual encoding used 19 planes (added en passant + halfmove clock). Caught while wiring up `PolicyNetwork.from_config()`, before it could cause a silent shape mismatch at training time.
- **`due4awinn` account discovered mid-project**, folded into the personal dataset (pushed total from 30,861 → 34,648 games). Reprocessing the larger dataset with the original list-then-write approach hit a `MemoryError` (this machine frequently runs under 4GB free RAM out of 24.7GB total). Fixed by rewriting `process_pgn.py` to stream rows to parquet in 50,000-row chunks via `pyarrow.parquet.ParquetWriter` instead of building one giant in-memory list.

## 6. Training runs (chronological, all real numbers)

### Windows DataLoader multiprocessing is fundamentally broken on this machine

`num_workers>0` reliably OOMs: Windows' `spawn`-based multiprocessing re-pickles the entire loaded dataframe to every worker process at startup, and this machine's available RAM can't absorb that duplication. All training runs use `num_workers=0` (single-process CPU-bound FEN→tensor encoding). This makes epochs slower but is the only configuration that doesn't crash.

### Run 1 — capped, time-budget-limited (finish same day)

- Pretrain: 1,000,000 of 4.17M positions, 1 epoch → val top-1 **30.9%**, top-3 **54.6%**.
- Fine-tune: 400,000 of 1.06M positions, 2 epochs → val top-1 **38.6%**, top-3 **62.3%**.
- Ablation (5,000-row personal test sample):

  | Model | Top-1 | Top-3 |
  |---|---|---|
  | Random | 7.08% | — |
  | Stockfish-best-move | 37.96% | — |
  | Pretrained-only | 33.10% | 58.46% |
  | Fine-tuned | 39.88% | 65.68% |

- Deployed live to `ChessTwinBot`. User reported real quality problems from actually playing it: bad blunders, and the bot not following Aman's known repertoire (French as Black, Catalan as White). Diagnosed directly (querying the model on those exact positions, no bot/API layer involved) — not a bug, just undertraining diluting personal signal against the pretrain prior.

### Run 2 — full-scale ("I don't care if it's a 24 hour run")

- Pretrain: full 4.17M positions, 4 epochs → val top-1 **40.9%**, top-3 **68.5%**.
- Fine-tune: full 1,062,664 positions, 6 epochs → val top-1 **47.1%**, top-3 **74.3%**.
- Ablation (same 5,000-row sample, for a clean comparison):

  | Model | Top-1 | Top-3 |
  |---|---|---|
  | Random | 7.08% | — |
  | Stockfish-best-move | 38.04% | — |
  | Pretrained-only | 41.52% | 70.68% |
  | **Fine-tuned** | **47.72%** | **75.26%** |

- French-response probability (after 1.e4): 10.3% → 17.7%. Catalan-setup probability (after 1.d4 Nf6 2.c4 e6): 8.4% → 17.8%. Real improvement, but neither became the model's single top prediction.
- Bot redeployed; then crashed unattended on a normal Lichess stream disconnect (`ChunkedEncodingError`) — fixed by adding reconnect handling to both the outer event loop and each in-game state stream, so a dropped connection no longer kills the process.

### Round 3 — still blundering, still not matching openings; two separate fixes

User's live-play assessment: still playing "like some 1500" against a real ~2500 rating, still not reliably following the French/Catalan repertoire. Two different root causes, two different fixes:

1. **Blundering is a move-selection problem, not (only) a training problem.** Pure greedy top-1 has no lookahead; on the ~50% of positions where the model doesn't guess Aman's exact move, some misses are real tactical blunders. **Fix (serving-time only, does not touch the trained model or the ablation metric)**: `serve/bot.py`'s `pick_move` takes the model's top-4 candidates by probability, runs a shallow Stockfish check (50ms/candidate) on each, and plays the highest-probability candidate that isn't drastically worse (>250cp) than the best-scoring one. Model-preference-first with a tactical veto. User confirmed keeping this configuration (vs. two more engine-forward alternatives offered) rather than trading more style fidelity for raw strength.
2. **Openings/midgame not matching is a training-signal-dilution problem.** Added `oversample_early_moves_factor` to `PositionDataset`: positions with `move_number < 10` are duplicated in the fine-tune train split only (never val, so val stays unbiased) to amplify Aman's most personally-recognizable signal against the much larger volume of generic mid/endgame positions. Set to 3x. Also pushed epoch counts further: `pretrain_epochs` 4→8, `finetune_epochs` 6→12.

### Run 3 — bigger push, hit repeated crashes, added checkpoint/resume

Kicked off pretrain (8 epochs, full data) autonomously. Hit **four consecutive crashes**, all at the exact same point (the very first CUDA kernel call of a freshly-started process), each with a different specific CUDA error (`unknown error`, then `CUBLAS_STATUS_NOT_SUPPORTED`, then `unknown error` again) — isolated sanity tests (bare Conv2d, bare Linear) succeeded immediately after each crash, ruling out a persistently broken GPU. Also hit one unrelated wandb IPC connection failure.

**Diagnosis**: the one constant across all 4 crashes was the Lichess bot process running concurrently, holding a `torch`-imported (if not yet CUDA-active) process alongside the training script's fresh CUDA context creation. Stopped the bot; pretrain then completed cleanly, which is consistent with (though not conclusively proof of) a WDDM-level conflict between two processes both touching CUDA on this laptop GPU. Current running policy: **do not run the bot and a training job on the GPU at the same time on this machine**; bot goes offline for the duration of any training run and is redeployed after.

**Fix, regardless of root cause**: added checkpoint-and-resume to both `pretrain.py` and `finetune.py` — a resume checkpoint (`{model,optimizer}_state_dict`, `epoch`) is saved after every epoch, and if the resume file exists at startup, training picks up from `epoch+1` instead of restarting from scratch. This turned a 4th crash from a wasted multi-hour run into a checkpoint-and-continue at zero real cost. (This also survived a session/process boundary: training continued unattended after the harness session ended, reached epoch 5 before stopping again for an unlogged reason, and resuming from the saved `pretrained_resume.pt` picked up cleanly at epoch 6.)

**Run 3 pretrain result** (8 full epochs, full 4.17M positions, resumed once mid-way): val top-1 **41.7%**, top-3 **69.0%** — only marginally better than Run 2's 4-epoch result (40.9%/68.5%), indicating the model was already near a plateau for this architecture/data size at 4 epochs; the extra epochs bought little at pretrain stage. Checkpoint: `checkpoints/pretrained.pt`.

**Run 3 fine-tune result** (12 epochs, 1,338,170 positions incl. 3x-oversampled early moves): val top-1 **47.8%**, top-3 **74.6%**. Checkpoint: `checkpoints/finetuned.pt`.

**Run 3 ablation** (same 5,000-row sample):

| Model | Top-1 | Top-3 |
|---|---|---|
| Random | 7.08% | — |
| Stockfish-best-move | 37.68% | — |
| Pretrained-only | 41.84% | 70.38% |
| **Fine-tuned** | **47.68%** | **75.42%** |

Statistically indistinguishable from Run 2 (47.72%/75.26%) — the aggregate metric has plateaued; 8 extra epochs and 3x opening-oversampling bought essentially nothing on the aggregate number.

**Opening spot-check, Run 2 vs. Run 3** — this is where it gets more interesting than the flat aggregate suggests:

| Position | Run 2 | Run 3 |
|---|---|---|
| French (1...e6 after 1.e4) | 17.7% | **8.3%** (regressed) |
| Catalan (2.g3 after 1.d4 Nf6 2.c4 e6) | 17.8% | **24.6%** (improved, closing on Nc3 43%/Nf3 29%) |

**Honest read**: the oversampling+more-epochs approach is not a clean, monotonic win. It reshuffled probability mass across openings rather than uniformly reinforcing "Aman's known repertoire" as a concept — Catalan preference strengthened, French preference weakened, and the aggregate barely moved. This is a real result, not swept under the rug: 12 epochs of repeated exposure to the same oversampled early-game rows plausibly started overfitting to specific move sequences/games rather than learning a generalizable "this is what Aman prefers in the opening" signal. Further improvement here likely needs a different lever than "more of the same" — e.g. a higher fine-tune learning rate specifically, contrastive/weighted loss favoring rare-but-repeated personal choices, or accepting this is close to the ceiling for a plain classification objective at this data scale.

Bot redeployed with the Run 3 checkpoint; back online with no crashes so far running solo (no concurrent training job).

### Run 4 — testing the "different lever": higher LR, fewer epochs, lighter oversampling

Backed up Run 3's checkpoint (`checkpoints/finetuned_run3_backup.pt`) before experimenting. Tried the alternative flagged above: `finetune_lr` 1e-4 → 2e-4, `finetune_epochs` 12 → 5, `finetune_oversample_early_moves_factor` 3 → 2 — hypothesis being that specializing faster with less repetition of the same rows would avoid the overfitting pattern seen in Run 3.

**Result**: val top-1 **48.1%**, top-3 **75.1%** — marginally higher aggregate than Run 3 (47.7%/75.4%), within noise.

**Opening spot-check vs. Run 3**:

| Position | Run 2 | Run 3 | Run 4 |
|---|---|---|---|
| French (1...e6 after 1.e4) | 17.7% | 8.3% | 10.4% (partial recovery, still well below Run 2) |
| Catalan (2.g3 after 1.d4 Nf6 2.c4 e6) | 17.8% | 24.6% | **15.6%** (regressed below both Run 2 and Run 3) |

**Decision: reverted to the Run 3 checkpoint**, not Run 4. The aggregate metric is a wash between the two, but Run 4's Catalan probability dropped notably — and the user was explicit that the Catalan is their *only* main White opening, so that specific regression outweighs a same-noise-level aggregate difference. `checkpoints/finetuned.pt` is Run 3's weights again; Run 4's checkpoint is kept at `checkpoints/finetuned_lr_experiment.pt` for reference, not deployed.

**Honest read, updated**: two different hyperparameter changes (more epochs+heavier oversampling in Run 3; higher LR+fewer epochs+lighter oversampling in Run 4) both moved the *aggregate* metric by less than 1pp in either direction, while swinging the two specific opening probabilities by 2-3x in different, uncorrelated directions each time. This is a strong signal that these particular position-level probabilities are high-variance under this training setup — small numbers of specific games dominate any one opening line's signal, so different training dynamics reshuffle which openings "win" more than they broadly improve personalization. Further blind hyperparameter search is unlikely to reliably fix a specific named opening; a more targeted fix (e.g., a small number of additional gradient steps on batches restricted to exactly the Catalan/French positions, weighted loss, or accepting the current level as the ceiling for a plain classification objective at this data scale) would be needed to move a specific opening deliberately rather than as a side effect.

### Run 5 — targeted opening-only specialization (a more surgical attempt)

Rationale: instead of another full retrain with different global hyperparameters, train a few more gentle-LR (5e-5, half of the main fine-tune's 1e-4) epochs starting from Run 3's checkpoint, using *only* early-game positions (`move_number < 10`, 137,753 train / 17,230 val rows) — the idea being to nudge opening behavior specifically without touching the far larger mid/endgame signal. New script `train/finetune_openings.py`, 4 epochs.

**On its own narrow slice, it worked great**: early-move-only val accuracy 63.3% top-1 / 90.0% top-3 (vs. ~48%/75% general) — unsurprising, since it's now specialized on a small, repeated distribution.

**But it cost general competence and still didn't clearly fix the openings**:

| Metric | Run 3 (deployed) | Run 5 (this experiment) |
|---|---|---|
| Full test sample, all move numbers (top-1) | 47.68% | **45.86%** (regressed) |
| French (1...e6 after 1.e4) | 8.3% | 12.6% (improved, still below Run 2's 17.7%) |
| Catalan (2.g3 after 1.d4 Nf6 2.c4 e6) | 24.6% | 16.8% (regressed) |

**Decision: kept Run 3 deployed, did not switch to Run 5.** The overall move-matching drop (-1.8pp on the full test sample) is a real cost — mild catastrophic forgetting from additional gradient steps touching shared weights, even though the training data was opening-only — and it didn't produce a clean win on the openings either (one up, one down, same pattern as every other experiment). Checkpoint kept at `checkpoints/finetuned_openings.pt` for reference, not deployed.

**Standing conclusion after 3 failed attempts to move the needle (Run 3→4, Run 3→5 general, Run 3→5 opening-only)**: further training-based tweaking at this data scale has not reliably improved on Run 3, and each attempt costs real compute time for a coin-flip (or negative) outcome. Stopping further blind training experiments here. Run 3 (`checkpoints/{pretrained,finetuned}.pt`) remains the best available and is what's deployed. Any further improvement likely needs either substantially more personal data, a fundamentally different technique (e.g. a per-opening classification head, weighted loss explicitly penalizing personal-repertoire misses more than generic ones), or accepting the current level.

## 7. Serving

- `serve/api.py`: FastAPI `/predict` (FEN → ranked moves + confidence, pure model output, no engine involvement — this is what `eval/evaluate.py` and the ablation table measure) and `/health`. 4/4 tests passing.
- `serve/bot.py`: live Lichess bot (`lichess.org/@/ChessTwinBot`, upgraded to BOT status after confirming via `/api/account` it was a genuinely fresh zero-game account — irreversible, so this was verified before acting). Reconnects on dropped streams (both the outer event loop and each in-game handler). Move selection: model top-4 candidates, Stockfish blunder veto (250cp threshold, 50ms/candidate) — see Round 3 above. **Live**, running the Run 3 checkpoint.

## 8. Known limitations (honest, not hidden)

- **Personalization has a real ceiling at this data/compute scale.** ~48% top-1 move-matching means the model still doesn't play like Aman more than half the time; openings/midgame style-matching has improved measurably across three training rounds but has not "locked in" (i.e., become the single top prediction for known repertoire choices like the Catalan).
- **The Stockfish blunder veto increases real playing strength but doesn't target ~2500 overall.** It only vetoes candidates that are much worse than the model's own best alternative — it doesn't upgrade a merely-mediocre-but-not-blundering move into a strong one. User was offered two more engine-forward alternatives (prefer Stockfish's move when model confidence is low; rank primarily by Stockfish eval) and explicitly chose to keep the current, most style-faithful configuration.
- **This machine cannot run parallel data loading** (Windows DataLoader multiprocessing OOMs) and has shown intermittent CUDA crashes when a GPU-resident bot process runs alongside a training job — current policy is training and live bot are never run concurrently.
- **Evaluation uses a 5,000-row sample** of the 133,956-row personal test set (Stockfish at 50ms/move over the full set would take ~1.6 hours). Large enough to be meaningful; exact percentages would shift slightly on the full set.
- **Blunder-rate fidelity and calibration analysis** (matching Aman's actual blunder rate, reliability diagrams) were planned as portfolio-differentiating extras but not built — deprioritized in favor of the core pipeline and the live-play quality fixes.

## 9. Reproducing the pipeline

```
python data/fetch_personal_games.py       # Stage 1
python data/fetch_pretrain_games.py       # Stage 2
python data/process_pgn.py                # Stage 4
python data/splits.py                     # Stage 5
python models/baselines.py                # Stage 7
python -m train.pretrain                  # Stage 8 (auto-resumes from pretrained_resume.pt if interrupted)
python -m train.finetune                  # Stage 9 (auto-resumes from finetuned_resume.pt if interrupted)
python -m eval.evaluate                   # Stage 10
uvicorn serve.api:app --reload            # Stage 11 (API)
python -m serve.bot                       # Stage 11 (Lichess bot; needs LICHESS_BOT_TOKEN in .env; stop before training)
```

Setup, checkpoint download and Stockfish install are in `README.md`.

## 10. Pack-up (2026-11-12)

The project is finished. Final cleanup before archiving:

- **CI had never passed.** There was no pytest config, so a bare `pytest` (what CI runs) couldn't import the repo's packages; it only worked locally via `python -m pytest`. Fixed with `pyproject.toml` (`pythonpath = ["."]`). CI also now installs CPU-only torch.
- **Config restored to Run 3.** `configs/default.yaml` had been left on the reverted Run 4 fine-tune settings. It now matches the config embedded in the deployed `finetuned.pt` (lr 1e-4, 12 epochs, 3x oversampling).
- **Stockfish path is portable.** It was hard-coded to a Windows `.exe`; it now defaults to `stockfish` on PATH.
- **Removed:** wandb (disabled since Run 3, so its calls were no-ops), an unused `process_pgn_file` helper, a placeholder `1+1==2` test, and a duplicated `.env` loader (`serve/matchmaking.py` no longer imports torch through `serve/bot.py`).
- **Stale summaries fixed.** `data/processed/summary.json` and `splits_summary.json` still described the pre-`due4awinn` dataset; regenerated from the parquet. The stored splits were confirmed to regenerate byte-for-byte from seed 42.
- **Artifacts moved off the dev machine.** Checkpoints are on [huggingface.co/amanDS5153/ChessTwin](https://huggingface.co/amanDS5153/ChessTwin) (public). Raw PGNs (zstd), processed parquet, splits and eval outputs are on the private `amanDS5153/ChessTwin-data` HF dataset. `finetuned_run3_backup.pt` was a byte-identical copy of `finetuned.pt` and was dropped.
