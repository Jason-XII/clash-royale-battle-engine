# Clash Royale imitation learning

Replays → native engine → compact games → imitation learning → live play on Null's Royale.

The programs you run are at the top level; what they share is the `cr` package.

| Program | What it does | Where it runs |
|---|---|---|
| `convert.py` | Replays the collected battles in the native engine and writes one archive per game (both players' inputs and labels) | Linux, Slurm |
| `train.py` | Imitation learning: the training loop | GPU, Slurm |
| `play.py` | Boots the emulator, reads the game's memory, runs the model and taps | Mac |
| `arena.py` | Paired-seat native battles between two models | Linux, Slurm |
| `ppo.py` | PPO from the IL model: the DDP learner with a separate critic (`learn`), and actor jobs (`act`) | GPU, Slurm |
| `loop.py` | The improvement loop around a PPO run: frozen champion vs learning challenger, gates with evaluation games, promotion, rollback, job recovery, `RUN/report.md` | Slurm login node (light) |

| `cr/` module | What it holds |
|---|---|
| `observation.py` | The observation of one decision (`PlayerView`), legal placement, the clock and elixir timing, the vocabulary. NumPy only |
| `policy.py` | The network, loading a checkpoint, packing inputs, sampling actions (masks, temperature), decoding them; `Agent` |
| `imitation.py` | Imitation's loss and data: converted games to chunks and padded batches |
| `engine.py` | The native simulator as the models play it (`Engine`, `Inputs`, matches from the deck pool), one game (`Game`: views, then step with both players' choices), and the rewards' parts |
| `matchmaking.py` | Who plays whom: opponent weights (PFSP, fixed shares), each game's two models, evaluation matches, a specialist's deck seat and opponent decks |
| `actors.py` | PPO's collectors and decision servers, their wire format, and the link to the learner |

`tools/` holds small standalone programs: `build_features.py` (the optional C helpers), `card_stats.py`, and the live
displays `winbar.py` and `overlay.py`. `experiments/` holds measurement scripts kept for reuse. `tests/` checks it all
without an engine, emulator or GPU. Checkpoints hold plain dicts (config, weights, vocabulary), so they load whatever
module the code is in.

`assets/` holds the game-build tables (card costs and rarities, effects, placement masks, replay card names, the
881-token vocabulary, the supported client's hashes). The device reader binary `assets/clapha-reader-nulls` is not in
the repository: get it from Arron and put it there before playing live.

## Convert

Needs the clapha-engine native adapter, the engine builds and pyarrow (not torch). The version oracle's
`assignment.jsonl` says which engine build replays each game exactly.

```bash
sbatch -A stu -p Students -q qos_stu_medium_2gpu -c 24 --mem=100G -t 1-00:00:00 --wrap "ulimit -u 65536; \
  python convert.py --assignment ~/oracle/assignment.jsonl --replays ~/cr-engine/data/IL_Replay/replays \
  --engine d86 ~/engine-86/engine ~/engine-86/data/assets --engine d29 ~/engine-29/engine ~/engine-29/data/assets \
  --clapha-engine ~/clapha-engine --out ~/il-data/next --workers 23"
```

Each game's archive is written whole or not at all, so rerunning the same command continues where it stopped. A game
that does not replay exactly, or that crashes the engine twice, gets an archive marked rejected with the reason.
`manifest.json` appears once every assigned game has an archive; that is what `train.py` reads.

## Train

```bash
sbatch -A competition -p P107-A100 -q qos_p107-a100 --gres=gpu:3 -c 16 --mem=100G -t 10:00:00 --exclude=anode02 \
  --wrap "torchrun --standalone --nproc_per_node=3 train.py --data ~/il-data/full-reuse-20261014 --out ~/il-runs/next"
```

The defaults are the current teacher's (4 epochs, global batch 32, chunks of 128 decisions, lr 3e-4 with 500 warm-up
steps, next-play loss 0.5). On CUDA it trains in bf16 with `torch.compile`. `--device cpu --workers 0` is for small
checks. The tile scores are computed in fp32 even under bf16: they drift together into the thousands, where bf16
rounds neighbouring tiles to the same score. `--final_lr 0.1` decays the learning rate linearly to a tenth by the last
step (default: constant). The medium model (3.4M parameters, ~12 steps/s on 3 A100s; ahead of the default 1.26M model
at every step): `--card_dim 64 --object_dim 96 --attention_layers 3 --conv_channels 96 --conv_layers 6 --core 384`.
The large model (8.0M, v200's network): `--card_dim 64 --object_dim 160 --attention_layers 6 --conv_channels 160
--conv_layers 12 --core 448 --board 32 --epochs 4.5 --final_lr 0.1 --warmup 2000`.

Architecture options, all off by default so every saved checkpoint loads unchanged (results in
[docs/TRAINING.md](docs/TRAINING.md), section 11):
- `--res_blocks 1`: the board network becomes a stem conv plus `conv_layers / 2` pre-activation residual blocks.
- `--tile_blocks N`: residual blocks in the tile head, after the card conditioning.
- `--res_norm 0`: residual blocks without GroupNorm. It trains faster, but the loss is worse.
- `--dist_bias 1`: object attention biased by the distance between objects. It helps a little but trains ~25% slower.
- `--flight_grid 1` (plays in flight drawn on the board) and `--board_rows / --board_cols` (the coarse map): no gain.
- `--card_stats 1`: every card and unit id also gets its fixed stats from the game data (range, targets, hitpoints,
  damage, speed, splash, ...: `assets/card_stats.json`, 28 fields), through a learned map added to its embedding.
  -0.037 held-out loss in the 30k-game ablation (section 12), but no gain at full data (r4, section 13). Rebuild the table for a new game build with
  `python tools/card_stats.py CSV_LOGIC assets/vocab.json assets/replay_cards.json UNIT_STATE_TABLES.json assets/card_stats.json`
  (`CSV_LOGIC`: the build's `data/assets/csv_logic`; the unit-state tables name the units' data ids).
- `--max_train N --max_val N`: random subsets of the games, for ablations.
- `--ability_labels 1`: each human ability press is labelled where it was tapped (4 decisions earlier) and on the
  controller that was ready; the converted labels mark it 22 ticks late and always on controller 1 (section 12).

The candidate to replace the large model has about 3x less compute per decision and a lower held-out loss: the large
flags with `--conv_channels 96 --conv_layers 8 --res_blocks 1 --tile_blocks 1`.

`latest.pt` is saved every 1,000 steps. Running the same command again resumes it with the settings in
`settings.json`, so a job killed at its time limit loses at most 1,000 steps. A resumed run shuffles the data afresh
instead of replaying the interrupted order exactly. Progress goes to `metrics.jsonl`, with validation after each pass.
Don't change the code under a running job: copy the checkout for long runs.

## Play

```bash
python play.py ~/cr-live/models/gen-large-v200/gen-large-v200.pt --tables ~/cr-live/models/unit_state_tables_150535029.json \
  --temperature 0.25
```

`--temperature` sets the card and tile temperature; when to play stays sampled as trained. Below 1 the model picks
its preferred cards and tiles more consistently, and 0.1–0.25 has been far stronger than the default 1.0 for every
model tested (see `docs/TRAINING.md`, section 9).

It boots the visible `Pixel_9_2` emulator if needed (`-gpu host`), verifies the game library, installs the reader,
re-applies the review-crash patch an emulator snapshot may have rolled back, and waits for a battle. Start a friendly
battle by hand, or add `--queue` to press Battle on the ladder. `--games N` plays several. Logs go to
`~/cr-live/runs/live` (OneDrive stalls writes). The model sees advance opponent commands from the reader's queue.
`--losses N` plays ladder games until the N-th loss. `python tools/winbar.py` opens a blue/red bar of the model's predicted
win chance, (value + 1) / 2 from the policy's value head, following the newest live log. That head is IL's (PPO
doesn't train it), so the bar is a rough display: it starts near 60% for most decks.

## PPO

```bash
# the learner (3 GPUs) and 13 collectors on the 4th GPU, in one A100 job
sbatch -J cr-ppo -A competition -p P107-A100 -q qos_p107-a100 --gres=gpu:4 -c 16 --mem=100G -t 4-00:00:00 --exclude=anode02 \
  --wrap "python ppo.py act RUN --collectors 13 --gpus 3 & torchrun --standalone --nproc_per_node=3 ppo.py learn \
  --init ~/il-runs/full-compact-ddp-20261014/latest.pt --out RUN --engine ~/clapha-engine --pool ~/cr-engine/data/deck_pool.json"
# more actors in any job with CPUs to spare; they find the learner through RUN/endpoint.json
sbatch -J cr-ppo ... --gres=gpu:4 -c 16 --wrap "python ppo.py act RUN --collectors 34 --servers 4"
```

Each collector plays one native game at a time; a decision server per GPU answers the waiting collectors in one CUDA-graph
call. Inputs use one pinned transfer per model group and results are read back together. Collectors route each sampled
opponent to a balanced server cohort, reducing the number of small model groups without changing opponent selection.
Use `act --no-fast-inference --no-opponent-affinity` for the previous serving path. PPO uses the same channels-last
convolution layout as IL on CUDA; `learn --no-channels-last` disables it (a saved setting takes precedence on resume).
`python tools/build_features.py` builds optional CPython helpers (into `cr/`) for memory reads and unit features with the system C compiler;
build once per Python/platform. They preserve the existing observation arrays. `CR_NATIVE_FEATURES=0` selects Python
helpers at process startup. Without the extension, the Python path is used automatically.
Half the games are self-play, the rest against the IL model, recent snapshots (`RUN/pool/`) or promoted champions
(`RUN/league/`), chosen by how often they still beat the policy. Reward: the result (±1), a small penalty per decision
at full elixir, tower-damage shaping, and an end-of-game margin term. The critic is a separate network used only in training: the policy's network (from the IL weights) plus the
opponent's hand, next cards and ability, with a P(win) head trained on the outcome and a head for the shaping rewards.
The first 20 updates train the critic alone. Progress goes to `RUN/metrics.jsonl` (`score`, `win_ev`, `kl_ref`,
`games_per_hour`, ...). Rerunning the learner resumes from `RUN/latest.pt`; actors reconnect by themselves.
`scancel --name=cr-ppo` stops a run started as above.

Settings live in `RUN/settings.json`; every default in `ppo.py`'s `DEFAULTS` is also a `learn` flag. Besides the
reward ones (`shaping`, `margin`: the end-of-game difference in lowest tower HP):
- `frozen_temperature`: the card/tile temperature of every frozen model (opponents, evaluation and benchmark
  games), so they play as deployed. Only the learner gets the 1% ability-press floor (`ABILITY_EXPLORE`).
- `learner_temperature` and `learner_mix`: the temperature the learner plays at, with that share of its untempered
  policy mixed in to keep exploring. PPO scores the log-probabilities of exactly that distribution.
- `against`: another run's directory. The run then plays only that run's current champion: an exploiter that looks
  for what beats it (use with `self_play` 0).

A specialist (one deck; FirstLight's Hog 2.6 recipe, 23 October): these go in `RUN/settings.json` before the first
start.
- `deck` (a deck as in the pool) and `self_play` 0: a learning seat always plays it. In gate games the candidate plays it.
- `fixed_opponents`, `{name: [checkpoint, share]}`: those frozen models get that share of the learning games (e.g. the
  IL model 0.4 and the general champion 0.4); the rest go by PFSP to the run's own snapshots and league.
- `deck_mix`, e.g. `{"focus": 0.3, "hard": 0.3}`: that share of the opponent decks comes from the pool's 20 most common
  decks, and that share from the decks the specialist scores below 55% against (`RUN/hard-decks.json`, weighted by how
  hard; its scores by deck are in `RUN/deck-scores.json`). The rest keep the pool pair's own deck.
- `bc_index` (a file from `experiments/deck_index.py`: the human games played with that deck), `bc_coef` 0.2,
  `bc_batch` 8: after each PPO update, one imitation step on that many random 128-decision chunks of the winners' games.
  It keeps the specialist close to how humans play the deck. Metrics `bc_*`.

```bash
python experiments/deck_index.py ~/il-data/full-reuse-20261014 hog26-views.json \
  --cards HogRider,Musketeer,Cannon,Fireball,Log,Skeletons,IceGolemite,IceSpirits --procs 20   # in a job
```

### The improvement loop

```bash
cd SNAPSHOT && nohup python loop.py RUN >> RUN/logs/loop.out 2>&1 &   # on the login node: it only submits and reads
```

`RUN/loop.json` holds the jobs' sbatch commands (with `--parsable`), the benchmark models and the loop's settings
(`LOOP_DEFAULTS` in `loop.py`). It keeps a frozen champion, which is also the KL reference. Every snapshot goes through
a gate of evaluation games played by the collectors: promoted at ≥ `promote` (default 55%) against the champion if its benchmark mean
isn't more than 3 points below the champion's, with every benchmark's `bench_games` games played (a timed-out gate
without them is inconclusive). Benchmark names must not look like this run's snapshots (`v000100`). Promoted champions stay in `RUN/league/` as opponents. The loop also
resubmits jobs that ended or hung, rolls back after repeated bad gates, stops after `stall_after` gates without a
promotion, and writes `RUN/report.md` and `RUN/loop-events.log`. To stop a loop run, kill `loop.py` first (it
resubmits cancelled jobs), then `scancel` the job ids in `RUN/loop-state.json`.

## Arena and tests

```bash
python arena.py A.pt B.pt --engine ~/clapha-engine --pool ~/cr-engine/data/deck_pool.json --games 20 --out runs/arena.jsonl \
  --a-temperature 0.1 --b-temperature 0.1   # compare models as they are deployed
# --a-overlap: A may tap a card while another of its plays is still in flight (humans do; off everywhere by default)
OMP_NUM_THREADS=2 python -m pytest tests
```

Python 3.10+, NumPy and PyTorch; pyarrow for conversion; pytest for the tests. On a Mac, torchrun needs
`--nnodes=1 --master_addr=127.0.0.1` because the hostname does not resolve.

Measured game behaviour (timing, the client's held taps, the reader) is in
[docs/GAME_INTEGRATION.md](docs/GAME_INTEGRATION.md). Model and features came from ClashAlpha/clapha-il; native execution
from clapha-engine and FirstLight's reverse engineering.

## Docs

- [docs/TRAINING.md](docs/TRAINING.md): IL and PPO training — design, metrics, results and why each change was made.
- [docs/THROUGHPUT.md](docs/THROUGHPUT.md): throughput measurements, optimized serving, validation and rollout details.
- [docs/GAME_INTEGRATION.md](docs/GAME_INTEGRATION.md): how the live client is read and controlled.
