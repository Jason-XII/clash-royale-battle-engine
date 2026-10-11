# Clash Royale Battle Engine

这个仓库允许你把皇室战争的原生战斗引擎当成 API 来调用，在 25 万场真人回放上做模仿学习，再用 PPO 自博弈继续训练，最后在 Null's Royale 上实战。开发过程中 agent 总结的经验蒸馏成了 3 个 skill，放在 `.agents/skills` 目录下。

本仓库是对 [FirstLight-CR](https://gitlab.com/firstlight3/FirstLight_CR) 逆向工作加以改进后的产物，极大简化了项目结构和通信接口。PPO 阶段的引擎来自 [clapha-engine](https://github.com/arronwang01/clapha-engine)（x86_64 Linux 上直接加载游戏的 `libg.so`，无需模拟器）。

## The pipeline

| Stage | Where | Engine | Code |
|---|---|---|---|
| 1. Rebuild 252k human replays, cache raw states | Mac | ARM64 `libg.so` in an Android emulator + our probe | `replay.py`, `build_cache.py` |
| 2. Imitation learning | GPU node | none (cache only) | `il/data.py`, `il/model.py`, `il/train.py` |
| 3. PPO self-play | GPU node (x86_64 Linux) | clapha-engine, in-process | `clapha.py`, `il/rl.py` |
| 4. Evaluation (model vs model) | GPU node | clapha-engine | `il/arena.py`, `il/eval.py` |
| 5. Live Null's Royale games | Mac | the game itself + our probe's live mode | `il/live.py` |

Every stage uses the same observation and model (`il/data.py` features, `il/model.py` policy), so a checkpoint
from any stage plays in any other.

| Path | What |
|---|---|
| `engine.py`, `protocol.py` | `Engine` (probe, emulator): reset any deck, step with plays, schedule replay commands; the `State` every stage uses |
| `clapha.py` | `ClaphaEngine`: the same interface and `State` on clapha-engine, plus the live deploy delay |
| `replay.py`, `build_cache.py` | rebuild IL_Replay games in the engine and cache raw states (Mac) |
| `il/` | features, model, IL training, PPO (`rl.py`), arena / evaluation, live client, tests |
| `tools/` | engine setup (`start_engine.sh`, `build_probe.sh`), measurements (`clapha_check.py`, `clapha_throughput.py`, `serve_bench.py`, `measure_delay.py`), `build_deck_pool.py`, Mac heat guard |
| `bootstrap_emulator.py`, `install_probe.sh`, `benchmark.py`, `status.py`, `batch.py` | emulator engine setup and checks |

## 1–2. Imitation learning

`data/` is a symlink to `~/cr-data` (outside OneDrive, not in git). IL_Replay (252,238 games) comes from
HuggingFace `VanguardX101/IL_Replay`.

**Engine (Mac).** Once per emulator install, put the probe into the prepared Nulls Royale APK, then start the
engine headless and run the regression check:

```bash
./install_probe.sh                    # prebuilt/libcrprobe.so; rebuild from source with tools/build_probe.sh
tools/start_engine.sh                 # headless emulator (-gpu host) + bootstrap_emulator.py
python -m native_engine.benchmark     # must print result=1 crowns=(0, 1) final_tick=3835
```

**Rebuild and cache (Mac, ~11 h, resumable).** FirstLight's converter (`../firstlight-cr`) handles side flips,
deal selection, forms and ability hints. About 91% of rebuilt games end with the real winner and crowns; only
those are used for training. 253 shards (17 GB) of raw states every 5 ticks, no features, so features can
change without rebuilding.

```bash
nohup nice -n 10 python -m native_engine.build_cache data/IL_Replay/replays data/cache > data/cache/build.out 2>&1 &
```

**Train (GPU).** ~880k-parameter policy: entities scattered on the 32×18 tile grid, a conv tower, one LSTM,
a joint (wait | ability | hand slot × cell) head and a value head. ~10 h for 4 epochs on an A100.

```bash
python -m il.train data/cache runs/il-002 --epochs 4     # metrics.jsonl, best.pt, last.pt
```

`il-002` (labels on the board just before each play executed) is the PPO starting point. `il-003` shifted labels
22 ticks earlier to match our live deploy delay and lost to il-002 (35% in an E-Barbs mirror): 22 ticks is our
Nulls pipeline's delay; the human games' delay is evidently shorter.

## 3. PPO on clapha-engine (x86_64 Linux)

**Setup** on the node: clapha-engine prepared with its August build (150535029, the game version of our IL data
and of the Nulls server), this repo installed, the cache and the deck pool copied in:

```bash
cd ~/clapha-engine && python3 prepare.py ... && python3 run_gates.py      # see clapha's README
cd ~/cr-engine && pip install -e .
# data/deck_pool.json: real deck pairs with evolutions, built on the Mac by tools/build_deck_pool.py
PYTHONPATH=. python tools/clapha_check.py ~/clapha-engine                 # threads, read equivalence, speed
```

**Run** (resumes from `latest.pt`; `nice` keeps an interactive session responsive):

```bash
nohup nice -n 10 env PYTHONPATH=. python -m il.rl ~/clapha-engine ~/cr-engine/runs/il-002/best.pt \
  ~/cr-engine/runs/ppo-general --collectors 48 --servers 2 --pool ~/cr-engine/data/deck_pool.json \
  > ~/cr-engine/runs/ppo-general.out 2>&1 &
```

**Processes.**
- *Collectors* (one engine each): play games with the live deploy delay (a play lands 22 ticks after the
  decision; a slightly unaffordable card waits for elixir, as the game does; one play in flight per player),
  build both seats' features and send trajectories to the learner.
- *Decision servers* (`--servers`, GPU): answer their collectors' decisions in batched calls (gathered for
  `--batch-ms`), keep each seat's LSTM state in one indexed GPU tensor, hold the frozen opponents.
- *Learner* (GPU): assigns games (deck pair, opponent), runs PPO, publishes weights to the servers through
  shared memory after every update, adds a snapshot to the opponent pool every `--snapshot-every` updates.

**How we use clapha** (`clapha.py`, nothing in clapha-engine is modified): `CrxEnv` boots the engine; observations
come from its `Reader` (objects, the engine's own result) plus a leaner read of hands, cycle and elixir with each
battle's deck cached (the reader re-reads both decks per call); plays go through `queue_hand_action_at`.
Before boot we register `sysconf` / `get_nprocs` shims in clapha's loader so the engine sizes its thread pool for
2 CPUs. Cards played are detected from hand changes between snapshots; projectiles are objects whose data class is
10 (`LogicProjectileData`); costs come from `card_costs.json` (variable ones from the engine's card selection).

**Algorithm.** Half the games are self-play (both seats learn), half against a frozen pool (IL + snapshots)
picked by PFSP, weight (1 − win rate)² + 0.05. Policy: wait with the model's p(wait), else card × cell sampled
at temperature 0.5 (ability masked: clapha cannot play abilities yet). Reward: ±1 result, ±0.1 per crown,
−0.002 per decision at full elixir. PPO (clip 0.2, γ 0.999, λ 0.95, lr 2e-5, 2 epochs, whole games as LSTM
sequences) with a KL penalty (0.05) towards the IL policy. Decks: real ladder deck pairs with evolutions and
the game's levels (mostly 16).

**Log** (`runs/<run>/metrics.jsonl`, one line per update): `pool` (score vs each frozen opponent, −1..1),
`kl_ref` (distance from IL), `approx_kl` / `clip_frac`, `plays_per_game`, `full_elixir`, and where time goes:
`collector_ms` (features / wait / engine per decision), `server_s`, `mean_batch`, `games_per_hour`.

**Throughput** on our allocation (8 physical cores / 16 hyper-threads, one A100): ~13–15k games per hour.
Measured costs: the engine steps at ~59k ticks/s per core; a decision server call costs ~7–9 ms whatever its
batch (kernel launches and Python, the GPU is mostly idle), so throughput comes from large batches (many
collectors per server), not many servers; collectors spend ~1.9 ms on features and ~1.1 ms in the engine per
decision. Next steps if needed: CUDA graphs for the server forward, one packed input transfer, faster features.

## 4. Evaluation

```bash
# any two checkpoints, mixed real deck pairs or a mirror deck (id or id:el)
PYTHONPATH=. python -m il.arena ~/clapha-engine A.pt B.pt --games 400 --decode policy
# every new snapshot of a PPO run (and --latest) vs a reference: 'mixed' and 'ebarbs' arenas -> <run>/eval.jsonl
nohup nice -n 10 env PYTHONPATH=. python -m il.eval ~/clapha-engine ~/cr-engine/runs/ppo-general \
  --reference ~/cr-engine/runs/il-002/best.pt --latest > ~/cr-engine/runs/eval.out 2>&1 &
```

The summary reports both sides' plays per minute: a high score against an opponent that stopped playing proves
nothing (FirstLight's lesson). First result: after ~9k games, PPO beats il-002 71.5% ± 6.3% on mixed decks and
76.0% ± 5.9% in the E-Barbs mirror, with the IL opponent equally active.

## 5. Live Null's Royale games (Mac)

The probe's live mode captures the battle the game itself runs (on the game thread after each battle
controller update), so the model gets exactly its training features; cards are played by adb taps.

```bash
python bootstrap_emulator.py --online
python -m il.live runs/il-002/best.pt --deck 26000043,26000020 --games 10 --queue   # --decode policy for PPO models
```

`--queue` presses Battle / OK by the button colour and restarts the game if it dies. il-002 went 9–3 from 0 to
~350 trophies. Measured deploy delay: 21–22 ticks from tap to card played (`tools/measure_delay.py`).

## Pitfalls we hit (read before changing things)

- **Levels.** A bare match template plays everything at level 1. Matches need `lvlcap`, `cardlvlmin`,
  `avatar*.expLevel` **and the king tower `hbd[*].kt`** (otherwise a level-1 king: 2400 hp instead of 7728,
  84% three-crowns). `il.arena.make_match` takes care of it.
- **Evolutions.** 86% of ladder decks have two `el=1` cards and one `el=2` card; the cache does not record them,
  so decks come from `data/deck_pool.json` (raw replays).
- **Threads.** An engine sized its thread pool from the node's 128 CPUs (~70 threads); `ulimit -u` (1024 on the
  cluster) counts every thread of the user. `ClaphaEngine` tells the engine it has 2 CPUs (70 → 25 threads) and
  `raise_thread_limit()` runs in the parent before any engine process starts. Add `ulimit -u 65536` to your shell
  profile, or a busy node stops your shell from forking.
- **CUDA and fork.** Anything that touches CUDA before forking (even `torch.cuda.is_available()` in an argparse
  default) breaks forked GPU processes; `il.rl` picks the device after forking and refuses to fork otherwise.
- **The live game's rate-app dialog** calls a Java class the patched APK lacks (JNI abort after every battle);
  the probe clears its pending flag before each controller update.
- **Git on the cluster.** GitHub is often unreachable from campus; we push to the node over SSH instead
  (`git push cluster main` with `receive.denyCurrentBranch updateInstead` there).

Tests: `python -m unittest il.test_il` (needs `data/cache-smoke`, see `build_cache --parts 0 --limit 100`):
features and labels, the PPO policy distribution, the decision server's per-seat state, and (with the emulator
engine running) online features equal to cached ones.
