# Clash Royale Battle Engine

这个仓库允许你把皇室战争的原生战斗引擎当成 API 来调用，在 25 万场真人回放上做模仿学习，再用 PPO 自博弈继续训练，最后在 Null's Royale 上实战。

本仓库是对 [FirstLight-CR](https://gitlab.com/firstlight3/FirstLight_CR) 逆向工作加以改进后的产物，极大简化了项目结构和通信接口。PPO 阶段的引擎来自 [clapha-engine](https://github.com/arronwang01/clapha-engine)（x86_64 Linux 上直接加载游戏的 `libg.so`，无需模拟器）。

## The pipeline

| Stage | Where | Engine | Code |
|---|---|---|---|
| 1. Rebuild 252k human replays, cache raw states | Mac | ARM64 `libg.so` in an Android emulator + our probe | `build_cache.py` |
| 2. Imitation learning | GPU node | none (cache only) | `il/data.py`, `il/model.py`, `il/train.py` |
| 3. PPO self-play | GPU node (x86_64 Linux) | clapha-engine, in-process | `clapha.py`, `il/rl.py` |
| 4. Evaluation (model vs model) | GPU node | clapha-engine | `il/arena.py`, `il/eval.py` |
| 5. Live Null's Royale games | Mac | the game itself + our probe's live mode | `il/live.py` |

Every stage uses the same observation and model (`il/data.py` features, `il/model.py` policy), so a checkpoint
from any stage plays in any other.

```
__init__.py, engine.py   Engine / RenderedEngine (probe in the emulator) and the State every stage uses
clapha.py                ClaphaEngine: the same interface and State on clapha-engine, plus the live deploy delay
emulator.py              build / install the probe, start the headless emulator, bootstrap, benchmark (Mac)
build_cache.py           rebuild IL_Replay games (FirstLight's converter), cache raw states, build the deck pool
prebuilt/libcrprobe.so   the probe, built from vendor/firstlight/probe (FirstLight's source + our live mode)
il/data.py               features and labels from raw states (cache), and `Observer` (engine states, both seats)
il/model.py              the policy and Agent (plays a checkpoint in any engine)
il/train.py              imitation learning
il/rl.py                 PPO: collectors, decision servers, learner; `il.rl join` for collectors in another job
il/arena.py, il/eval.py  model vs model; PPO snapshots vs a reference
il/live.py               live Nulls games by adb taps; --watch: self-play in the emulator engine
il/test_il.py            features/labels, PPO distribution, decision server, online = cached features
tools/collectors.sbatch  a CPU-only Slurm job of extra collectors
```

## 1–2. Imitation learning

`data/` is a symlink to `~/cr-data` (outside OneDrive, not in git). IL_Replay (252,238 games) comes from
HuggingFace `VanguardX101/IL_Replay`.

**Engine (Mac).** Once per emulator install, put the probe into the prepared Nulls Royale APK, then start the
engine headless and run the regression check:

```bash
python -m native_engine.emulator install-probe   # prebuilt/libcrprobe.so; rebuild with build-probe (Android NDK)
python -m native_engine.emulator start           # headless emulator (-gpu host) + bootstrap
python -m native_engine.emulator benchmark       # must print result=1 crowns=(0, 1) final_tick=3835
```

**Rebuild and cache (Mac, ~11 h, resumable).** FirstLight's converter (`../firstlight-cr`) handles side flips,
deal selection, forms and ability hints. About 91% of rebuilt games end with the real winner and crowns; only
those are used for training. 253 shards (17 GB) of raw states every 5 ticks, no features, so features can
change without rebuilding. A crashed emulator is restarted and the replay retried.

```bash
nohup nice -n 10 python -m native_engine.build_cache data/IL_Replay/replays data/cache > data/cache/build.out 2>&1 &
python -m native_engine.build_cache data/IL_Replay/replays data/deck_pool.json --deck-pool 600   # for PPO
python -m native_engine.build_cache data/IL_Replay/replays/part-000000.parquet - --check 5       # fidelity
```

**Train (GPU).** ~880k-parameter policy: entities scattered on the 32×18 tile grid, a conv tower, one LSTM,
a joint (wait | ability | hand slot × cell) head and a value head. ~10 h for 4 epochs on an A100.

```bash
python -m il.train data/cache runs/il-002 --epochs 4     # metrics.jsonl, best.pt, last.pt
```

## 3. PPO on clapha-engine (x86_64 Linux)

**Setup** on the node: clapha-engine prepared with its August build (150535029, the game version of our IL data
and of the Nulls server), this repo installed, the cache and the deck pool copied in:

```bash
cd ~/clapha-engine && python3 prepare.py ... && python3 run_gates.py      # see clapha's README
cd ~/cr-engine && pip install -e .
ulimit -u 65536                                                           # also in ~/.bash_profile
```

**Run** (resumes from `latest.pt`; `nice` keeps an interactive session responsive). The current run:

```bash
cd ~/cr-engine
nohup nice -n 10 env PYTHONPATH=. python -m il.rl ~/clapha-engine ~/cr-engine/runs/il-002/best.pt ~/cr-engine/runs/ppo-general \
  --collectors 40 --servers 4 --pool ~/cr-engine/data/deck_pool.json \
  --lr 2e-4 --games-per-update 256 --lam 0.98 --kl-coef 0.01 > ~/cr-engine/runs/ppo-general.out 2>&1 &
```

With two GPUs in the job, `--server-device cuda:1` puts the decision servers on the second one, so the learner's
update and the servers stop sharing a GPU (one A100 was 96% busy). With `--remote`, more collectors can join from another job (`sbatch tools/collectors.sbatch`, or
`python -m il.rl join <run>`); they find the learner through `<run>/endpoint.json`, reconnect when it restarts
and give up on a peer silent for 180 s. Not needed now: the decision servers, not the collectors, limit us.

**E-Barbs specialist** (from the general model; the learning seat always plays our deck, the other seat a real
ladder deck played by the current model or a frozen one; only our seat's games train):

```bash
nohup nice -n 10 env PYTHONPATH=. python -m il.rl ~/clapha-engine ~/cr-engine/runs/ppo-general/latest.pt ~/cr-engine/runs/ppo-ebarbs \
  --learner-deck 26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026 \
  --collectors 40 --servers 4 --server-device cuda:1 --pool ~/cr-engine/data/deck_pool.json \
  --lr 2e-4 --games-per-update 256 --lam 0.98 --kl-coef 0.01 > ~/cr-engine/runs/ppo-ebarbs.out 2>&1 &
```

**Inputs added after IL** (zero-initialised, so older checkpoints play exactly as before; the cache lacks them, so
PPO learns them): our play or ability press in flight (masks everything else, as the engine allows one action in
flight), the hero / champion ability (ready, active, charges, cooldown; the ability is a play option only when
ready), per hand slot whether its next play is evolved and the evolution progress, the opponent's elixir as a
player tracks it (regeneration 1 per 2.8 s, x2 from tick 2400, x3 from 4800, measured on the cache), our next three
cycle cards, and, for the value head only, the opponent's whole deck (training-time information: it takes the deck
matchup out of the advantages and never reaches a decision). Live, the probe's minimal capture reports abilities and
evolution (schema cr-minimal.v4) and `il.live` taps the ability button.

**Processes.**
- *Collectors* (one engine each): play games with the live deploy delay (a play lands 22 ticks after the
  decision; a slightly unaffordable card waits for elixir, as the game does; one play in flight per player),
  build both seats' features and send trajectories to the learner.
- *Decision servers* (`--servers`, GPU): answer decisions in batched calls (gathered for `--batch-ms`) through
  CUDA graphs per model and batch size, keep each seat's LSTM state in one indexed GPU tensor. Each frozen
  opponent lives on one home server and its games go there.
- *Learner* (GPU): assigns games (deck pair, opponent, server), runs PPO, publishes weights to the servers through
  shared memory after every update, adds a snapshot to the opponent pool every `--snapshot-every` updates.

**How we use clapha** (`clapha.py`, nothing in clapha-engine is modified): `CrxEnv` boots the engine; observations
come from its `Reader` (objects, the engine's own result) plus a leaner read of hands, cycle and elixir with each
battle's deck cached; plays go through `queue_hand_action_at`. Before boot we register `sysconf` / `get_nprocs`
shims in clapha's loader so the engine sizes its thread pool for 2 CPUs. Cards played are detected from hand
changes; projectiles are objects whose data class is 10; costs are embedded (`COSTS`).

**Algorithm.** Half the games are self-play (both seats learn), half against a frozen pool (IL + snapshots)
picked by PFSP, weight (1 − win rate)² + 0.05. Policy: wait with the model's p(wait), else card × cell sampled
at temperature 0.5 (ability masked: clapha cannot play abilities yet). Reward: ±1 result, ±0.1 per crown,
−0.002 per decision at full elixir. PPO (clip 0.2, γ 0.999, whole games as LSTM sequences) with a KL penalty
towards the IL policy. Decks: real ladder deck pairs with evolutions and the game's levels (mostly 16).

**Log** (`runs/<run>/metrics.jsonl`, one line per update): `pool` (score vs each frozen opponent, −1..1),
`kl_ref` (distance from IL; flat at the penalty's equilibrium means the anchor holds learning back),
`approx_kl` / `clip_frac`, `value_ev_mc` / `value_ev_gae` (is the value head an informative baseline?),
`plays_per_game`, `full_elixir`, and where time goes: `collector_ms`, `server_s`, `mean_batch`, `games_per_hour`.

**Throughput** on our allocation (8 physical cores / 16 hyper-threads, one A100): ~9.2k games per hour of wall
clock before the changes below (the log's `games_per_hour` then left out the PPO update, during which collectors
stalled). Now the learner keeps receiving games while it trains (`cycle_s`, wall clock per update, and
`games_per_hour` over it; `wait_s`: how long the learner waited for games after its update), and a server call is one packed input copy, one CUDA graph (state gather, forward, sampling) and one
copy back per model: 1.7 ms instead of 3.2 ms, and 2.3 ms instead of 9.0 ms when a frozen opponent shares the
round. Per game, collectors spend ~1.5 CPU-s on features and ~0.8 CPU-s in the engine, so the node's CPUs are the
next limit: ~17.6k games/h measured (35 s per 256 trajectories, the update itself 28 s); more collectors in a second job (`tools/collectors.sbatch`) add CPUs.

## 4. Evaluation

```bash
# any two checkpoints, mixed real deck pairs or a mirror deck (id or id:el)
PYTHONPATH=. python -m il.arena ~/clapha-engine A.pt B.pt --games 400 --decode policy
# every new snapshot of a PPO run (and --latest) vs a reference: 'mixed' and 'ebarbs' arenas -> <run>/eval.jsonl
nohup nice -n 10 env PYTHONPATH=. python -m il.eval ~/clapha-engine ~/cr-engine/runs/ppo-general \
  --reference ~/cr-engine/runs/il-002/best.pt --latest > ~/cr-engine/runs/eval.out 2>&1 &
```

The summary reports both sides' plays per minute: a high score against an opponent that stopped playing proves
nothing (FirstLight's lesson).

## 5. Live Null's Royale games (Mac)

The probe's live mode captures the battle the game itself runs (on the game thread after each battle
controller update), so the model gets exactly its training features; cards are played by adb taps.

```bash
python -m native_engine.emulator bootstrap --online
python -m il.live runs/il-002/best.pt --deck 26000043,26000020 --games 10 --queue   # --decode policy for PPO models
python -m il.live runs/il-002/best.pt --watch data/IL_Replay/replays/part-000051.parquet --render   # self-play
```

`--queue` presses Battle / OK by the button colour and restarts the game if it dies.

## Results so far

| | |
|---|---|
| il-002 (IL) | val NLL 0.311, card accuracy 57%; plays like a human (7.2 plays/min vs 9.3), some elixir leaking |
| il-002 live | 9–3 on the Nulls ladder, 0 → ~350 trophies, E-Barbs / Giant Skeleton deck |
| PPO update 109 (~9k games) vs il-002 | 71.5% ± 6.3 mixed decks, 76.0% ± 5.9 E-Barbs mirror, IL opponent equally active |
| PPO latest vs its own update-450 snapshot | 54.8% ± 4.9 (400 games), still improving after the KL fix |

## How we got here

The project started as a from-scratch Python simulator with RL (a separate repo). Months of it ended in a dead end:
the simulator was never the real game, and the agents learned its quirks. On 2026-10-07 we switched to the real
engine and to human data, following what FirstLight had done (IL → General PPO → a Hog specialist that reached
Hall of Fame).

1. **The engine as an API.** FirstLight runs `libg.so` in a rooted ARM64 emulator with a TCP probe. We rewrote the
   Python side into one small `Engine` (reset any deck, step with plays, schedule replay commands, one `State`)
   and the probe's capture into a minimal one. On the Mac it is latency-bound at ~26–32k ticks/s.
2. **Replays → cache.** Rather than port FirstLight's ~2k-line replay converter, we borrow it. The cache stores raw
   states, not features, so features changed several times without the 11-hour rebuild. The headless emulator
   crashed once mid-build; build_cache now restarts it, and the recovered shard was byte-identical.
3. **Our own small model.** FirstLight's policy is 12.7M parameters; PPO speed was the bottleneck, so we designed a
   ~0.9M one. The first run (il-001) had a frozen LSTM: unnormalised pooled inputs saturated it and the value never
   learned. LayerNorms fixed it (il-002). Watching il-002 play was the moment IL proved to be the right path.
4. **Live games.** The probe gained a live mode: it hooks the battle controller's update and buffers a minimal
   capture on the game thread (the step hook's manager had no world). The game aborted after every battle (a
   rate-app dialog calls a Java class the patched APK lacks); the probe now clears that flag. Taps use the screen
   calibration of the old simulator's visualizer. We measured the deploy delay instead of assuming it: 21–22 ticks.
   il-002 went 9–3.
5. **The 22-tick question.** il-003 trained on labels shifted 22 ticks (to match our delay) and lost to il-002
   (35% in the E-Barbs mirror): 22 ticks is our pipeline's delay, not the human players'. PPO trains with the
   delay instead, so the policy learns to live with it.
6. **An engine on the GPU node.** The A100 node has no KVM, so no emulator. A friend's clapha-engine loads the
   x86_64 `libg.so` in-process; we wrapped it into the same `State` (`ClaphaEngine`). Each engine sized its thread
   pool from the node's 128 CPUs (~70 threads) and hit the user's 1024-thread limit; the CPU-count shims and
   `raise_thread_limit` fixed it.
7. **Matches that look like the ladder.** Arena games ended in three-crowns until we found the king tower level
   (`hbd.kt`) sits outside the card levels: level-1 kings. Human decks carry evolutions (86% of ladder decks), which
   the cache lacks, so the deck pool is extracted from the raw replays.
8. **PPO and its speed.** First one Python thread answered every decision (GPU 17% busy). Then: decision server
   processes, per-seat LSTM rows in one GPU tensor, one input copy per round, CUDA graphs, frozen opponents on a
   home server (with four frozen models every server ran five models per round, halving throughput). Remote
   collectors in a second Slurm job worked across nodes but did not help: the servers, not the engines, are the
   limit. A NODE_FAIL showed collectors hanging on half-open TCP; they now time out after 180 s.
9. **Watching learning.** `kl_ref` sat flat at the penalty's equilibrium for 400 updates: the IL anchor (0.05) was
   holding PPO in place. At 0.01 learning resumed (latest beats update-450 54.8%). `value_ev_*` show the value head
   explains ~35–50% of the returns.

Next: General PPO until it plateaus, then an E-Barbs specialist from it, then climb the Nulls ladder.

## Pitfalls we hit (read before changing things)

- **Levels.** A bare match template plays everything at level 1. Matches need `lvlcap`, `cardlvlmin`,
  `avatar*.expLevel` **and the king tower `hbd[*].kt`** (otherwise a level-1 king: 2400 hp instead of 7728,
  84% three-crowns). `il.arena.make_match` takes care of it.
- **Evolutions.** The cache does not record them, so decks come from `data/deck_pool.json` (raw replays).
- **Threads.** `ulimit -u` (1024 on the cluster) counts every thread of the user. `ClaphaEngine` tells the engine it
  has 2 CPUs (70 → 25 threads) and `raise_thread_limit()` runs in the parent before any engine process starts.
- **CUDA and fork.** Anything that touches CUDA before forking (even `torch.cuda.is_available()` in an argparse
  default) breaks forked GPU processes; `il.rl` picks the device after forking and refuses to fork otherwise.
- **Slurm.** `qos_stu_cpu_long` is rejected (its association names partition `students`, the partition is
  `Students`); `tools/collectors.sbatch` uses `qos_p107-rtx5090` without a GPU.
- **Git on the cluster.** GitHub is often unreachable from campus; we push to the node over SSH instead
  (`git push cluster main` with `receive.denyCurrentBranch updateInstead` there).
- **OneDrive** flips file modes: `git config core.fileMode false`.

Tests: `python -m unittest il.test_il` (needs `data/cache-smoke`, see `build_cache --parts 0 --limit 100`;
the online-features test needs the emulator engine running).
