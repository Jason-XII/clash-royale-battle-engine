# Clash Royale Battle Engine

这个仓库把皇室战争的原生战斗引擎（[clapha-engine](https://github.com/arronwang01/clapha-engine)：x86_64 Linux 上直接加载游戏的 `libg.so`）当成 API 来调用，用 PPO 自博弈训练 [clapha-il](https://github.com/ClashAlpha/clapha-il) 在 25 万场真人回放上模仿学习出的模型，最后在 Null's Royale 上实战。它源于对 [FirstLight-CR](https://gitlab.com/firstlight3/FirstLight_CR) 逆向工作的改进。

## Where things live

| What | Where |
|---|---|
| Self-play (PPO), arena, defense stats, the engine wrapper | this repo (`~/cr-engine` on the cluster; `git push cluster main`) |
| Model, inputs, replay conversion, imitation learning | clapha-il, branch `jason` (`~/clapha-il-dev` on the Mac, `~/clapha-il` on the cluster; `git push cluster jason`) |
| The engine (loader, per-build reads, version oracle) | clapha-engine (Arron, private; `~/clapha-engine`) |
| Legal tiles, replay converter | FirstLight's `native_runner` (`~/firstlight-cr`) |
| Replays, converted games, runs | `~/cr-engine/data/IL_Replay`, `~/oracle`, `~/il-raw`, `~/il-runs`, `~/cr-engine/runs` |

## The pipeline

| Stage | Code | Job |
|---|---|---|
| 1. Keep the replays the engine replays exactly (all six towers end as recorded, ~69.5%) | clapha-engine `crx/version_oracle.py` | `tools/oracle.sbatch` |
| 2. Convert them: the raw record (lean frame, objects, unit states) every 5 ticks | clapha-il `convert.py` | `tools/convert.sbatch` |
| 3. Imitation learning, then 400 arena games | clapha-il `train.py` | `tools/il.sbatch` |
| 4. Check a checkpoint in the engine: inputs = IL inputs, decision time, arena, 5 PPO updates | `tools/check.sbatch` | |
| 5. PPO self-play, autopilot | `il/rl.py`, `il/actors.py`, `il/policy.py`, `il/autopilot.py` | `tools/train.sbatch` + `tools/satellite.sbatch` |
| 6. Model vs model, defense | `il/arena.py`, `il/defense.py` | |
| 7. Live Null's Royale games | open: `il/live.py` plays only the old models (emulator probe); clapha-il models need the record live | |

The model's inputs are built by one function in every stage: clapha-il's `PlayerView` on the raw record (converted
replays in IL, `ClaphaEngine.record()` in self-play and the arena). `tools/check_clapha_inputs.py` proves a game played
here gives the same inputs as the same game converted.

```
__init__.py, engine.py   the State / Play every stage uses (and Engine: the emulator probe, old live path)
clapha.py                ClaphaEngine: clapha-engine in-process; reset / step with plays landing 23-27 ticks later; record()
il/policy.py             a clapha-il checkpoint playing: load, Inputs (PlayerView per player), packed rows, the policy
                         (what / which card / which tile), Agent
il/rl.py                 PPO learner: matchmaking (opponents, decks), the update, `il.rl join` (another job's collectors)
il/actors.py             collectors (games, reward shaping), decision servers (GPU, MPS)
il/arena.py              model vs model (clapha-il or old checkpoints); il/defense.py: how a player defends
il/autopilot.py          evaluates every PPO snapshot, keeps best.pt, rolls the learner back when it degrades
il/test_policy.py        packed inputs, log-probs, one PPO update (no engine needed)
tools/                   Slurm jobs; check_clapha_inputs.py, time_decision.py
old stack                il/data.py, il/model.py, il/live.py, build_cache.py, emulator.py, prebuilt/, vendor/: the
                         il-002 / ppo-ebarbs2 models, kept as arena baselines until the clapha-il pipeline beats them
```

## Commands

```bash
cd ~/cr-engine
sbatch tools/oracle.sbatch && sbatch tools/convert.sbatch                     # stages 1-2 (stu QoS, resumable)
RUN=pilot-base FLAGS="" sbatch -J pilot-base tools/il.sbatch                  # stage 3 (A100; FLAGS: train.py options)
CKPT=~/il-runs/pilot-base/latest.pt sbatch tools/check.sbatch                 # stage 4 (REF=<branch> checks a branch)
INIT=~/il-runs/<run>/latest.pt RUN=~/cr-engine/runs/ppo-general sbatch tools/train.sbatch   # stage 5: learner (A100)
RUN=~/cr-engine/runs/ppo-general sbatch tools/satellite.sbatch                # + collectors and autopilot (RTX 5090)
scancel --name=cr-ppo                                                         # stop both; a restart resumes latest.pt
PYTHONPATH=. python -m il.arena ~/clapha-engine A.pt B.pt --games 400         # stage 6, in a job
```

Never run engine games, profiling or training on the login node: in a job, or on the Mac for what needs no engine
(`CLAPHA_IL=~/clapha-il-dev python -m il.test_policy CHECKPOINT GAMES`).

## PPO

**Processes.** *Collectors* (one engine each) play games: both players' inputs from the record (`il.policy.Inputs`),
sent packed (present objects, int32 ids, legal tiles as bits: ~4 kB a decision); a play lands 23-27 ticks after the
decision (as humans' plays in the labels), a slightly unaffordable card waits for elixir as the game does, one play in
flight per player. *Decision servers* (`--servers`, GPU, MPS) answer batched decisions and keep each seat's LSTM state
in one GPU tensor; each frozen opponent lives on one server. The *learner* assigns games (decks, opponent, server), runs
PPO, publishes weights to the servers through shared memory, snapshots into the opponent pool. `--remote` and
`il.rl join` (the satellite job) add another job's collectors and servers.

**Policy** (`il/policy.py`): what to do, which hand card, which tile, each sampled from its head; only waiting while a
play is in flight; P(ability) >= 1% while one is ready. The arena plays the same policy.

**Algorithm.** Half the games self-play, half against a frozen pool (the IL model + snapshots) by PFSP, weight
(1 − win rate)² + 0.05. Reward: ±1 result, −0.002 per decision at full elixir, FirstLight's tower shaping β (γΦ′ − Φ)
(Φ = crowns taken − lost + each tower's lost hp fraction, king ×3; 0 once the game ends: damage dealt and prevented is
credited when it happens, no bonus for a crown margin). PPO (clip 0.2, γ 0.999, whole games as LSTM sequences), KL
penalty towards the IL policy (exact for what / card, the k3 estimate for the tile taken). Decks: real ladder pairs
with evolutions and levels (`data/deck_pool.json`). A specialist (`--learner-deck`) plays our deck against ladder
decks: 30% the most played, 40% any, 30% decks with the cards we score worst against.

**Autopilot** (in the satellite job): every snapshot plays the panel (the IL model, more with PANEL); one line per
snapshot in `<run>/autopilot.jsonl`; a better snapshot that does not defend or play less becomes `<run>/best.pt`; two
bad ones in a row write `rollback.json` (restart from best.pt at half the learning rate).

**Log** (`<run>/metrics.jsonl`, one line per update): `pool` (score vs each frozen opponent), `kl_ref`, `approx_kl`,
`clip_frac`, `value_ev_mc` / `value_ev_gae` (is the value an informative baseline?), `plays_per_game`, `full_elixir`,
`defense`, and where time goes: `collector_ms` (features / wait / engine), `server_ms_per_call`, `mean_batch`,
`games_per_hour`, `backlog`.

**Throughput.** CPUs are the limit: each QoS gives 16 (A100, RTX 5090) or 24 (stu) per user. The old model cost
~1 CPU-s a game (~67k games/h on two jobs); the clapha-il path ~1.4 CPU-s (engine 0.4, record 0.3, inputs 0.7;
`tools/time_decision.py`).

## Defense

`il/defense.py`: a push is enemy troops on our side of the river; it is answered when our play lands in its lane (or
the middle) before it ends: answered or not, latency from the crossing to our landing, tower hp lost. The same numbers
for humans (cache), arena games (`defense_a/b`), training (`defense` in metrics.jsonl) and live games.

| mid pushes (3–6k hp) | answered | latency | tower hp lost / push |
|---|---|---|---|
| humans (cache) | 91% | 0.4 s | 4.9% |
| humans with our E-Barbs deck | 88% | 0.4 s | 7.1% |
| il-002 with E-Barbs vs General | 90% | 1.6 s | 12.9% |
| specialist update 2709 vs General | 82% | 2.1 s | 13.0% |

Humans tap ~0.6 s before a push crosses; the old models decided 0.75–1.5 s after it. Deciding "play now?" by a coin
flip every 0.25 s on a diffuse P(play) gives geometric waits; clapha-il's `--next-play` head (when the next play
comes, on every decision) is the pilot testing whether a sharper timing fixes it. Judge pilots by `timing_gain` and
the arena's defense numbers.

## How we got here

1. **The engine as an API** (2026-10-07). After months on a from-scratch simulator whose quirks the agents learned, we
   switched to the real engine and human data, as FirstLight did (IL → General PPO → a specialist that reached Hall of
   Fame): FirstLight's probe in an emulator, then Arron's clapha-engine on the GPU nodes (no KVM there).
2. **The old pipeline** (il-002 → ppo-general → ppo-ebarbs2): a 0.9 M model on cached states, 9–3 live; PPO up to
   ~67k games/h. It plateaued (value_ev_mc 0.21–0.30) and raced instead of defending: answers ~1.5 s late, tower
   shaping and IL opponents did not fix it.
3. **The IL rebuild** (2026-10-13): Arron's clapha-il: only replays the engine replays exactly, the full record (unit
   states, effects, targets, 128 objects, legal tiles, plays in flight both sides, counted opponent cycle), one input
   function for IL, self-play and live. Pilots: base, no-attention, no-unit-states, next-play. PPO ported to it.

## Pitfalls (read before changing things)

- **Levels.** A bare match template plays everything at level 1. Matches need `lvlcap`, `cardlvlmin`,
  `avatar*.expLevel` **and the king tower `hbd[*].kt`** (a level-1 king: 2400 hp instead of 7728). `il.arena.make_match`
  takes care of it. Decks with evolutions come from `data/deck_pool.json`.
- **Threads.** `ulimit -u` counts every thread of the user. `ClaphaEngine` tells the engine it has 2 CPUs (70 → 25
  threads) and `raise_thread_limit()` runs in the parent before any engine process starts.
- **CUDA and fork.** Touching CUDA before forking (even `torch.cuda.is_available()`) breaks forked GPU processes;
  `il.rl` picks the device after forking.
- **A new head or a fresh Adam** collapsed a run within 15 updates: `il.rl` warms the learning rate up after any fresh
  Adam; watch the first ~15 updates after a restart that changes the model.
- **Slurm.** `qos_p107-a100` and `qos_p107-rtx5090`: 16 CPUs per user each; the stu route only as account `stu`,
  partition `Students`, `qos_stu_medium_2gpu` (24 CPUs, 1 day). anode02 cannot start CUDA (excluded in the jobs).
- **Git on the cluster.** GitHub is often unreachable from campus: push to the cluster over SSH (`updateInstead`).
- **OneDrive** flips file modes: `git config core.fileMode false`.

Docs: [how the live client takes a play](docs/GAME_INTEGRATION.md), [working with Arron](docs/COLLABORATION.md),
[the IL rebuild's handoff](docs/HANDOFF-2026-10-13.md); the model's inputs: clapha-il's docs/INPUTS.md.
