# Clash Royale imitation learning

Replays → native engine → compact games → imitation learning → live play on Null's Royale.

| File | What it does | Where it runs |
|---|---|---|
| `convert.py` | Replays the collected battles in the native engine and writes one archive per game (both players' inputs and labels) | Linux, Slurm |
| `train.py` | The policy network, its loss, the data loader and the training loop; inference helpers | GPU, Slurm |
| `play.py` | Boots the emulator, reads the game's memory, runs the model and taps | Mac |
| `common.py` | What all three share: the observation of one decision, legal placement, the clock and elixir timing | — |
| `arena.py` | Paired-seat native battles between two models | Linux, Slurm |
| `ppo.py` | Self-play PPO from the IL model: a DDP learner with a separate critic, and actor jobs (native games, batched GPU decisions) | GPU, Slurm |

`assets/` holds the game-build tables (card costs and rarities, effects, placement masks, replay card names, the
881-token vocabulary, the supported client's hashes). The device reader binary `assets/clapha-reader-nulls` is not in
the repository: get it from Arron and put it there before playing live. `test_pipeline.py` checks it all
without an engine, emulator or GPU.

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

`latest.pt` is saved every 1,000 steps. Running the same command again resumes it with the settings in
`settings.json`, so a job killed at its time limit loses at most 1,000 steps. A resumed run shuffles the data afresh
instead of replaying the interrupted order exactly. Progress goes to `metrics.jsonl`, with validation after each pass.
Don't change the code under a running job: copy the checkout for long runs.

## Play

```bash
python play.py ~/cr-live/models/full-compact-20261014/latest.pt --tables ~/cr-live/models/unit_state_tables_150535029.json
```

It boots the visible `Pixel_9_2` emulator if needed (`-gpu host`), verifies the game library, installs the reader,
re-applies the review-crash patch an emulator snapshot may have rolled back, and waits for a battle. Start a friendly
battle by hand, or add `--queue` to press Battle on the ladder. `--games N` plays several. Logs go to
`~/cr-live/runs/live` (OneDrive stalls writes). The model sees advance opponent commands from the reader's queue.
`--losses N` plays ladder games until the N-th loss. `python winbar.py` opens a blue/red bar of the model's predicted
win chance, (value + 1) / 2 from the policy's value head, following the newest live log.

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
`python build_features.py` builds optional CPython helpers for memory reads and unit features with the system C compiler;
build once per Python/platform. They preserve the existing observation arrays. `CR_NATIVE_FEATURES=0` selects Python
helpers at process startup. Without the extension, the Python path is used automatically.
Half the games are self-play, the rest against the IL model or recent snapshots (`RUN/pool/`), chosen by how often
they still beat the policy. Reward: the result (±1), a small penalty per decision at full elixir, and tower-damage
shaping. The critic is a separate network used only in training: the policy's network (from the IL weights) plus the
opponent's hand, next cards and ability, with a P(win) head trained on the outcome and a head for the shaping rewards.
The first 20 updates train the critic alone. Progress goes to `RUN/metrics.jsonl` (`score`, `win_ev`, `kl_ref`,
`games_per_hour`, ...). Rerunning the learner resumes from `RUN/latest.pt`; actors reconnect by themselves.
`scancel --name=cr-ppo` stops everything.

## Arena and tests

```bash
python arena.py A.pt B.pt --engine ~/clapha-engine --pool ~/cr-engine/data/deck_pool.json --games 20 --out runs/arena.jsonl
OMP_NUM_THREADS=2 python -m pytest test_pipeline.py
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
