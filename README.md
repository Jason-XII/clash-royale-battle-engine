# Clash Royale Battle Engine

这个仓库允许你把皇室战争的原生战斗引擎当成 API 来调用，并在 25 万场真人回放上做模仿学习。开发过程中 agent 总结的经验蒸馏成了 3 个 skill，放在 `.agents/skills` 目录下。

本仓库是对 [FirstLight-CR](https://gitlab.com/firstlight3/FirstLight_CR) 逆向工作加以改进后的产物，极大简化了项目结构和通信接口。

The real `libg.so` battle engine runs inside a rooted ARM64 Android emulator, driven by the
`libcrprobe.so` probe (source in `vendor/firstlight/probe/`, binary in `prebuilt/`).

| Path | What |
|---|---|
| `engine.py`, `protocol.py` | `Engine`: reset any deck, step with plays, schedule replay commands, `run` many ticks in one round trip |
| `batch.py` | `Batch`: up to 16 resident battles per round trip (for PPO throughput) |
| `replay.py`, `build_cache.py` | rebuild IL_Replay games in the engine and cache raw states (Mac) |
| `il/` | features, model, imitation training, engine agent; needs only numpy + torch for training |
| `tools/` | `start_engine.sh`, `build_probe.sh`, `cputemp`, `heat_guard.sh` |
| `bootstrap_emulator.py`, `install_probe.sh`, `benchmark.py`, `status.py` | engine setup and checks |

## Engine (Mac)

Once per emulator install, put the probe into the prepared Nulls Royale APK. Then start the
engine headless (the script starts the emulator with `-gpu host` if needed; without a window the
default is CPU SwiftShader at ~8 cores) and run the regression check:

```bash
./install_probe.sh                    # prebuilt/libcrprobe.so; rebuild from source with tools/build_probe.sh
tools/start_engine.sh                 # headless emulator + bootstrap_emulator.py
python -m native_engine.benchmark     # must print result=1 crowns=(0, 1) final_tick=3835
```

`bootstrap_emulator.py` verifies `libg.so`, installs UID-scoped offline firewall rules, restores
the port-26789 forward, launches the app and waits for `contentReady`. It is safe to repeat.

`pip install -e .` makes `from native_engine import Engine, Play` work from notebooks
(`example.ipynb` drives the rendered engine).

On the fanless MacBook Air, guard long jobs: `tools/heat_guard.sh <pid> 70 62` pauses a job above
70 °C and resumes it below 62 °C (`tools/cputemp`, build: `clang -O2 -framework IOKit
-framework CoreFoundation -o tools/cputemp tools/cputemp.c`).

## Imitation learning

`data/` is a symlink to `~/cr-data` (outside OneDrive, not in git). IL_Replay (252,238 games)
comes from HuggingFace `VanguardX101/IL_Replay`.

1. **Rebuild replays (Mac, ~11 h, resumable).** Uses FirstLight's converter from
   `../firstlight-cr` for side flips, deal selection, forms and ability hints. About 91% of
   rebuilt games end with the same winner and crowns as the real game. If the emulator crashes,
   the build restarts it and retries the game.
   ```bash
   python -m native_engine.replay data/IL_Replay/replays/part-000000.parquet --count 20   # fidelity report
   nohup nice -n 10 python -m native_engine.build_cache data/IL_Replay/replays data/cache > data/cache/build.out 2>&1 &
   ```
   253 shards (`part-XXXXXX-K.npz`, 1000 games each, 17 GB in total) hold raw states every
   5 ticks, the human commands and fidelity flags, plus `vocab.json`. They hold no features, so
   the features can change without rebuilding.
2. **Train (GPU machine).** Clone this repo there and copy the cache next to it (it is not in git):
   ```bash
   rsync -a --progress ~/cr-data/cache/ <gpu-host>:<repo>/data/cache/
   python -m il.train data/cache runs/il-001 --epochs 4     # metrics.jsonl, best.pt, last.pt
   ```
   The last shard is held out for validation. Only games that rebuilt faithfully are used
   (`--keep-diverged` uses all of them). Use `--max-steps 300` first to measure speed.
3. **Play it in the engine (Mac).**
   ```bash
   python -m il.play runs/il-001/best.pt data/IL_Replay/replays/part-000051.parquet --games 10
   ```

Tests: `python -m unittest il.test_il`. It needs a smoke cache:
`python -m native_engine.build_cache data/IL_Replay/replays data/cache-smoke --parts 0 --limit 100`.
With the engine running, it also checks that the agent's online features equal the cached
training features.
