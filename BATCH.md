# Resident batch data plane

Goal: raise engine throughput above the single-lane ~11k ticks/s, which is capped by
the **~0.66 ms adb round-trip per decision**, not by the engine (native ceiling is
~140–170k ticks/s).

## Result

Stable numbers on `emulator-5554` (`Pixel_9_2`, 2 GB), 10-tick decisions, realistic
affordable-random deploys, observations decoded:

| Mode | Ticks/s |
|---|---|
| Single lane (baseline `benchmark.py`) | ~11,500 |
| Resident batch, 16 battles, one round trip/step | **~26,000** |
| Resident batch, no actions (pure stepping) | ~56,000 |
| Resident batch, 50-tick decisions | ~60,000 |

Each batch round advances **16 battles** in one host request, so the adb latency is
paid once per policy step instead of once per battle.

## What changed

The installed probe is now a **strict superset** of the baseline minimal probe. All
baseline commands (`session-v1`, `status`, `configure`, `step`, `minimal`,
`minimal-transition`, `observe`, `inject`) behave identically; the batch commands are
additive.

Source changes (all in `vendor/firstlight/probe/`):
1. `cr_replay_probe.cpp` — under new `-DCR_RESIDENT_COMMANDS`, allow the `env …`,
   `multi-status`, `multi-stop`, `engine-status` commands through the minimal guard.
2. `cr_replay_probe.cpp` — add `build_minimal_batch_capture()`, which reuses
   `build_minimal_capture()` (the ~1 KB compact state) for the batch plane.
3. `resident_multimatch.inc` — in `handle_resident_batch_session`, when the CRTF
   training capture fails (always, in the minimal build, because the runtime
   telemetry hooks are not installed) fall back to the compact minimal capture
   instead of the ~44 KB full atomic observation.
4. `minimal_capture.inc` — `build_minimal_capture()` gained an optional
   `transmitted_next_out` parameter (unused by the batch, kept for the incremental
   cursor work).

## Build

```sh
NDK="$HOME/Library/Android/sdk/ndk/30.0.16248370/toolchains/llvm/prebuilt/darwin-x86_64/bin"
"$NDK/aarch64-linux-android24-clang++" \
  -DCR_MINIMAL_CAPTURE=1 -DCR_RESIDENT_COMMANDS=1 \
  -std=c++17 -O2 -fPIC -fvisibility=hidden -Wall -Wextra \
  -shared -Wl,-z,max-page-size=16384 \
  -o build/libcrbatch.so \
  vendor/firstlight/probe/cr_replay_probe.cpp \
  vendor/firstlight/probe/native_touch_interceptor.cpp \
  vendor/firstlight/probe/native_call_arm64.S \
  vendor/firstlight/probe/remaining_runtime_arm64.S \
  -llog -ldl -lz
```

Installed as `/data/app/.../lib/arm64/libcrprobe.so`
sha256 `9f1df7edf1bd7c1bbcd87695972116586b53868b52993eb3847d9dbc306f9780`.

## Wire protocol (resident batch)

Handshake: connect, send `batch-fast-v1\n`, read 12-byte `CRBH` + version + status +
slots.

Request: 20-byte header `CRBQ`, version u16, reserved u16, sequence u64, count u32.
Then per slot: 8-byte entry `slot u16, action_count u16, ticks u32`, followed by
`action_count` 20-byte actions `kind u8(=1 play), owner u8, execute_offset u16,
hand_index i32, x i32, y i32, reserved u32`.

Response: 20-byte header `CRBS`, version, status, sequence, count. Then per slot:
12-byte entry `slot u16, status u16, receipt_count u16, flags u16, payload_size u32`,
then `receipt_count * 40` receipt bytes, then `payload_size` bytes of JSON. The
payload is the `cr-minimal.v3` state (decodable by `native_engine.protocol.State`).
Each entity includes its native type, deployment timer, target ID, and attack cooldown.
Python adds the basic deck's fixed attack, movement, targeting, and spell traits.
Target and cooldown are `null` without an attack component; target is also `null`
while none is held.
Cooldown is rounded up to the next 50 ms simulation tick, with zero meaning ready.

Slots are created with `env <id> configure <match-json>` on a normal `session-v1`
connection (max 16 slots).

## Run

```sh
python -m native_engine.status                 # mode/occupancy for every lane
python -m native_engine.batch_benchmark 26790  # batch throughput on one lane
python -m native_engine.batch_demo 26790       # one battle via the batch client
python -m native_engine.minimal_demo 26790     # one battle via the single client
```

The benchmark discards one complete warm-up, runs five complete seeded samples,
and computes throughput from returned tick deltas. It reports attempted, queued,
and consumed card plays so action-heavy results cannot silently become no-op runs.

Lane ports must be given explicitly; 26789 is often the main app lane and does not
serve battles. The cluster lanes are 26790–26792.

## Modes and clients

A lane is in exactly one mode, and the client must match it.

| Mode | Client | Use |
|---|---|---|
| `legacy-headless` | `Engine` (`minimal_demo`) | one battle, demos, debugging |
| `resident-headless` | `Batch` (even `battles=1`) | many battles, training |
| `native-render` | `RenderedEngine` | one exactly stepped battle in the game UI |

`stop_resident(port)` returns a lane to `legacy-headless`; `Batch` re-enters resident
mode by itself, so neither direction needs a process restart. `status.py` reports
which mode each lane is in. Always `batch.close()` before `stop_resident(port)`: the
control server handles one client at a time.

`RenderedEngine.reset()` pauses at tick 90. Its `step()` animates an exact number
of ticks and pauses again; `resume()` releases the battle to run in real time.
The emulator must be visible and the game must be in the foreground.

```python
from native_engine import Play, RenderedEngine

engine = RenderedEngine()
state = engine.reset(seed=41)
transition = engine.step([Play(owner=0, slot=0, x=3500, y=13000)], ticks=20)
engine.resume()
```

The stock UI owns a separate battle manager, so it cannot attach to a resident
`Batch` slot. To inspect a collected episode exactly, create a `RenderedEngine`
with the same seed and replay that episode's actions at their recorded ticks.

## Rollback

`native_engine/restore_baseline.sh` restores the baseline minimal probe and was
re-verified after this change (contentReady + ~11k ticks/s). See `BASELINE.md`.

## Known limitations
- One batch round binds every slot, and `resident_bind_slot` resets the combat event
  epoch per switch; per-slot bind cost is the main overhead at small tick counts.
- Multiple Android engine processes only gave ~1.4x (the guest serializes).
- The compact capture is `cr-minimal.v3` JSON, not tensors; a binary encoder is the
  next step if JSON parsing ever becomes the bottleneck (currently it is ~1% of cost).
