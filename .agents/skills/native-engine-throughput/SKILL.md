---
name: native-engine-throughput
description: Measure and raise this repository's headless battle-engine throughput by separating host round-trip latency from native stepping, then batching resident battles through one probe connection. Use for ticks-per-second optimization, resident batching, or compact batch capture; use the headless-engine skill for initial injection and fidelity work.
---

# Native Engine Throughput

Raising ticks/s means finding whether the limit is native stepping, host round-trip
latency, or guest serialization — then moving that limit. Measure before changing
anything; the wrong bottleneck is easy to "optimize" forever.

Work from the repository root with the project's configured Python interpreter.
The runtime client is in the root Python modules and the probe is under
`vendor/firstlight/probe/`. Measured numbers: single lane ~11k ticks/s
(`python -m native_engine.benchmark`), resident batch of 16 ~26-32k ticks/s, and
`minimal-run` (a whole replay in one round trip) ~0.15 s per 3-minute game.

## 1. Separate native cost from round-trip cost

Run these on a lane with content ready (see the headless-engine skill):

- Time `step 1`, `step 10`, `step 100`, `step 3600` in chunks. The engine is not
  the limit if large chunks reach ~140k+ ticks/s while small chunks collapse.
- Time a no-op like `status` from the host over `adb forward`. If it costs about
  the same as a real step, the cost is the transport, not the engine.
- Time the same calls **on-device** with a tiny pushed C client over loopback.
  Host-vs-device difference is the adb round-trip (typically ~0.65 ms).
- Run two on-device clients at once. If each per-call time doubles, the guest
  serializes control calls and adding lanes will not scale.
- While batching, check `adb shell top` and host `top`. If guest and host CPUs are
  mostly idle, the workload is latency-bound; more cores will not help.

Only after this should you decide the lever.

## 2. The lever: fewer host round trips per tick

The decision cadence is fixed by the policy (e.g. 10 ticks). The only scalable
change is to advance **many battles per host request**. Firstlight's probe already
contains this as the resident multi-match plane (`resident`/`env` commands plus the
`batch-fast-v1` binary session), with up to 16 slots per process.

Expected on the current 2–4 GB AVD, 16 slots, 10-tick decisions, realistic actions:
~26k–32k ticks/s, versus ~11.5k for the single-lane client.

## 3. Enable the batch plane as a strict superset

The shipped probe is the **minimal** build, and its command guard rejects
`env …`, `multi-status`, and `multi-stop`. Do not loosen the guard globally; add a
flag and keep every existing command working.

1. Add `-DCR_RESIDENT_COMMANDS` and allow only the resident commands through the
   minimal guard.
2. The stock CRTF training capture always fails in the minimal build (the runtime
   telemetry hooks are not installed) and silently falls back to a ~44 KB full
   atomic observation per slot. Add a fallback that uses the existing
   `build_minimal_capture()` (~1 KB `cr-minimal.v3` JSON) instead.
3. Build with both `-DCR_MINIMAL_CAPTURE=1 -DCR_RESIDENT_COMMANDS=1`, keeping the
   baseline flags (`-std=c++17 -O2 -fPIC -fvisibility=hidden -Wall -Wextra`,
   `-Wl,-z,max-page-size=16384`). The full command is `tools/build_probe.sh`.
4. Verify the batch payload is the compact schema and stays near 1 KB/slot; a jump
   back to tens of KB means the compact path failed and fell back.

## 4. Protocol essentials

- Configure slots on a `session-v1` connection: `env <id> configure <match-json>`.
- Open the data plane with `batch-fast-v1`; read the 12-byte `CRBH` handshake.
- Request: 20-byte `CRBQ` header (version, reserved, sequence, count) then per slot
  an 8-byte entry `slot, action_count, ticks` followed by 20-byte actions
  `kind=1, owner, execute_offset, hand_index i32, x i32, y i32, reserved`.
- Response: 20-byte `CRBS` header then per slot a **12-byte** entry
  `slot, status, receipt_count, flags, payload_size u32`, then
  `receipt_count*40` receipt bytes, then `payload_size` payload bytes.
- Read struct sizes from the source constants; do not infer them from field layout.

## 5. Drive it: modes, clients, and status

A lane is in exactly one mode at a time, and the client must match it:

| Mode | Client | Use |
|---|---|---|
| `legacy-headless` | `Engine` | one battle, replay rebuilding, debugging |
| `resident-headless` | `Batch` (even `battles=1`) | many battles, training |

- Check before guessing: `python -m native_engine.status [port ...]` prints mode,
  occupancy, bound slot, and tick span for every lane.
- `stop_resident(port)` returns a lane to `legacy-headless`; `Batch` re-enters
  resident mode by itself, so no restart is needed either way.
- Call `batch.close()` **before** `stop_resident(port)`. A `Batch` holds its socket
  for its whole life and the control server serves one client at a time.
- Pass the lane port explicitly. 26789 is often the main app lane; cluster lanes are
  26790+.

## 6. Validate, then decide

- After every probe swap, run `python -m native_engine.benchmark` and require the
  same result, crowns, and final tick before trusting any new number.
- Benchmark with `python -m native_engine.batch_benchmark`; it decodes each slot's
  observation and sends affordable, in-bounds deploys so the cadence is realistic.
- Discard warm-up runs. Numbers drift for the first ~30 s after a lane boots; sample
  repeatedly and compare medians, not single runs.
- When comparing configurations, keep only the lanes under test running; idle
  background lanes steal time and depress the apparent single-lane rate.

## 7. Rollback is part of the work

Keep the verified probe bytes and checksum in `prebuilt/`. Re-verify the rollback
procedure after each change. The installed probe is one file under
`/data/app/.../lib/arm64/`; never touch app data. If a new probe fails the
regression gate, restore and stop.

## 8. What does not scale

Adding RAM removes guest memory pressure (2 GB evicts lanes; 4 GB keeps them) and
gives a modest single-lane gain, but it does not multiply throughput. Adding CPU,
lanes, or process count does not help while the workload is latency-bound and the
host/guest CPUs are idle. It also will not move the single-lane number, which is
set by adb round-trip latency.

Read [references/troubleshooting.md](references/troubleshooting.md) for the concrete
failures hit during this work and their fixes.
