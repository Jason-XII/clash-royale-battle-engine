# Troubleshooting: native engine throughput

Concrete failures from the resident-batch work, with the fix. Each entry says what
it looked like, because a wrong diagnosis here costs hours.

## Measurements and diagnosis

- **"The engine maxes out at ~11k ticks/s."**
  It does not; the control path does. A no-op `status` costs ~0.66 ms from the host
  over `adb forward` but ~0.35 ms on-device, and `step 1` is ~0.75 ms host vs
  ~0.10 ms on-device. Always time on-device before calling the engine slow.

- **Large chunks are wildly faster than small ones.**
  `step 3600` reaches ~140–170k ticks/s while `step 10` drops to ~13k. The fixed
  per-call cost dominates small steps. This is the signal to batch, not to tune the
  step itself.

- **Two lanes do not add throughput.**
  Two on-device clients each slow by ~2x (0.10 ms -> 0.20 ms), so aggregate stays
  flat. The guest serializes control calls. Do not spend time on more lanes or more
  CPUs until this is fixed.

- **"More RAM/CPU should scale it."**
  It mostly does not. With the workload latency-bound, guest `top` shows ~93% idle
  and the host ~50% idle. RAM matters only because a 2 GB guest evicts lanes; 4 GB
  removes that pressure. Adding CPU changes nothing measurable.

- **Single-lane reads look much lower than expected while other lanes run.**
  Background lanes still consume time. Force-stop or stop them, or measure one lane
  in isolation, before quoting a per-lane number.

- **Numbers are unstable in the first minute after a lane boots.**
  Warm-up. Wait ~30 s and take repeated samples; report medians.

## Batch protocol

- **Response parsing desyncs and reads garbage or times out mid-frame.**
  The batch response entry header is **12 bytes**, not 16:
  `slot u16, status u16, receipt_count u16, flags u16, payload_size u32`. Reading 16
  swallows the first 4 payload bytes. Read the size from
  `kResidentBatchResponseEntryBytes` rather than assuming.

- **Every slot payload is ~44 KB instead of ~1 KB.**
  The compact CRTF training capture failed and the probe silently fell back to the
  full atomic observation JSON. In the minimal build CRTF always fails because the
  runtime telemetry hooks are not installed. Add the compact
  `build_minimal_capture()` fallback and confirm the payload schema is
  `cr-minimal.v3`.

- **Compact capture works for a few rounds, then permanently falls back.**
  A persistent per-slot capture cursor goes stale: `resident_bind_slot` resets the
  combat event epoch on every slot switch and advances the epoch's first sequence.
  The saved cursor then violates `cursor < oldest` forever. Use `cursor=0` (capture
  from the current bind's epoch); each bind happens right before that slot's step,
  so this is exactly the current step's plays.

- **`unsupported command in minimal profile` for `env …`, `multi-status`, `render …`.**
  The minimal guard rejects anything outside status/observe/configure/step/inject.
  Add the resident commands behind `-DCR_RESIDENT_COMMANDS`; note `render on/off/status`
  are still unavailable in this build, so do not rely on `render off` for headroom.

- **`resident mode requires semantic headless mode`.**
  Resident mode only starts from the headless runner. Configure the lane normally
  first; do not try to start it from a native-render runner.

- **`could not prepare native tilemap` on `env <id> configure`.**
  Almost always guest memory pressure: the lane never finished content init. Raise
  AVD RAM or boot fewer lanes.

## Emulator and process lifecycle

- **The whole display is black and `screencap` is black on every screen.**
  A stale snapshot restored a broken GPU/display surface. `-qt-hide-window` alone is
  not the cause; cold-boot the AVD (`-no-snapshot`/`-no-snapshot-load`).

- **Engine lanes never reach `contentReady`.**
  The lane whose activity is foreground finishes content init; lanes backgrounded
  too early stall (`visibilityChanged ... newVisibility=false`, then
  `PlayerBase::~PlayerBase`). Start lanes **one at a time** and wait for
  `contentReady` before starting the next.

- **The control server wedges: `session-v1` times out on a lane that was fine.**
  The server handles one client at a time and an abandoned persistent session keeps
  it busy. Always close sockets (use context managers / `try/finally`); to recover,
  force-stop and relaunch the process.

- **A second probe cannot be injected.**
  Do not try to run two probes in one process; they fight over hooks and ports. Build
  the new behaviour as a superset of the one probe and swap the single
  `libcrprobe.so` file.

- **Port forwards vanish after `adb root` / emulator restart.**
  Re-run `adb forward tcp:<port> tcp:<port>` for every lane. Firewall rules also do
  not survive an emulator restart.

## Mode: single-battle (`Engine`) vs resident (`Batch`)

A lane is in exactly one mode at a time, and the client must match it. This causes
most "it worked yesterday" failures.

- **`ConnectionError: battle engine closed an incomplete response`.** Two causes:
  a port with no lane, or a resident lane driven by the single-battle client. Check
  the lane first:
  ```sh
  python -m native_engine.status            # when installed as native_engine
  python -m native_engine.status 26790      # one lane
  ```
  The single-battle client (`Engine`) needs `legacy-headless`; `Batch` needs
  `resident-headless`.

- **`resident mode requires env ID command prefixes`.** The lane is resident and the
  caller used plain `configure`/`step`. Either drive it with `Batch`, or return it to
  single-battle mode with `stop_resident(port)`.

- **`no battle engine responded on port <n>` / empty reply / JSON decode error.**
  That port is an `adb forward` with no engine behind it — adb accepts the
  connection and immediately closes it. Note the default port 26789 is often the
  main app lane, not a cluster lane; the live lanes are 26790–26792. Always name the
  port explicitly.

- **Switching modes.** `stop_resident(port)` clears the resident flag and returns the
  lane to `legacy-headless`. It does not by itself free the 16 live match objects
  (those die with the process); the mode flag is what blocks `Engine`. `Batch`
  re-enters resident mode by itself, so a lane can go back and forth without a
  restart.

- **Corruption when clients are mixed.** A `Batch` holds its `batch-fast-v1` socket
  open for its whole lifetime and the control server handles one client at a time.
  Call `batch.close()` before `stop_resident(port)`, and never interleave `Engine`
  calls into a live resident session. Two clients on one port will fight.

- **Reading `multi-status`.** `mode`, `capacity` (16), `occupied` (live matches),
  `boundEnvId` (active slot), `switches` (bind count), and per-slot `tick`/`ended`.
  Useful smell: `occupied=16` with every slot `ended: true` means leftover finished
  matches; `ticks` differing across slots just means uneven progress.

## Shell and environment

- **`UID: readonly variable` in bash.** `UID` is reserved; use another name such as
  `APPUID`.

- **`timeout: command not found` on macOS.** GNU `timeout` is absent. Use scoped
  paths, background jobs, or `gtimeout` if coreutils is installed, instead of
  blindly searching.

- **Firewall `-D` fails with "No chain/target/match by that name".** The rule
  includes the `! -o lo` and comment matches; delete by line number
  (`iptables -D OUTPUT <n>`) or repeat the full original spec.

- **`adb push` onto a loaded library seems to work but nothing changes.** Stop the
  app first; Android will not let you replace a mapped `.so` meaningfully. The
  installed probe is one file under `/data/app/.../lib/arm64/`.
