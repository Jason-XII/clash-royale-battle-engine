---
name: build-and-run-headless-battle-engine
description: Build, inject, validate, and simplify this repository's ARM64 Clash Royale battle engine using the matching Android libg.so and Firstlight-derived native harness. Use for headless-engine setup, compact capture, emulator bootstrap, or battle-runtime verification; do not use for ordinary Clash Royale gameplay or unrelated Android work.
---

# Headless Battle Engine

Work from the repository root and preserve existing code and unrelated changes. Use the project's configured Python interpreter. Keep new Python direct, small, and explicit.

## Establish compatibility

Verify the emulator with `adb devices -l`, root access, ARM64 ABI, package version, and installed `libg.so` hash. Treat the matching `libg.so` as the fixed-offset compatibility boundary. Investigate APK or resource differences only when they prevent initialization or battle execution.

Compile the probe with the Android NDK using C++17, optimization, PIC, and 16 KB maximum page alignment. Record the compiler, command, source hashes, output hash, and runtime evidence.

## Start the runtime

Keep the game process offline while using the harness. Scope IPv4 and IPv6 firewall rules to the installed game's current Android UID and preserve loopback traffic. Start Frida server, forward host port 26789, start the game, and obtain its current PID.

Load the game's `libc++_shared.so` globally before loading the probe. Load injected libraries from the installed native-library directory when Android prohibits executable mappings from `/data/local/tmp`. Invoke `JNI_OnLoad` through Frida and confirm the control server is ready.

## Validate behavior

Create a seeded detached match and verify reset, exact tick advancement, structured state, card deployment, action consumption, and the native terminal result. Compare compact captures with the full reference endpoint using identical seeded action traces. Test both owners and include spells and crowded states before making performance or fidelity claims.

Normalize ARM64 top-byte-tagged pointers only for comparisons with untagged `/proc/self/maps` ranges. Preserve exact pointers for actual engine access and exact libg segment validation.

## Keep the interface compact

Capture required fields directly in native code instead of producing rich diagnostic telemetry and filtering it on the host. Retain synchronized identity and ticks, towers, own cards and elixir, public entities, necessary public events, action receipts, and terminal state. Send static metadata once when practical.

Keep diagnostic/full capture available as a comparison path. Reject unsupported decks or mechanics, stale generations, event loss, overflow, and unsupported commands explicitly. Separate engine state, actor observations, reward inputs, and diagnostics so each consumer receives only its contract.

Before changing model dimensions, measure native stepping, capture and encoding, wire bytes, host decoding, tensorization, active versus padded capacity, model latency, rollout storage, and end-to-end decisions per second. Treat observation-schema changes as a new model version requiring retraining, conversion, or distillation.

Read [references/troubleshooting.md](references/troubleshooting.md) when injection, observation encoding, or emulator startup fails.
