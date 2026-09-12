---
name: operate-rendered-battle-engine
description: Replay and inspect this repository's native battles through Nulls Royale's stock Android UI with exact ticks and compact observations. Use for rendered debugging and episode visualization; use the headless or throughput skills for training and benchmarks.
---

# Rendered Battle Engine

Work from the repository root, preserve unrelated changes, and use the project's configured Python interpreter.

## Choose the correct mode

Use `native_engine.RenderedEngine` for one visible battle. The stock UI owns a separate native battle manager and cannot attach to a resident `Batch` slot. To visualize a collected episode, create a rendered match with the same seed and replay its actions at their recorded ticks.

A lane has one mode. Close an active `Batch` connection and call `stop_resident(port)` before entering `native-render`. Keep the emulator visible and Nulls Royale in the foreground.

Use `build-and-run-headless-battle-engine` when the probe is missing, incompatible, or needs rebuilding. Do not duplicate its bootstrap and installation procedure here.

## Drive a visible battle

Call `RenderedEngine.reset(seed)` to create the stock scene, wait for `nativeRenderReady`, pause it, and return compact state. Call `step(actions, ticks)` to schedule actions for the next native tick, advance exactly `ticks`, pause, and return a `Transition`. Use `resume()` for real-time playback and `pause()` to recover synchronized compact state.

Check the returned tick delta and card-play events when validating an action. Inspect `deployment_remaining_ms` before treating a spawned troop as active. Use an emulator screenshot when the task requires visual confirmation.

## Interpret time correctly

The battle simulation runs at 20 Hz: one tick is 50 ms and `ticks=20` advances one simulated second. `ticks` is a duration, not a displayed match-clock timestamp.

The current wrapper returns reset state at tick 90. Treat this as an inherited Firstlight warm-up convention, not a proven gameplay mechanic. Native time already elapsed during those 90 ticks, so the UI clock is later than 2:59. Verify the earliest safe action tick before changing this boundary.

A play can be recorded before its troops are ready to move or attack. Separate the command's play tick from each entity's deployment countdown and from UI animation timing.

## Preserve native-render identity

When changing compact capture, resolve the active manager and generation through the mode-aware observation identity. Do not assume `g_controlled_manager`, which belongs to the singleton headless path. Keep resident isolation and stale-generation checks intact.

Prefer the rendered replay scheduler for exact application ticks; the low-level command encoding includes a native 20-tick age interval. Keep capture cursors tied to the rendered generation and verify changes in both compact state and the visible scene.
