# Known-good baseline and rollback contract

This is the state we must always be able to return to while experimenting with the
probe. Everything below is verifiable in under two minutes.

## What "working" means
- Package `nullsroyale.rel.free` (the cluster APK) installed on `emulator-5554`,
  app data intact (`update/`, `save/`, `shared_prefs/` untouched).
- Installed native probe:
  `/data/app/.../nullsroyale.rel.free-*/lib/arm64/libcrprobe.so`
  sha256 `04fa085250598e20096dde4eb4ce4eb7cebe2b8408925c0c05e691e50047c8fe`
- `python -m native_engine.benchmark` reaches `contentReady` and reports
  ~11k ticks/s with `result=1 crowns=(0, 1) final_tick=3835`.

## The known-good probe is already saved
`build/known_good/libcrprobe.so` is byte-identical to the probe currently installed
in the APK. That is the rollback artifact, independent of the running emulator.

## Rollback (one command)
```sh
native_engine/restore_baseline.sh
```
It force-stops the game, copies the known-good probe back into the app's lib dir,
verifies the sha256, re-forwards `26789`, and relaunches the game.

This path was exercised for real: the installed probe was overwritten with the
known-good bytes, the sha256 re-verified, the game restarted, `contentReady` came
back true, and the benchmark ran at 11.5k ticks/s.

## Rules for experiments
1. Never modify the app's data directories.
2. The new probe must be a **strict superset** of the current one: every command the
   current client uses (`session-v1`, `status`, `configure`, `step N`,
   `minimal`, `minimal-transition`, `observe`, `inject`) must still work.
3. After every probe swap, run the regression check before doing anything else:
   ```sh
   python -m native_engine.benchmark
   ```
   If `contentReady` is false, the result/crowns/tick changed, or the command
   fails, run `restore_baseline.sh` and stop.
4. Keep the previous probe file in the app lib dir until the new one passes the
   regression check (optional extra copy, e.g. `libcrprobe.prev.so`).
5. The Python client (`native_engine/*.py`) and these docs live in the repo/OneDrive;
   a probe experiment never touches them.

## Stronger isolation (optional)
If we want zero risk to the current emulator, clone the AVD and experiment on the
clone. The current instance stays booted and working the whole time.
