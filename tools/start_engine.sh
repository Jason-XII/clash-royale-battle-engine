#!/usr/bin/env bash
# Start the headless emulator if it is not running, then bootstrap the battle engine.
# -gpu host keeps rendering on the GPU (without a window the default is CPU SwiftShader, ~8 cores).
set -euo pipefail
cd "$(dirname "$0")/.."
SDK="$HOME/Library/Android/sdk"
ADB="$SDK/platform-tools/adb"
if ! "$ADB" devices | grep -q "^emulator-5554[[:space:]]*device"; then
  pkill -f "[q]emu-system-aarch64" || true  # [q]: never match this shell
  nohup "$SDK/emulator/emulator" -avd Pixel_9_2 -no-window -no-audio -no-boot-anim -no-snapshot -gpu host \
    > "$HOME/cr-data/emulator.log" 2>&1 &
  "$ADB" wait-for-device
  until [ "$("$ADB" shell getprop sys.boot_completed | tr -d '\r')" = 1 ]; do sleep 3; done
fi
/opt/anaconda3/bin/python bootstrap_emulator.py
