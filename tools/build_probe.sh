#!/usr/bin/env bash
# Build the probe (vendor/firstlight/probe) into build/libcrprobe.so with the Android NDK.
# After testing it, copy it to prebuilt/ and refresh prebuilt/sha256.txt so git carries it.
set -euo pipefail
cd "$(dirname "$0")/.."
NDK="${NDK:-$HOME/Library/Android/sdk/ndk/30.0.16248370/toolchains/llvm/prebuilt/darwin-x86_64/bin}"
P=vendor/firstlight/probe
mkdir -p build
"$NDK/aarch64-linux-android24-clang++" -DCR_MINIMAL_CAPTURE=1 -DCR_RESIDENT_COMMANDS=1 \
  -std=c++17 -O2 -fPIC -fvisibility=hidden -Wall -Wextra -shared -Wl,-z,max-page-size=16384 \
  -o build/libcrprobe.so $P/cr_replay_probe.cpp $P/native_touch_interceptor.cpp \
  $P/native_call_arm64.S $P/remaining_runtime_arm64.S -llog -ldl -lz
shasum -a 256 build/libcrprobe.so
