#!/usr/bin/env bash
# Replace the installed libcrprobe.so and restart the game. App data is untouched.
#
#   ./install_probe.sh                          # prebuilt/libcrprobe.so (any deck, replay scheduler, minimal-run)
#   ./install_probe.sh build/libcrprobe.so      # a fresh tools/build_probe.sh build
#
# Then run: python bootstrap_emulator.py
set -euo pipefail

ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
PKG=nullsroyale.rel.free
SO="${1:-$(dirname "$0")/prebuilt/libcrprobe.so}"
[ -f "$SO" ] || { echo "missing $SO"; exit 1; }

"$ADB" root >/dev/null 2>&1 || true
"$ADB" wait-for-device
"$ADB" shell am force-stop "$PKG"
PROBE=$("$ADB" shell "ls /data/app/*/$PKG*/lib/arm64/libcrprobe.so" | tr -d '\r' | head -1)
[ -n "$PROBE" ] || { echo "installed probe not found; the APK may need reinstalling"; exit 1; }

"$ADB" push "$SO" /data/local/tmp/libcrprobe_new.so >/dev/null
"$ADB" shell "cp /data/local/tmp/libcrprobe_new.so '$PROBE' && sync"
WANT=$(shasum -a 256 "$SO" | awk '{print $1}')
GOT=$("$ADB" shell "sha256sum '$PROBE'" | awk '{print $1}' | tr -d '\r')
[ "$GOT" = "$WANT" ] || { echo "INSTALL FAILED: $GOT != $WANT"; exit 1; }
echo "installed $SO -> $PROBE ($GOT)"
