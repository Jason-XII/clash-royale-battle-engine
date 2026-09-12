# Troubleshooting

## Compatibility checks block a matching engine

If full APK or runtime-resource checks fail while `libg.so` matches, do not weaken checks silently. Use the engine hash as the function-offset boundary for focused runtime experiments, then document resource-dependent failures separately.

## No Android target

Run `adb devices -l` before runtime work. Verify the selected serial, root access, ABI, Android version, installed package, and current PID.

## ADB cannot start its smart socket

This can be a sandbox restriction or a competing daemon. Use the already-running ADB server when possible. If a required operation fails because the sandbox cannot bind the smart socket, rerun that operation with the environment's approved escalation mechanism.

## `libc++_shared.so` is not found

Resolve the directory containing the installed `libg.so`. Load that directory's `libc++_shared.so` with `dlopen` and global visibility before loading the injected probe.

## Android refuses to map the probe from `/data/local/tmp`

Stop the app, place the probe beside the installed native libraries, set executable permissions, restore its SELinux context, and restart the app. Do not replace `libg.so`.

## Battle configuration succeeds but observation JSON is truncated

Check for ARM64 top-byte-tagged heap pointers. Strip the top byte only inside the general mapped-memory readability check used with `/proc/self/maps`. Do not alter libg code-address validation or actual object pointers.

## A probe operation fails after a mutating request

Do not replay an ambiguous configure, inject, or step request automatically. Treat the client state as invalid and reset the episode, because the native side may already have applied the request.
