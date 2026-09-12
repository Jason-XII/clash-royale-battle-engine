#!/usr/bin/env python3
"""Prepare a restarted Android emulator for the native battle engine."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import time


PACKAGE = "nullsroyale.rel.free"
ACTIVITY = f"{PACKAGE}/com.supercell.clashroyale.GameApp"
DEVICE_PORT = 26789
RULE_COMMENT = "crack-cr-offline"


def command(adb, serial, *arguments, check=True):
    result = subprocess.run(
        [adb, "-s", serial, *map(str, arguments)],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"adb command failed: {' '.join(map(str, arguments))}")
    return result


def shell(adb, serial, *arguments, check=True):
    return command(adb, serial, "shell", *arguments, check=check)


def app_directory(adb, serial):
    output = shell(adb, serial, "pm", "path", PACKAGE).stdout
    paths = [line.removeprefix("package:") for line in output.splitlines()]
    base = next((path for path in paths if path.endswith("/base.apk")), None)
    if base is None:
        raise RuntimeError(f"{PACKAGE} is not installed")
    return PurePosixPath(base).parent


def replace_offline_rule(adb, serial, table, uid):
    listing = shell(adb, serial, table, "-L", "OUTPUT", "--line-numbers", "-n").stdout
    numbers = [
        int(line.split()[0])
        for line in listing.splitlines()
        if RULE_COMMENT in line and line.split()[0].isdigit()
    ]
    for number in reversed(numbers):
        shell(adb, serial, table, "-D", "OUTPUT", number)
    shell(
        adb,
        serial,
        table,
        "-I",
        "OUTPUT",
        "!",
        "-o",
        "lo",
        "-m",
        "owner",
        "--uid-owner",
        uid,
        "-m",
        "comment",
        "--comment",
        RULE_COMMENT,
        "-j",
        "REJECT",
    )


def read_status(port):
    with socket.create_connection(("127.0.0.1", port), timeout=1) as connection:
        stream = connection.makefile("rb")
        connection.sendall(b"session-v1\n")
        greeting = json.loads(stream.readline())
        if not greeting.get("ok"):
            raise RuntimeError("probe rejected the control session")
        connection.sendall(b"status\n")
        return json.loads(stream.readline())


def wait_until_ready(port, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            status = read_status(port)
            if status.get("contentReady"):
                return status
        except (ConnectionError, OSError, ValueError) as error:
            last_error = error
        time.sleep(0.5)
    raise RuntimeError(f"battle engine did not become ready: {last_error or 'content is not ready'}")


def verify_engine(adb, serial, directory, repository):
    if shell(adb, serial, "ls", directory / "lib/arm64/libcrprobe.so", check=False).returncode:
        raise RuntimeError("libcrprobe.so is missing; install the prepared APK before bootstrapping")
    manifest = json.loads((repository / "vendor/firstlight/supported_engine.json").read_text())
    output = shell(adb, serial, "sha256sum", directory / "lib/arm64/libg.so").stdout
    installed_hash = output.split()[0] if output.split() else ""
    if installed_hash != manifest["libg_sha256"]:
        raise RuntimeError(f"unsupported libg.so: {installed_hash}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--port", type=int, default=DEVICE_PORT)
    parser.add_argument("--timeout", type=float, default=60)
    arguments = parser.parse_args()

    adb = shutil.which("adb")
    if adb is None:
        raise RuntimeError("adb is not on PATH")
    repository = Path(__file__).resolve().parent

    command(adb, arguments.serial, "root")
    command(adb, arguments.serial, "wait-for-device")
    abi = shell(adb, arguments.serial, "getprop", "ro.product.cpu.abi").stdout.strip()
    if abi != "arm64-v8a":
        raise RuntimeError(f"an ARM64 emulator is required, found {abi!r}")

    directory = app_directory(adb, arguments.serial)
    verify_engine(adb, arguments.serial, directory, repository)
    uid = shell(adb, arguments.serial, "stat", "-c", "%u", f"/data/user/0/{PACKAGE}").stdout.strip()
    replace_offline_rule(adb, arguments.serial, "iptables", uid)
    replace_offline_rule(adb, arguments.serial, "ip6tables", uid)
    command(adb, arguments.serial, "forward", f"tcp:{arguments.port}", f"tcp:{DEVICE_PORT}")
    shell(adb, arguments.serial, "am", "start", "-n", ACTIVITY)

    status = wait_until_ready(arguments.port, arguments.timeout)
    print(f"Battle engine ready at 127.0.0.1:{arguments.port}")
    print(f"device={arguments.serial} uid={uid} mode={status.get('mode')}")


if __name__ == "__main__":
    main()
