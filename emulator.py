"""The Android emulator side of the engine (Mac): build and install the probe, start the headless emulator, bootstrap
the battle engine and check it.

    python -m native_engine.emulator build-probe      # vendor/firstlight/probe -> build/libcrprobe.so (Android NDK)
    python -m native_engine.emulator install-probe    # prebuilt/libcrprobe.so (or --so PATH) into the installed APK
    python -m native_engine.emulator start            # headless emulator (-gpu host), then bootstrap
    python -m native_engine.emulator bootstrap        # offline engine; --online for live Nulls games
    python -m native_engine.emulator benchmark        # must print result=1 crowns=(0, 1) final_tick=3835
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import socket
import subprocess
import time

PACKAGE = 'nullsroyale.rel.free'
ACTIVITY = f'{PACKAGE}/com.supercell.clashroyale.GameApp'
DEVICE_PORT = 26789
RULE_COMMENT = 'crack-cr-offline'
SDK = Path.home() / 'Library/Android/sdk'
ADB = str(SDK / 'platform-tools/adb')
NDK = SDK / 'ndk/30.0.16248370/toolchains/llvm/prebuilt/darwin-x86_64/bin'
REPO = Path(__file__).resolve().parent
PROBE_SHA256 = '2c6e10de7e80c0414e8c295224f9a1d9e0110debc2897b724ee48810ab57accf'  # prebuilt/libcrprobe.so


def adb(serial, *arguments, check=True):
    result = subprocess.run([ADB, '-s', serial, *map(str, arguments)], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f'adb {" ".join(map(str, arguments))} failed')
    return result


def shell(serial, *arguments, check=True):
    return adb(serial, 'shell', *arguments, check=check)


def app_directory(serial):
    paths = [line.removeprefix('package:') for line in shell(serial, 'pm', 'path', PACKAGE).stdout.splitlines()]
    base = next((p for p in paths if p.endswith('/base.apk')), None)
    if base is None:
        raise RuntimeError(f'{PACKAGE} is not installed')
    return PurePosixPath(base).parent


def replace_offline_rule(serial, table, uid, offline):
    listing = shell(serial, table, '-L', 'OUTPUT', '--line-numbers', '-n').stdout
    numbers = [int(line.split()[0]) for line in listing.splitlines() if RULE_COMMENT in line and line.split()[0].isdigit()]
    for number in reversed(numbers):
        shell(serial, table, '-D', 'OUTPUT', number)
    if offline:
        shell(serial, table, '-I', 'OUTPUT', '!', '-o', 'lo', '-m', 'owner', '--uid-owner', uid,
              '-m', 'comment', '--comment', RULE_COMMENT, '-j', 'REJECT')


def read_status(port):
    with socket.create_connection(('127.0.0.1', port), timeout=1) as connection:
        stream = connection.makefile('rb')
        connection.sendall(b'session-v1\n')
        if not json.loads(stream.readline()).get('ok'):
            raise RuntimeError('probe rejected the control session')
        connection.sendall(b'status\n')
        return json.loads(stream.readline())


def wait_until_ready(port, timeout):
    deadline, last_error = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            status = read_status(port)
            if status.get('contentReady'):
                return status
        except (OSError, ValueError) as error:
            last_error = error
        time.sleep(0.5)
    raise RuntimeError(f'battle engine did not become ready: {last_error or "content is not ready"}')


def bootstrap(serial, port, timeout, online):
    adb(serial, 'root')
    adb(serial, 'wait-for-device')
    abi = shell(serial, 'getprop', 'ro.product.cpu.abi').stdout.strip()
    if abi != 'arm64-v8a':
        raise RuntimeError(f'an ARM64 emulator is required, found {abi!r}')
    directory = app_directory(serial)
    if shell(serial, 'ls', directory / 'lib/arm64/libcrprobe.so', check=False).returncode:
        raise RuntimeError('libcrprobe.so is missing; run install-probe')
    want = json.loads((REPO / 'vendor/firstlight/supported_engine.json').read_text())['libg_sha256']
    got = (shell(serial, 'sha256sum', directory / 'lib/arm64/libg.so').stdout.split() or [''])[0]
    if got != want:
        raise RuntimeError(f'unsupported libg.so: {got}')
    uid = shell(serial, 'stat', '-c', '%u', f'/data/user/0/{PACKAGE}').stdout.strip()
    for table in ('iptables', 'ip6tables'):
        replace_offline_rule(serial, table, uid, not online)
    adb(serial, 'forward', f'tcp:{port}', f'tcp:{DEVICE_PORT}')
    shell(serial, 'am', 'start', '-n', ACTIVITY)
    status = wait_until_ready(port, timeout)
    if status.get('mode') != 'headless':
        print(f"Restarting the game to leave {status.get('mode')} mode; the current battle will end.")
        shell(serial, 'am', 'force-stop', PACKAGE)
        shell(serial, 'am', 'start', '-n', ACTIVITY)
        status = wait_until_ready(port, timeout)
    if status.get('mode') != 'headless':
        raise RuntimeError(f"expected a fresh headless runtime, found {status.get('mode')}")
    print(f'Battle engine ready at 127.0.0.1:{port} (device={serial} uid={uid} network={"on" if online else "off"})')


def start(serial, port, timeout):
    """-gpu host keeps rendering on the GPU (without a window the default is CPU SwiftShader, ~8 cores)."""
    if f'{serial}\tdevice' not in subprocess.run([ADB, 'devices'], capture_output=True, text=True).stdout:
        subprocess.run(['pkill', '-f', '[q]emu-system-aarch64'])
        log = open(Path.home() / 'cr-data/emulator.log', 'w')
        subprocess.Popen([str(SDK / 'emulator/emulator'), '-avd', 'Pixel_9_2', '-no-window', '-no-audio',
                          '-no-boot-anim', '-no-snapshot', '-gpu', 'host'], stdout=log, stderr=log,
                         start_new_session=True)
        adb(serial, 'wait-for-device')
        while shell(serial, 'getprop', 'sys.boot_completed', check=False).stdout.strip() != '1':
            time.sleep(3)
    bootstrap(serial, port, timeout, online=False)


def build_probe():
    source = REPO / 'vendor/firstlight/probe'
    (REPO / 'build').mkdir(exist_ok=True)
    subprocess.run([str(NDK / 'aarch64-linux-android24-clang++'), '-DCR_MINIMAL_CAPTURE=1', '-DCR_RESIDENT_COMMANDS=1',
                    '-std=c++17', '-O2', '-fPIC', '-fvisibility=hidden', '-Wall', '-Wextra', '-shared',
                    '-Wl,-z,max-page-size=16384', '-o', str(REPO / 'build/libcrprobe.so'),
                    *(str(source / f) for f in ('cr_replay_probe.cpp', 'native_touch_interceptor.cpp',
                                                'native_call_arm64.S', 'remaining_runtime_arm64.S')),
                    '-llog', '-ldl', '-lz'], check=True)
    print(REPO / 'build/libcrprobe.so')


def install_probe(serial, so):
    want = hashlib.sha256(Path(so).read_bytes()).hexdigest()
    adb(serial, 'root', check=False)
    adb(serial, 'wait-for-device')
    shell(serial, 'am', 'force-stop', PACKAGE)
    found = shell(serial, f'ls /data/app/*/{PACKAGE}*/lib/arm64/libcrprobe.so', check=False).stdout.split()
    if not found:
        raise RuntimeError('installed probe not found; the APK may need reinstalling')
    adb(serial, 'push', so, '/data/local/tmp/libcrprobe_new.so')
    shell(serial, f"cp /data/local/tmp/libcrprobe_new.so '{found[0]}' && sync")
    got = (shell(serial, 'sha256sum', found[0]).stdout.split() or [''])[0]
    if got != want:
        raise RuntimeError(f'install failed: {got} != {want}')
    print(f'installed {so} -> {found[0]} ({got})')


def random_play(randomizer, player):
    from .engine import Play
    affordable = [card for card in player.hand if card.cost * 10000 <= player.elixir]
    if not affordable:
        return None
    card = randomizer.choice(affordable)
    x = randomizer.randint(1000, 17000)
    if card.card >= 28000000:
        y = randomizer.randint(3000, 29000)
    else:
        y = randomizer.randint(7000, 14500) if player.owner == 0 else randomizer.randint(17500, 25000)
    return Play(player.owner, card.slot, x, y)


def benchmark():
    """One battle with random plays every ten ticks; a regression check of the probe and the engine."""
    from .engine import Engine
    randomizer, engine = random.Random(1), Engine()
    started = time.perf_counter()
    state = engine.reset(seed=1)
    rollout = time.perf_counter()
    while not state.finalized:
        plays = tuple(filter(None, (random_play(randomizer, p) for p in state.players)))
        state = engine.step(plays, ticks=min(10, 7200 - state.tick)).state
    seconds = time.perf_counter() - rollout
    print(f'result={state.result} crowns={state.crowns} final_tick={state.tick}')
    print(f'reset={rollout - started:.3f}s rollout={seconds:.3f}s ticks_per_second={(state.tick - 90) / seconds:.1f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build-probe', 'install-probe', 'start', 'bootstrap', 'benchmark'])
    parser.add_argument('--serial', default='emulator-5554')
    parser.add_argument('--port', type=int, default=DEVICE_PORT)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--online', action='store_true', help='bootstrap: allow network access (live games)')
    parser.add_argument('--so', default=str(REPO / 'prebuilt/libcrprobe.so'), help='install-probe: the probe')
    args = parser.parse_args()
    if args.command == 'build-probe':
        build_probe()
    elif args.command == 'install-probe':
        if args.so == str(REPO / 'prebuilt/libcrprobe.so'):
            assert hashlib.sha256(Path(args.so).read_bytes()).hexdigest() == PROBE_SHA256, 'prebuilt probe changed'
        install_probe(args.serial, args.so)
    elif args.command == 'start':
        start(args.serial, args.port, args.timeout)
    elif args.command == 'bootstrap':
        bootstrap(args.serial, args.port, args.timeout, args.online)
    else:
        benchmark()


if __name__ == '__main__':
    main()
