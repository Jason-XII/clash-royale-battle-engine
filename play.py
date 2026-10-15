"""Play live Null's Royale games with a trained model on the visible Android emulator.

    python play.py ~/cr-live/models/full-compact-20261014/latest.pt --tables ~/cr-live/models/unit_state_tables_150535029.json

Boots the Pixel_9_2 emulator if it is not running, checks the game build, installs the read-only reader and restores the
review-crash patch (an emulator snapshot rollback can undo both), then waits for a battle: start a friendly battle by
hand, or pass --queue to press Battle on the ladder. Logs go to ~/cr-live/runs/live (outside OneDrive: it stalls writes).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile
import time

import torch

import common as C
import train

SDK = Path(os.environ.get('ANDROID_SDK_ROOT', Path.home() / 'Library/Android/sdk'))
ADB = shutil.which('adb') or str(SDK / 'platform-tools/adb')
PACKAGE = 'nullsroyale.rel.free'
READER = '/data/local/tmp/clapha-reader-nulls'
MANAGER_RVA, ROOT_CONTEXT = 0x19BD358, 0x20  # the Nulls build's (clapha-console nulls/README.md)
GET_MANAGER_GLOBAL = 0x19BBDD8  # libg global getManager() returns (0xBB6508); battle = [[manager+0x248]+0x90]
ACCOUNT = 0x043EC2F6  # the emulator's Nulls account (roster entry of our side, read 2026-10-14)
FRAME_MS = 50
FIRST_DECISION = 90  # as in imitation learning; taps earlier are not taken anyway
STALL_S = 2.0  # the tick standing still this long once it has moved: the battle is over (the results screen keeps the
               # battle object; it also stands still for a few seconds before the start)

# Pixel_9_2 emulator screen (1080x2424): the 30 inner arena rows span y 295..1714, so 32 rows span one row more at each end.
X0, X1, Y0, Y1 = 16, 1053, 295, 1714
TILE_W, TILE_H = (X1 - X0) / 18, (Y1 - Y0) / 30
BOTTOM = Y1 + TILE_H
BUTTON = (440, 2035)  # home screen Battle (yellow) and results screen OK (blue) overlap here, below their labels
ROAD_OK = (539, 2345)  # OK at the bottom of the trophy road page after a new arena
ABILITY_BUTTON = (973, 1954)  # hero / champion ability, bottom right above the hand


# ---- the emulator -------------------------------------------------------------------------------------------------------
def adb(serial, *arguments, check=True):
    result = subprocess.run([ADB, '-s', serial, *map(str, arguments)], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'adb command failed')
    return result


def disable_reviews(serial, library, supported):
    """Route this exact client's broken review request through its 'unavailable' callback (one instruction)."""
    patch = supported['live_review_patch']
    with tempfile.TemporaryDirectory(prefix='cr-review-') as directory:
        local = Path(directory) / 'libg.so'
        adb(serial, 'pull', library, local)
        original = local.read_bytes()
        actual = hashlib.sha256(original).hexdigest()
        if actual == patch['libg_sha256']:
            return
        if actual != supported['libg_sha256']:
            raise RuntimeError(f'refusing to patch unknown game library: {actual}')
        offset = int(patch['file_offset'], 0)
        before, after = bytes.fromhex(patch['before']), bytes.fromhex(patch['after'])
        if original[offset:offset + len(before)] != before:
            raise RuntimeError('review instruction does not match the supported client')
        modified = original[:offset] + after + original[offset + len(before):]
        if hashlib.sha256(modified).hexdigest() != patch['libg_sha256']:
            raise RuntimeError('review patch hash does not match')
        local.write_bytes(modified)
        backup = '/data/local/tmp/nulls-libg-review-enabled.so'
        adb(serial, 'shell', 'cp', '-p', library, backup)
        if adb(serial, 'shell', 'sha256sum', backup).stdout.split()[0] != supported['libg_sha256']:
            raise RuntimeError('original library backup verification failed')
        adb(serial, 'shell', 'am', 'force-stop', PACKAGE)
        try:
            adb(serial, 'push', local, library)
            adb(serial, 'shell', 'chmod', '755', library)
            adb(serial, 'shell', 'restorecon', library)
            if adb(serial, 'shell', 'sha256sum', library).stdout.split()[0] != patch['libg_sha256']:
                raise RuntimeError('review patch verification failed')
        except Exception:
            adb(serial, 'shell', 'cp', '-p', backup, library)
            raise
    print(f'Reviews disabled; original library saved at {backup}')


def prepare(serial='emulator-5554', avd='Pixel_9_2', timeout=90):
    """Boot the visible emulator if needed (never touching other emulators), check the game, install the reader and the
    review patch, start the game. Never queues a match."""
    devices = subprocess.run([ADB, 'devices'], capture_output=True, text=True, check=True).stdout
    if f'{serial}\tdevice' not in devices:
        log = Path.home() / 'cr-live/emulator.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:  # -gpu host: SwiftShader is far more expensive on the fanless Mac
            subprocess.Popen([str(SDK / 'emulator/emulator'), '-avd', avd, '-no-audio', '-no-boot-anim',
                              '-no-snapshot', '-gpu', 'host'], stdout=stream, stderr=stream, start_new_session=True)
    deadline = time.monotonic() + timeout
    while adb(serial, 'shell', 'getprop', 'sys.boot_completed', check=False).stdout.strip() != '1':
        if time.monotonic() > deadline:
            raise TimeoutError(f'{serial} did not boot within {timeout}s')
        time.sleep(1)
    adb(serial, 'root')
    adb(serial, 'wait-for-device')
    if adb(serial, 'shell', 'getprop', 'ro.product.cpu.abi').stdout.strip() != 'arm64-v8a':
        raise RuntimeError('live play requires an ARM64 emulator')
    apks = adb(serial, 'shell', 'pm', 'path', PACKAGE).stdout.splitlines()
    base = next((p.removeprefix('package:') for p in apks if p.endswith('/base.apk')), None)
    if base is None:
        raise RuntimeError(f'{PACKAGE} is not installed')
    library = PurePosixPath(base).parent / 'lib/arm64/libg.so'
    supported = json.loads((C.ASSETS / 'engine.json').read_text())
    actual = adb(serial, 'shell', 'sha256sum', library).stdout.split()[0]
    if actual not in (supported['libg_sha256'], supported['live_review_patch']['libg_sha256']):
        raise RuntimeError(f'unsupported game library: {actual}')
    if actual != supported['live_review_patch']['libg_sha256']:
        disable_reviews(serial, library, supported)
    for table in ('iptables', 'ip6tables'):  # remove only the old project's offline rules; ladder needs networking
        lines = adb(serial, 'shell', table, '-L', 'OUTPUT', '--line-numbers', '-n').stdout.splitlines()
        for rule in reversed([int(l.split()[0]) for l in lines if 'crack-cr-offline' in l and l.split()[0].isdigit()]):
            adb(serial, 'shell', table, '-D', 'OUTPUT', rule)
    reader = C.ASSETS / 'clapha-reader-nulls'
    expected = hashlib.sha256(reader.read_bytes()).hexdigest()
    installed = adb(serial, 'shell', 'sha256sum', READER, check=False).stdout.split()
    if not installed or installed[0] != expected:
        adb(serial, 'push', reader, READER)
        if adb(serial, 'shell', 'sha256sum', READER).stdout.split()[0] != expected:
            raise RuntimeError('installed live reader failed hash verification')
    adb(serial, 'shell', 'chmod', '755', READER)
    adb(serial, 'shell', 'am', 'start', '-n', f'{PACKAGE}/com.supercell.clashroyale.GameApp')


def screen(serial):
    """Which screen is up, by a few pixels: 'home' (Battle, yellow at BUTTON), 'results' (OK, blue at BUTTON), 'road' (the
    trophy road page after a battle: light blue at both ends of ROAD_OK), or None (matchmaking, loading, battle)."""
    raw = subprocess.run([ADB, '-s', serial, 'exec-out', 'screencap'], capture_output=True).stdout
    w, h = struct.unpack('<II', raw[:8])

    def pixel(x, y):
        i = len(raw) - w * h * 4 + (y * w + x) * 4
        return raw[i:i + 3]
    r, g, b = pixel(*BUTTON)
    if r > 200 and g > 130 and b < 80:
        return 'home'
    if b > 180 and r < 120:
        return 'results'
    if all(r < 120 and 150 <= g <= 210 and b > 230 for r, g, b in (pixel(470, ROAD_OK[1]), pixel(610, ROAD_OK[1]))):
        return 'road'
    return None


def world_screen(x, y, owner):
    """Native world units -> screen pixels. Our side is at the bottom; owner 0 sees x mirrored."""
    vx, vy = (18 - x / 1000, y / 1000) if owner == 0 else (x / 1000, 32 - y / 1000)
    return round(X0 + TILE_W * vx), round(BOTTOM - TILE_H * vy)


class Taps:
    """One persistent adb shell; each play is `tap card && tap tile`, acknowledged."""

    def __init__(self, serial):
        self.shell = subprocess.Popen([ADB, '-s', serial, 'shell'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      text=True, bufsize=1)

    def run(self, command):
        self.shell.stdin.write(f'{command}; echo done:$?\n')
        self.shell.stdin.flush()
        line = self.shell.stdout.readline().strip()
        if line != 'done:0':
            raise RuntimeError(f'adb failed: {command!r} -> {line!r}')

    def tap(self, x, y):
        self.run(f'input tap {x} {y}')

    def play(self, slot, x, y, owner):
        (sx, sy), (tx, ty) = (335 + 200 * slot, 2220), world_screen(x, y, owner)
        self.run(f'input tap {sx} {sy} && input tap {tx} {ty}')
        return tx, ty


def frames(serial):
    """The reader's frames, for good: it keeps reading across battles and menus, and starts again when the game restarts
    (a new process; a battle in progress then is lost)."""
    command = f'{READER} $(pidof {PACKAGE}) {FRAME_MS} {MANAGER_RVA:#x} {ROOT_CONTEXT:#x} --unified 0'
    while True:
        reader = subprocess.Popen([ADB, '-s', serial, 'shell', command], stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in reader.stdout:
            if line.startswith('{'):
                yield json.loads(line)
        print(f'the reader stopped (exit {reader.wait()}; is the game running?); starting it again', flush=True)
        time.sleep(5)


def roster(serial):
    """The battle's account ids in roster order (= side), read-only from /proc/<pid>/mem (adb root): battle+0xa8 is the
    roster, its count at +0x60, its accounts at +0x30 + 8 i, each account's id the u32 at +4."""
    def sh(command):
        return subprocess.run([ADB, '-s', serial, 'shell', command], capture_output=True, text=True, check=True).stdout
    pid = sh(f'pidof {PACKAGE}').split()[0]
    base = int(next(l for l in sh(f'cat /proc/{pid}/maps').splitlines() if 'libg.so' in l).split('-')[0], 16)

    def read(address, size):
        address &= 0x00FFFFFFFFFFFFFF  # Android heap pointers carry a tag in the top byte
        words = sh(f'dd if=/proc/{pid}/mem bs=1 skip={address} count={size} 2>/dev/null | od -An -tx1').split()
        return bytes(int(w, 16) for w in words)

    def pointer(address):
        return struct.unpack('<Q', read(address, 8))[0]
    battle = pointer(pointer(pointer(base + GET_MANAGER_GLOBAL) + 0x248) + 0x90)
    entries = pointer(battle + 0xa8)
    count = struct.unpack('<i', read(entries + 0x60, 4))[0]
    return [struct.unpack('<I', read(pointer(entries + 0x30 + 8 * i) + 4, 4))[0] for i in range(count)]


# ---- reader frames -> the raw decision records PlayerView reads -------------------------------------------------------------
# Plays an evolution needs, by card (evolutionRuntime's cycleRequired in the August build's converted replays). Static game
# data: the live reader reads only the progress.
CYCLE_REQUIRED = {
    26000000: 2, 26000001: 2, 26000004: 1, 26000007: 1, 26000008: 1, 26000010: 2, 26000011: 2, 26000012: 2, 26000013: 2,
    26000014: 2, 26000015: 2, 26000017: 1, 26000022: 1, 26000024: 1, 26000026: 2, 26000030: 2, 26000035: 2, 26000036: 2,
    26000037: 2, 26000040: 2, 26000043: 1, 26000044: 2, 26000045: 1, 26000047: 1, 26000049: 2, 26000050: 2, 26000055: 1,
    26000056: 2, 26000058: 2, 26000059: 2, 26000060: 1, 26000063: 1, 26000064: 2, 27000000: 2, 27000002: 2, 27000006: 2,
    27000010: 2, 27000012: 2, 27000013: 2, 28000004: 2, 28000008: 2, 28000017: 2}
EVOLUTION = 1  # deck_form_flags: 1 evolution, 2 hero
STATE_NAMES = {0: 'idle', 1: 'moving', 2: 'attacking', 3: 'dashing', 4: 'deploying', 5: 'river_jump', 6: 'underground',
               7: 'ability_underground', 10: 'ability_cast', 11: 'waiting_to_appear', 12: 'pulled',
               13: 'reeling_to_building', 14: 'hooking', 16: 'deflecting'}


class LiveRecords:
    """Reader frames -> the decision records converted replays hold (lean frame, objects(), unit_states() as
    clapha-engine's crx/observe.py builds them), using the build's exported unit-state tables."""

    def __init__(self, tables):
        data = json.loads(Path(tables).read_text())
        if str(data.get('build')) != '150535029' or not data.get('characters') or not data.get('effects'):
            raise ValueError('live play requires populated unit-state tables exported from build 150535029')
        self.effects = {int(k): v for k, v in data['effects'].items()}
        self.characters = {int(k): v for k, v in data['characters'].items()}
        self.ability_buffs = set(data['abilityBuffs'])
        self.last = {}  # unit address -> (use count, use tick): time since an ability use

    def reset(self):
        self.last = {}

    def __call__(self, frame):
        entities = frame['entities']
        ids = {e['address']: (e['side'], e['category']) for e in entities}
        units = self.unit_states(frame)
        players = sorted(frame['players'], key=lambda p: p['side'])
        return {'tick': frame['game_tick'],
                'lean': {'state': {'players': [_player(p) for p in players]}, 'players': [_lean(p) for p in players]},
                'objects': [_object(e, ids) for e in entities],
                'units': [units.get(e['address'], {'nativeObjectId': e['category'], 'kind': 'other'}) for e in entities]}

    def unit_states(self, frame):
        tick = int(frame.get('game_tick') or 0)
        by_address = {}
        controllers = {}     # unit id -> its hero / champion controller (a Clone's copy has none)
        for player in frame.get('players') or ():
            for c in player.get('abilities') or ():
                for unit in c.get('units') or ():
                    if unit >= 0:
                        controllers[int(unit)] = c
        for e in frame.get('entities') or ():
            us = e.get('us')
            if us is None:
                continue
            oid = int(e.get('category') or 0)
            if e.get('kind') == 0 and 'aim' in us:
                row = {'nativeObjectId': oid, 'kind': 'projectile', 'sourceObjectId': us['source'], 'x': e['x'], 'y': e['y'],
                       'z': us['z'], 'dx': us['dx'], 'dy': us['dy'], 'aimX': us['aim'][0], 'aimY': us['aim'][1],
                       'aimObjectId': us['aim_object'] if us['aim_object'] >= 0 else None,
                       'startX': us['start'][0], 'startY': us['start'][1]}
            elif e.get('kind') == 0:
                row = {'nativeObjectId': oid, 'kind': 'area', 'remainingMs': us['area_ms'], 'durationMs': us['area_life']}
            elif int(e.get('hp', -1)) < 0:
                # no hitpoints: a death bomb's fuse (clapha-engine kind 'timed')
                row = {'nativeObjectId': oid, 'kind': 'timed', 'fuseMs': int(e.get('deploy_remaining') or 0)}
            else:
                row = self._unit(e, us, oid, int(e.get('data_id') or 0) & 0xFFFFFFFF, controllers, tick)
            by_address[e['address']] = row
        return by_address

    def _unit(self, e, us, oid, data_id, controllers, tick):
        state = int(e.get('behavior_state_raw') or 0)
        character = self.characters.get(data_id, {})
        buffs = []
        for effect_id, left, elapsed, side, speed, hit, source in us.get('buffs') or ():
            name = self.effects.get(effect_id & 0xFFFFFFFF, {}).get('name', str(effect_id))
            buffs.append({'name': name, 'globalId': effect_id & 0xFFFFFFFF, 'remainingMs': left, 'elapsedMs': elapsed,
                          'sourceOwner': side, 'sourceObjectId': source if source >= 0 else None,
                          'hitSpeedPercent': hit, 'speedPercent': speed,
                          'stopsMovement': speed <= -100, 'stopsAttacks': hit <= -100})
        row = {'nativeObjectId': oid, 'kind': 'unit', 'name': character.get('name', ''), 'state': state,
               'stateName': STATE_NAMES.get(state), 'hidden': bool(us['hidden']), 'immune': bool(us['immune']),
               'invisible': bool(us['invisible']), 'airborne': bool(us['airborne']), 'buffs': buffs, 'attack': None,
               'charge': int(e['charge']) if e.get('has_move') else None, 'pushed': bool(us['pushed']),
               'pushRemainingMs': int(us['push_ms']), 'shield': int(us['shield'][0]), 'shieldMax': int(us['shield'][1]),
               'z': int(us.get('z', 0)), 'jumpRemainingMs': int(us.get('jump_ms', 0)),
               'invisibleAgainMs': int(us.get('invisible_again_ms', 0)),
               'surfaceAt': tuple(us['surface']) if state in (6, 7) and us.get('surface', [-1])[0] >= 0 else None}
        if 5000000 <= oid < 5000006 and 'King' in row['name']:
            row['kingAwake'] = int(e.get('kind') or 0) != 12      # kind 12 asleep, 13 awake (clapha-engine observe.py)
        elif us.get('daggers', -1) >= 0:
            row['daggers'] = int(us['daggers'])                   # Dagger Duchess's daggers left
        if character.get('spawns') and us.get('spawn_ms', -1) >= 0:
            row['spawns'] = character['spawns']
            row['spawnTimerMs'] = us['spawn_ms']
        if e.get('has_attack') and character:
            hit_speed = int(character.get('hitSpeedMs') or 0)
            timeline = int(e.get('atk_timeline') or 0)
            row['attack'] = {'stage': int(e.get('atk_stage') or 0), 'timelineMs': timeline, 'loadMs': int(e.get('atk_load') or 0),
                             'hitCount': us['hit_count'], 'hitSpeedMs': hit_speed, 'loadTimeMs': character.get('loadTimeMs'),
                             'cycleMs': timeline % hit_speed if state == 2 and hit_speed > 0 else None,
                             'dashCooldownMs': us['dash_ms'], 'dashHop': us.get('hop', 0),
                             'specialCooldownMs': us.get('special_ms', 0),
                             'returnShot': {'elapsedMs': us['shot'][1], 'totalMs': us['shot'][2]} if us.get('shot', [0])[0] else None}
        controller = controllers.get(oid)
        if controller is not None:
            most = character.get('abilityMaxCharges')
            charges = int(controller.get('charges', -1))
            uses = most - charges if most and most > 0 and charges >= 0 else None
            ability = {'buttonState': controller.get('button'), 'remainingCooldownMs': controller.get('cooldown_ms'),
                       'configuredCooldownMs': controller.get('configured_ms'), 'remainingChargesRaw': charges,
                       'maxCharges': most, 'uses': uses,
                       'used': uses > 0 if uses is not None else bool(controller.get('used')),
                       'effectDelayMs': max(0, int(controller.get('effect_delay_ms') or 0))}
            own = [b for b in buffs if b['name'] in self.ability_buffs and b['sourceObjectId'] == oid]
            active = bool(own) or state in (7, 16) or (ability['used'] and state == 4) \
                or (ability['used'] and state == 3 and row['name'] == 'GoldenKnight')
            ability['phase'] = 'casting' if state == 10 else 'active' if active \
                else 'casting' if ability['effectDelayMs'] > 0 else None
            ability['phaseElapsedMs'] = own[0]['elapsedMs'] if own and ability['phase'] == 'active' else None
            C.ability_clock(self.last, tick, [{'nativeObjectId': e['address'], 'ability': ability}])
            row['ability'] = ability
        return row


def _object(e, ids):
    """One objects() row: the attack's target, load and the deploy timer only with an attack component."""
    attack = bool(e.get('has_attack'))
    target = ids.get(e.get('target'))
    return {'nativeObjectId': e['category'], 'owner': e['side'], 'cardId': e['card_id'],
            'dataGlobalId': (e.get('data_id') or 0) & 0xFFFFFFFF, 'x': e['x'], 'y': e['y'],
            'hp': e['hp'] if e['hp'] >= 0 else None, 'maxHp': e['max_hp'] if e['max_hp'] >= 0 else None,
            'targetEntityKey': [target[0], -2, target[1]] if attack and target else None,
            'phaseRuntime': {'attackValidated': attack, 'deployRemainingMs': e['deploy_remaining'] if attack else None,
                             'loadRemainingMs': e['atk_load'] if attack else None}}


def _player(p):
    """players_state()'s row: hand and cycle by deck slot, elixir, deck."""
    deck = p['deck_card_ids']
    return {'owner': p['side'], 'elixirRaw': p['elixir_raw'], 'elixir': p['elixir_raw'] / 10000.0,
            'hand': [{'handIndex': i, 'deckSlot': s, 'cardId': deck[s]} for i, s in enumerate(p['hand_deck_indices']) if s >= 0],
            'cycle': [{'cycleIndex': i, 'deckSlot': s, 'cardId': deck[s]} for i, s in enumerate(p['cycle_deck_indices']) if s >= 0],
            'deck': [{'deckSlot': i, 'cardId': c} for i, c in enumerate(deck)]}


def _lean(p):
    """players_lean()'s row: the hero / champion controllers and each deck slot's evolution progress."""
    abilities = [{'controllerSlot': a['controller_slot'], 'available': a['character_id'] != 0, 'buttonState': a['button'],
                  'remainingChargesRaw': a['charges'], 'remainingCooldownMs': a['cooldown_ms'],
                  'configuredCooldownMs': a['configured_ms'], 'championObjectIds': a['units'], 'actionDataName': ''}
                 for a in p['abilities']]
    evolutions = []
    for slot, (card, form, progress) in enumerate(zip(p['deck_card_ids'], p['deck_form_flags'], p['evo_progress'])):
        need = CYCLE_REQUIRED.get(card) if form == EVOLUTION else None
        evolutions.append({'deckSlot': slot, 'cardId': card, 'evolvable': need is not None, 'progress': progress,
                           'cycleRequired': need, 'cycleRemaining': max(0, need - progress) if need else None,
                           'ready': progress >= need if need else None})
    return {'owner': p['side'], 'abilityRuntime': abilities, 'evolutionRuntime': evolutions}


class LivePlays:
    """Confirmed hand changes and currently visible queued cards, in PlayerView's command format.

    A queue entry is intent, not proof of execution. Only a hand change confirms a play; matching it to a previously
    observed command recovers its coordinates. Create a new instance per battle: sequence numbers and native addresses
    may be reused."""

    def __init__(self):
        self.hands, self.landed, self.seen, self.flying = {}, [], {}, []

    def update(self, frame):
        if not frame.get('coherent', True) or frame['failure'] != 'none':
            return list(self.landed)
        tick = frame['game_tick']
        self.flying = []
        for c in frame.get('commands', ()) if frame.get('queue_ok') else ():
            key = (c['owner'], c['sequence'], c['issue_tick'])
            command = self.seen.setdefault(key, dict(c, first_seen_tick=tick, confirmed=False))
            if c['execute_tick'] > tick and not command['confirmed']:
                self.flying.append((c['execute_tick'], command))
        for p in frame['players']:
            deck = p['deck_card_ids']
            hand = {i: deck[s] for i, s in enumerate(p['hand_deck_indices']) if s >= 0}
            before = self.hands.get(p['side'])
            if before is not None:
                for i, card in before.items():
                    if hand.get(i) == card:
                        continue
                    candidates = [c for c in self.seen.values() if c['owner'] == p['side'] and c['card_id'] == card
                                  and not c['confirmed'] and 0 <= tick - c['execute_tick'] <= 10]
                    command = min(candidates, key=lambda c: c['execute_tick']) if candidates else \
                        {'owner': p['side'], 'card_id': card}
                    command['confirmed'] = True
                    self.landed.append((tick, dict(command)))
            self.hands[p['side']] = hand
        self.flying = [(due, c) for due, c in self.flying if not c['confirmed']]
        return list(self.landed)

    def commands(self, pending=None, owner=None):
        """Known history plus visible commands. A just-sent own tap fills the queue's acknowledgement delay only."""
        flying = list(self.flying)
        if pending and not any(c['owner'] == owner and c['card_id'] == pending['card'] for _, c in flying):
            flying.append((pending['due'], {'owner': owner, 'card_id': pending['card'],
                                            'x': pending['x'], 'y': pending['y']}))
        return self.landed + flying


# ---- a battle -----------------------------------------------------------------------------------------------------------
def board(record):
    """Live log frame rows: owner, card, x, y, hp, max hp, type (0 troop, 1 tower, 2 projectile, 3 area)."""
    def kind(o):
        if o['hp'] is None:
            return 2 if 4000000 <= o['nativeObjectId'] < 5000000 else 3
        return 1 if o['cardId'] == -1 else 0
    return [[o['owner'], o['cardId'], o['x'], o['y'], -1 if o['hp'] is None else o['hp'],
             -1 if o['maxHp'] is None else o['maxHp'], kind(o)] for o in record['objects']]


def play_battle(stream, first, model, vocab, records, owner, taps, log, levels=None):
    """One battle from its first frame to its end -> the last record."""
    other = 1 - owner
    records.reset()
    record = records(first)
    forms = next(p['deck_form_flags'] for p in first['players'] if p['side'] == other)
    view = C.PlayerView(owner, vocab, C.opponent_heroes(record, forms, owner), C.cached_legal(), levels=levels)
    agent, plays = train.Agent(model, owner), LivePlays()
    log({'event': 'start', 'owner': owner, 'tick': first['game_tick']})
    decided, pending, reported = None, None, 0  # pending: {'card', 'tick', 'due', 'x', 'y'} of a card tap not landed
    battle = first['chain']['battle']
    frame, last_tick, moved = first, first['game_tick'], time.monotonic()
    while True:
        if frame['failure'] == 'none' and frame.get('coherent', True):
            if frame['game_tick'] >= FIRST_DECISION and not frame.get('queue_ok'):
                raise RuntimeError('command queue unavailable: install the queue-enabled Nulls reader')
            landed = plays.update(frame)
            for tick, c in landed[reported:]:
                log({'event': 'landed' if c['owner'] == owner else 'opponent', 'tick': tick, 'card': c['card_id'],
                     'world': [c['x'], c['y']] if 'x' in c else None,
                     **{k: c[k] for k in ('issue_tick', 'execute_tick', 'first_seen_tick', 'sequence') if k in c}})
                if c['owner'] == owner and pending and pending['card'] == c['card_id']:
                    pending = None
            reported = len(landed)
            tick = frame['game_tick']
            if pending and tick > pending['due'] + C.HOLD_TICKS + 10:
                log({'event': 'unconfirmed', 'tick': tick, 'card': pending['card']})
                pending = None
            if tick >= FIRST_DECISION and (decided is None or tick >= decided + 5):
                decided = tick
                record = records(frame)
                choice = agent.act(view.step(record, plays.commands(pending, owner)))
                me = next(p for p in record['lean']['state']['players'] if p['owner'] == owner)
                log({'event': 'decision', 'tick': tick, 'elixir': me['elixirRaw'], 'choice': choice[0], 'board': board(record),
                     'opponent_commands': [c for _, c in plays.flying if c['owner'] == other]})
                if choice[0] == 'ability':  # pressed at once: nothing in flight to wait for
                    taps.tap(*ABILITY_BUTTON)
                    log({'event': 'tap', 'tick': tick, 'card': None, 'ability': True})
                elif choice[0] == 'card' and not pending:
                    _, slot, x, y = choice
                    card = next((h['cardId'] for h in me['hand'] if h['handIndex'] == slot), None)
                    short = C.COSTS.get(card, 3) * 10000 - me['elixirRaw']
                    if card is not None and short <= C.hold_slack(tick):  # the client holds a short tap ~HOLD_TICKS
                        started = time.perf_counter()
                        pixels = taps.play(slot, x, y, owner)
                        wait = next(k for k in range(C.HOLD_TICKS + 1) if C.regen(tick, tick + k) >= short) if short > 0 else 0
                        pending = {'card': card, 'tick': tick, 'due': tick + wait + C.DEPLOY_DELAY, 'x': x, 'y': y}
                        log({'event': 'tap', 'tick': tick, 'card': card, 'slot': slot, 'world': [x, y], 'screen': pixels,
                             'input_ms': round((time.perf_counter() - started) * 1000)})
        frame = next(stream)
        if frame['game_tick'] != last_tick:
            last_tick, moved = frame['game_tick'], time.monotonic()
        replaced = frame['failure'] == 'none' and frame['chain']['battle'] != battle
        stalled = last_tick > first['game_tick'] and time.monotonic() - moved > STALL_S
        if not frame['battle_active'] or replaced or stalled:
            return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help="a train.py checkpoint (latest.pt)")
    parser.add_argument('--tables', required=True, type=Path, help="clapha-engine export_tables.py's JSON, August build")
    parser.add_argument('--account', type=lambda s: int(s, 0), default=ACCOUNT, help='our account id (finds our seat)')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--serial', default='emulator-5554')
    parser.add_argument('--out', type=Path, default=Path.home() / 'cr-live/runs/live')
    parser.add_argument('--levels', type=Path, help='JSON {card id: level} of our deck (models with Config.levels)')
    parser.add_argument('--queue', action='store_true', help='press OK and Battle to start each ladder game')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, vocab, _ = train.load(args.checkpoint)
    levels = None
    if model.config.levels:  # the reader does not read card levels yet: our deck's, as the collection shows them
        if not args.levels:
            parser.error('this model reads card levels: pass --levels (JSON {card id: level} of our deck)')
        levels = {int(k): int(v) for k, v in json.loads(args.levels.read_text()).items()}
    records = LiveRecords(args.tables)
    prepare(args.serial)
    taps, stream = Taps(args.serial), frames(args.serial)
    args.out.mkdir(parents=True, exist_ok=True)
    finished = None  # the last battle played: the results screen keeps it readable
    for game in range(args.games):
        print('waiting for a battle...', flush=True)
        checked = time.monotonic()
        frame = next(stream)
        # a battle from its start: the results screen keeps a finished one active with its tick standing still
        while not (frame['battle_active'] and frame['failure'] == 'none' and frame['chain']['battle'] != finished
                   and frame['game_tick'] < FIRST_DECISION):
            if args.queue and time.monotonic() - checked > 5:
                checked = time.monotonic()
                now = screen(args.serial)  # OK on results / trophy road, then Battle
                if now:
                    taps.tap(*(ROAD_OK if now == 'road' else BUTTON))
            frame = next(stream)
        path = args.out / time.strftime('%Y%m%d_%H%M%S.jsonl')
        with path.open('w') as f:
            def log(row):
                f.write(json.dumps(row) + '\n')
                f.flush()
            log({'event': 'checkpoint', 'path': str(args.checkpoint)})
            finished = frame['chain']['battle']
            for _ in range(20):  # the roster is not linked yet at the battle's first frames
                try:
                    ids = roster(args.serial)
                    if ids.count(args.account) == 1:
                        break
                except (struct.error, ValueError, StopIteration, subprocess.CalledProcessError):
                    ids = []
                time.sleep(0.2)
            if ids.count(args.account) != 1:
                raise SystemExit(f'account {args.account:#x} is not exactly one seat: {[hex(i) for i in ids]}')
            owner = ids.index(args.account)
            log({'event': 'roster', 'accounts': ids, 'owner': owner})
            last = play_battle(stream, frame, model, vocab, records, owner, taps, log, levels)
            alive = {'objects': [o for o in last['objects'] if o['hp'] is None or o['hp'] > 0]}  # the last frame still
            score = (C.crowns(alive, owner), C.crowns(alive, 1 - owner))                     # lists a tower at 0 hp
            log({'event': 'end', 'tick': last['tick'], 'crowns': score})
            print(f'game {game + 1}: crowns {score[0]}-{score[1]} -> {path}', flush=True)


if __name__ == '__main__':
    main()
