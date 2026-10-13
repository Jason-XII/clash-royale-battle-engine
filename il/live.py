"""Play Null's Royale battles with a clapha-il checkpoint (Mac side). Arron's live reader, ported to the Nulls build
(~/clapha-console/nulls/README.md: build it, push it to READER), streams the battle from the game's memory; il.live_record
turns each frame into the record a clapha-il model reads in self-play; il.policy decides every 5 ticks; adb taps play
(card, then tile).

    python3 -m native_engine.emulator start && python3 -m native_engine.emulator bootstrap --online
    CLAPHA_IL=~/clapha-il-dev python -m il.live CHECKPOINT --tables TABLES.json --training      # Training Camp
    CLAPHA_IL=~/clapha-il-dev python -m il.live CHECKPOINT --tables TABLES.json --games 10 --queue   # ladder

TABLES: clapha-engine crx/export_tables.py of the August build. Our seat is the player whose deck is --deck. Each battle
is logged to runs/live/<time>.jsonl (decisions with the board, taps, plays, the end): il.defense --live reads them.
"""
import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

import torch

from native_engine.engine import hold_slack, regen

from . import policy
from .live_record import LivePlays, LiveRecords

FE = policy.FE
PACKAGE = 'nullsroyale.rel.free'
READER = '/data/local/tmp/clapha-reader-nulls'
MANAGER_RVA, ROOT_CONTEXT = 0x19BD358, 0x20  # the Nulls build's (clapha-console nulls/README.md)
FRAME_MS = 50
FIRST_DECISION = 90  # as in imitation learning and self-play (ClaphaEngine.reset); taps earlier are not taken anyway
STALL_S = 2.0  # the tick standing still this long once it has moved: the battle is over (the results screen keeps the
               # battle object; it also stands still for a few seconds before the start)
DEPLOY_DELAY = 22  # tap -> landing, measured live 21-25 ticks (docs/GAME_INTEGRATION.md)
HOLD_TICKS = 23  # a tap the elixir cannot pay for yet is held this long (engine.HOLD_TICKS)
EBARBS = '26000043,26000020,26000036,26000052,26000017,26000083,28000015,28000026'

# Pixel_9_2 emulator screen (1080x2424), measured in clash-simulator-refactored/minimal_visualizer.py:
# the 30 inner arena rows span y 295..1714, so 32 rows span one row more at each end.
X0, X1, Y0, Y1 = 16, 1053, 295, 1714
TILE_W, TILE_H = (X1 - X0) / 18, (Y1 - Y0) / 30
BOTTOM = Y1 + TILE_H
# Home screen Battle (yellow) and results screen OK (blue) overlap here, below their labels.
BUTTON = (440, 2035)
ROAD_OK = (539, 2345)  # OK at the bottom of the trophy road page after a new arena
ABILITY_BUTTON = (973, 1954)  # hero / champion ability, bottom right above the hand
MENU, TRAINING_CAMP, TRAINING_OK = (983, 202), (646, 622), (732, 1360)


def screen(serial):
    """Which screen is up, by a few pixels: 'home' (Battle, yellow at BUTTON), 'results' (OK, blue at BUTTON), 'road' (the
    trophy road page after a battle: light blue at both ends of ROAD_OK), or None (matchmaking, loading, battle)."""
    raw = subprocess.run(['adb', '-s', serial, 'exec-out', 'screencap'], capture_output=True).stdout
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


def to_home(serial, taps, timeout=60):
    """Press the results' and trophy road's OK until the home screen is up."""
    started = time.monotonic()
    while (now := screen(serial)) != 'home' and time.monotonic() - started < timeout:
        if now:
            taps.tap(*(BUTTON if now == 'results' else ROAD_OK))
        time.sleep(2)


def slot_screen(slot):
    return 335 + 200 * slot, 2220


def world_screen(x, y, owner):
    """Native world units -> screen pixels. Our side is at the bottom; owner 0 sees x mirrored."""
    vx, vy = (18 - x / 1000, y / 1000) if owner == 0 else (x / 1000, 32 - y / 1000)
    return round(X0 + TILE_W * vx), round(BOTTOM - TILE_H * vy)


class Taps:
    """One persistent adb shell; each play is `tap card && tap tile`, acknowledged."""

    def __init__(self, serial):
        self.shell = subprocess.Popen(['adb', '-s', serial, 'shell'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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
        (sx, sy), (tx, ty) = slot_screen(slot), world_screen(x, y, owner)
        self.run(f'input tap {sx} {sy} && input tap {tx} {ty}')
        return tx, ty


def frames(serial):
    """The reader's frames, for good: it keeps reading across battles and menus, and starts again when the game restarts
    (a new process; a battle in progress then is lost)."""
    command = f'{READER} $(pidof {PACKAGE}) {FRAME_MS} {MANAGER_RVA:#x} {ROOT_CONTEXT:#x} --unified 0'
    while True:
        reader = subprocess.Popen(['adb', '-s', serial, 'shell', command], stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in reader.stdout:
            if line.startswith('{'):
                yield json.loads(line)
        print(f'the reader stopped (exit {reader.wait()}; is it on the device at {READER}, the game running?); '
              'starting it again', flush=True)
        time.sleep(5)


def board(record):
    """il.defense's frame rows: owner, card, x, y, hp, max hp, type (0 troop, 1 tower, 2 projectile, 3 area)."""
    def kind(o):
        if o['hp'] is None:
            return 2 if 4000000 <= o['nativeObjectId'] < 5000000 else 3
        return 1 if o['cardId'] == -1 else 0
    return [[o['owner'], o['cardId'], o['x'], o['y'], -1 if o['hp'] is None else o['hp'],
             -1 if o['maxHp'] is None else o['maxHp'], kind(o)] for o in record['objects']]


def play_battle(stream, first, model, vocab, records, deck, taps, log):
    """One battle from its first frame to its end -> the last record."""
    seats = [p['side'] for p in first['players'] if set(p['deck_card_ids']) == deck]
    if len(seats) != 1:
        raise SystemExit(f"--deck is not exactly one seat's deck: {[p['deck_card_ids'] for p in first['players']]}")
    owner, other = seats[0], 1 - seats[0]
    record = records(first)
    forms = next(p['deck_form_flags'] for p in first['players'] if p['side'] == other)
    view = FE.PlayerView(owner, vocab, FE.opponent_heroes([record], {'episode': {f'deck{other}_form_availability': forms}},
                                                          owner), policy.legal_fn())
    agent, plays = policy.Agent(model, owner), LivePlays()
    log({'event': 'start', 'owner': owner, 'tick': first['game_tick']})
    decided, pending, reported = None, None, 0  # pending: {'card', 'tick', 'due', 'x', 'y'} of a card tap not landed
    battle = first['chain']['battle']
    frame, last_tick, moved = first, first['game_tick'], time.monotonic()
    while True:
        if frame['failure'] == 'none':
            landed = plays.update(frame)
            for tick, c in landed[reported:]:
                log({'event': 'landed' if c['owner'] == owner else 'opponent', 'tick': tick, 'card': c['card_id']})
                if c['owner'] == owner and pending and pending['card'] == c['card_id']:
                    pending = None
            reported = len(landed)
            tick = frame['game_tick']
            if pending and tick > pending['due'] + HOLD_TICKS + 10:
                log({'event': 'unconfirmed', 'tick': tick, 'card': pending['card']})
                pending = None
            if tick >= FIRST_DECISION and (decided is None or tick >= decided + 5):
                decided = tick
                record = records(frame)
                flying = [(pending['due'], {'owner': owner, 'card_id': pending['card'], 'x': pending['x'], 'y': pending['y']})] \
                    if pending else []
                choice = agent.act(view.step(record, landed + flying))
                me = next(p for p in record['lean']['state']['players'] if p['owner'] == owner)
                log({'event': 'decision', 'tick': tick, 'elixir': me['elixirRaw'], 'choice': choice[0], 'board': board(record)})
                if choice[0] == 'ability':  # pressed at once: nothing in flight to wait for
                    taps.tap(*ABILITY_BUTTON)
                    log({'event': 'tap', 'tick': tick, 'card': None, 'ability': True})
                elif choice[0] == 'card' and not pending:
                    _, slot, x, y = choice
                    card = next((h['cardId'] for h in me['hand'] if h['handIndex'] == slot), None)
                    short = FE.COSTS.get(card, 3) * 10000 - me['elixirRaw']
                    if card is not None and short <= hold_slack(tick):  # the client holds a short tap ~HOLD_TICKS
                        started = time.perf_counter()
                        screen = taps.play(slot, x, y, owner)
                        wait = next(k for k in range(HOLD_TICKS + 1) if regen(tick, tick + k) >= short) if short > 0 else 0
                        pending = {'card': card, 'tick': tick, 'due': tick + wait + DEPLOY_DELAY, 'x': x, 'y': y}
                        log({'event': 'tap', 'tick': tick, 'card': card, 'slot': slot, 'world': [x, y], 'screen': screen,
                             'input_ms': round((time.perf_counter() - started) * 1000)})
        frame = next(stream)
        if frame['game_tick'] != last_tick:
            last_tick, moved = frame['game_tick'], time.monotonic()
        replaced = frame['failure'] == 'none' and frame['chain']['battle'] != battle
        stalled = last_tick > first['game_tick'] and time.monotonic() - moved > STALL_S
        if not frame['battle_active'] or replaced or stalled:
            return record


def crowns(record, owner):
    """Crowns `owner` has taken, from the towers left standing (clapha-il features._crowns; the last frame still lists a
    tower destroyed in it, at 0 hp)."""
    return FE._crowns({'objects': [o for o in record['objects'] if o['hp'] is None or o['hp'] > 0]}, owner)


def start_training(serial, taps):
    """Menu -> Training Camp -> OK: a battle against the trainer, no trophies at stake."""
    to_home(serial, taps)
    for button in (MENU, TRAINING_CAMP, TRAINING_OK):
        taps.tap(*button)
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path, help='a clapha-il checkpoint')
    parser.add_argument('--tables', required=True, type=Path, help="clapha-engine export_tables.py's JSON, August build")
    parser.add_argument('--deck', default=EBARBS, help='our deck: 8 card ids, comma separated (finds our seat)')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--serial', default='emulator-5554')
    parser.add_argument('--queue', action='store_true', help='press OK and Battle to start each ladder game')
    parser.add_argument('--training', action='store_true', help='start each game in Training Camp (no trophies)')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, vocab, _ = policy.load(args.checkpoint)
    records, taps, deck = LiveRecords(args.tables), Taps(args.serial), {int(c) for c in args.deck.split(',')}
    stream = frames(args.serial)
    Path('runs/live').mkdir(parents=True, exist_ok=True)
    finished = None  # the last battle played: the results screen keeps it readable
    for game in range(args.games):
        print('waiting for a battle...', flush=True)
        if args.training:
            start_training(args.serial, taps)
        checked = time.monotonic()
        frame = next(stream)
        while not (frame['battle_active'] and frame['failure'] == 'none' and frame['chain']['battle'] != finished):
            if args.queue and time.monotonic() - checked > 5:
                checked = time.monotonic()
                now = screen(args.serial)  # OK on results / trophy road, then Battle
                if now:
                    taps.tap(*(ROAD_OK if now == 'road' else BUTTON))
            frame = next(stream)
        path = Path('runs/live') / time.strftime('%Y%m%d_%H%M%S.jsonl')
        with path.open('w') as f:
            def log(row):
                f.write(json.dumps(row) + '\n')
                f.flush()
            log({'event': 'checkpoint', 'path': str(args.checkpoint)})
            finished = frame['chain']['battle']
            last = play_battle(stream, frame, model, vocab, records, deck, taps, log)
            owner = next(p['owner'] for p in last['lean']['state']['players'] if {c['cardId'] for c in p['deck']} == deck)
            score = (crowns(last, owner), crowns(last, 1 - owner))
            log({'event': 'end', 'tick': last['tick'], 'crowns': score})
            print(f'game {game + 1}: crowns {score[0]}-{score[1]} -> {path}', flush=True)
        if args.training:
            to_home(args.serial, taps)


if __name__ == '__main__':
    main()
