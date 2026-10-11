"""Measure the deploy delay live: tap a card in a Training Camp battle and read the exact tick the probe logs for
that play. Mode a taps card then tile (as il.live); mode b selects the card first and times only the tile tap.
Elixir is kept above the card's cost so no play waits for elixir.

    python bootstrap_emulator.py --online
    PYTHONPATH=. python tools/measure_delay.py --deck 26000043,26000020 --trials 30
"""
import argparse
import statistics
import time

from native_engine import Engine
from il.live import Taps, capture, find_owner, slot_screen, world_screen

MENU, TRAINING_CAMP, CONFIRM = (981, 188), (645, 623), (735, 1360)

parser = argparse.ArgumentParser()
parser.add_argument('--deck', required=True, help='card ids only our deck has')
parser.add_argument('--trials', type=int, default=30)
parser.add_argument('--serial', default='emulator-5554')
args = parser.parse_args()
engine = Engine()
print('probe:', engine.request('live on'), flush=True)
taps = Taps(args.serial)
for xy in (MENU, TRAINING_CAMP, CONFIRM):
    taps.tap(*xy)
    time.sleep(2.5)
state = None
while state is None or state.tick < 150:  # the opening seconds: let the battle settle
    time.sleep(0.2)
    state = capture(engine)
owner = find_owner(state, {int(c) for c in args.deck.split(',')}, None)
x, y = (9000, 8000) if owner == 0 else (9000, 24000)  # own half, centre
results = {'a': [], 'b': []}
for trial in range(args.trials):
    mode = 'ab'[trial % 2]
    while True:  # a card we can afford with a margin, so it never waits for elixir
        s = capture(engine)
        if s is None or s.finalized:
            break
        me = s.players[owner]
        card = min(me.hand, key=lambda c: c.cost)
        if me.elixir >= (card.cost + 1) * 10000:
            break
        time.sleep(0.05)
    if s is None or s.finalized:
        break
    cursor = s.next_sequence
    if mode == 'b':
        taps.tap(*slot_screen(card.slot))
        time.sleep(0.4)
    before = capture(engine).tick
    if mode == 'a':
        taps.play(card.slot, x, y, owner)
    else:
        taps.tap(*world_screen(x, y, owner))
    after = capture(engine).tick
    landed = None
    deadline = time.time() + 4
    while landed is None and time.time() < deadline:
        n = capture(engine, cursor)
        mine = [p for p in (n.plays if n else ()) if p.owner == owner]
        landed = mine[0].tick if mine else None
        time.sleep(0.02)
    if landed is None:
        print(f'trial {trial} ({mode}): no play logged within 4 s', flush=True)
        continue
    results[mode].append(landed - before)
    print(f'trial {trial} ({mode}): tick before tap {before}, after tap {after}, card played at {landed}: '
          f'{landed - before} ticks', flush=True)
    time.sleep(1.5)
engine.request('live off')
for mode, label in (('a', 'card + tile tap'), ('b', 'tile tap only (card preselected)')):
    d = results[mode]
    if d:
        print(f'{label}: n={len(d)} min {min(d)} median {statistics.median(d)} max {max(d)} ticks '
              f'= {statistics.median(d) / 20:.2f} s median')
