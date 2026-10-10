"""Play live Nulls Royale battles with a checkpoint (Mac side).

    python bootstrap_emulator.py --online      # game with network access, probe on port 26789
    python -m il.live runs/il-002-best.pt --deck 26000000,26000001,...   # then start a battle by hand

The probe's live mode captures the battle the game itself runs, so the model sees exactly the
training features. Cards are played by tapping the screen (card, then tile) through one adb
shell. Our seat is the player whose 8 cards match --deck (or --owner). Each battle is logged to
runs/live/<time>.jsonl.
"""
import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import torch

from native_engine import Engine
from native_engine.protocol import State

from .agent import Agent, load

# Pixel_9_2 emulator screen (1080x2424), measured in clash-simulator-refactored/minimal_visualizer.py:
# the 30 inner arena rows span y 295..1714, so 32 rows span one row more at each end.
X0, X1, Y0, Y1 = 16, 1053, 295, 1714
TILE_W, TILE_H = (X1 - X0) / 18, (Y1 - Y0) / 30
BOTTOM = Y1 + TILE_H


def slot_screen(slot):
    return 335 + 200 * slot, 2220


def world_screen(x, y, owner):
    """Native world units -> screen pixels. Our side is at the bottom; owner 0 sees x mirrored."""
    vx, vy = (18 - x / 1000, y / 1000) if owner == 0 else (x / 1000, 32 - y / 1000)
    return round(X0 + TILE_W * vx), round(BOTTOM - TILE_H * vy)


class Taps:
    """One persistent adb shell; each play is `tap card && tap tile`, acknowledged."""

    def __init__(self, serial):
        self.shell = subprocess.Popen(['adb', '-s', serial, 'shell'], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1)

    def play(self, slot, x, y, owner):
        (sx, sy), (tx, ty) = slot_screen(slot), world_screen(x, y, owner)
        self.shell.stdin.write(f'input tap {sx} {sy} && input tap {tx} {ty}; echo done:$?\n')
        self.shell.stdin.flush()
        line = self.shell.stdout.readline().strip()
        if line != 'done:0':
            raise RuntimeError(f'adb tap failed: {line!r}')
        return tx, ty


def capture(engine, cursor=0):
    """Latest live state, keeping only plays from sequence `cursor` on (live captures carry
    every play of the battle)."""
    data = engine.request('minimal 0 0')
    if not data.get('ok'):
        return None
    state = State.decode(data)
    return dataclasses.replace(state, plays=tuple(p for p in state.plays if p.sequence >= cursor))


def find_owner(state, deck, owner):
    if owner is not None:
        return owner
    seats = [p.owner for p in state.players if {c.card for c in p.hand} | set(p.cycle) == deck]
    if len(seats) == 1:
        return seats[0]
    decks = {p.owner: sorted({c.card for c in p.hand} | set(p.cycle)) for p in state.players}
    raise SystemExit(f'cannot tell our seat from --deck; decks by owner: {decks} (use --owner for mirrors)')


def play_battle(engine, agent, taps, state, args, log):
    """From the first capture of a battle to its end. Returns the final state."""
    generation, cursor = state.generation, state.next_sequence
    plays = list(state.plays)
    decided = None  # tick of the last decision
    pending = None  # (card, tick) of a tap whose play has not shown up yet
    seen = time.monotonic()
    while not state.finalized:
        time.sleep(0.05)
        new = capture(engine, cursor)
        if new is None and time.monotonic() - seen < 5:
            continue
        if new is None or new.generation != generation:
            log({'event': 'lost', 'tick': state.tick})  # the battle vanished or another one began
            return state
        seen = time.monotonic()
        state, cursor = new, new.next_sequence
        plays += new.plays
        if pending and any(p.owner == agent.owner for p in new.plays):
            log({'event': 'landed', 'tick': state.tick, 'delay_ticks': state.tick - pending[1]})
            pending = None
        if decided is not None and state.tick < decided + 5:
            continue
        decided = state.tick
        # The agent sees every play since its last decision, as in training.
        choice = agent.act(dataclasses.replace(state, plays=tuple(plays)))
        plays = []
        if pending and state.tick - pending[1] > 30:
            log({'event': 'unconfirmed', 'tick': state.tick, 'card': pending[0]})
            pending = None
        if choice[0] != 'card' or pending:
            continue  # ponytail: ability presses are ignored (no button calibration yet)
        _, slot, x, y = choice
        me = state.players[agent.owner]
        card = next((c for c in me.hand if c.slot == slot), None)
        if card is None or card.cost * 10000 > me.elixir:
            continue  # a failed placement would leave the card selected on screen
        started = time.perf_counter()
        screen = taps.play(slot, x, y, agent.owner)
        pending = (card.card, state.tick)
        log({'event': 'tap', 'tick': state.tick, 'card': card.card, 'slot': slot, 'world': [x, y],
             'screen': screen, 'input_ms': round((time.perf_counter() - started) * 1000)})
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--deck', help='our 8 card ids, comma separated')
    parser.add_argument('--owner', type=int, choices=(0, 1), help='our native seat (mirror matches)')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--decode', choices=('gate', 'sample', 'argmax'), default='gate')
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--serial', default='emulator-5554')
    args = parser.parse_args()
    if args.deck is None and args.owner is None:
        parser.error('give --deck (or --owner)')
    deck = {int(c) for c in args.deck.split(',')} if args.deck else None
    torch.set_num_threads(2)
    model, vocab = load(args.checkpoint)
    engine = Engine(port=args.port)
    print('probe:', engine.request('live on'), flush=True)
    taps = Taps(args.serial)
    Path('runs/live').mkdir(parents=True, exist_ok=True)
    try:
        played, done = 0, set()
        while played < args.games:
            print('waiting for a battle...', flush=True)
            state = None
            while state is None or state.finalized or state.generation in done:
                time.sleep(0.5)
                state = capture(engine)
            done.add(state.generation)
            owner = find_owner(state, deck, args.owner)
            agent = Agent(model, vocab, owner, decode=args.decode)
            path = Path('runs/live') / time.strftime('%Y%m%d_%H%M%S.jsonl')
            with path.open('w') as f:
                def log(record):
                    f.write(json.dumps(record) + '\n')
                    f.flush()
                log({'event': 'start', 'checkpoint': str(args.checkpoint), 'owner': owner,
                     'generation': state.generation, 'tick': state.tick})
                state = play_battle(engine, agent, taps, state, args, log)
                if state.finalized:
                    won = state.result == owner
                    log({'event': 'end', 'result': state.result, 'won': won, 'crowns': state.crowns,
                         'tick': state.tick})
                    print(f'{"WIN" if won else "LOSS" if state.result in (0, 1) else "DRAW"} '
                          f'crowns {state.crowns[owner]}-{state.crowns[1 - owner]} -> {path}', flush=True)
                    played += 1
    finally:
        engine.request('live off')


if __name__ == '__main__':
    main()
