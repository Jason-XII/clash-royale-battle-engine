"""Play live Nulls Royale battles with a checkpoint (Mac side), or watch it play itself in the emulator engine.

    python -m native_engine.emulator bootstrap --online      # game with network access, probe on port 26789
    python -m il.live runs/il-002/best.pt --deck 26000043,26000020 --games 10 --queue   # --queue starts each battle
    python -m il.live runs/il-002/best.pt --watch data/IL_Replay/replays/part-000051.parquet --games 2 --render

The probe's live mode captures the battle the game itself runs, so the model sees exactly the
training features. Cards are played by tapping the screen (card, then tile) through one adb
shell. Our seat is the player whose 8 cards match --deck (or --owner). Each battle is logged to
runs/live/<time>.jsonl. --watch: self-play with the decks of human replays, drawn in the game UI with --render.
"""
import argparse
import dataclasses
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch

from native_engine import Engine, Play, RenderedEngine
from native_engine.engine import State

from . import data as D
from .model import Agent, load

# Pixel_9_2 emulator screen (1080x2424), measured in clash-simulator-refactored/minimal_visualizer.py:
# the 30 inner arena rows span y 295..1714, so 32 rows span one row more at each end.
X0, X1, Y0, Y1 = 16, 1053, 295, 1714
TILE_W, TILE_H = (X1 - X0) / 18, (Y1 - Y0) / 30
BOTTOM = Y1 + TILE_H
# Home screen Battle (yellow) and results screen OK (blue) overlap here, below their labels.
BUTTON = (440, 2035)
BATTLE_TAB = (541, 2323)  # bottom bar; a restarted game opens on the trophy road page
ROAD_OK = (539, 2345)  # OK at the bottom of the trophy road page after a new arena

WIZARD = 26000017  # its hero form's ability button is still uncalibrated: --shots captures it


def menu_button(serial):
    """Where to tap on the way to the next battle: Battle (yellow) or results OK (blue) at BUTTON, the OK of a trophy
    road page (shown on reaching a new arena; light blue at both ends of ROAD_OK), or None (matchmaking, loading,
    battle)."""
    raw = subprocess.run(['adb', '-s', serial, 'exec-out', 'screencap'], capture_output=True).stdout
    w, h = struct.unpack('<II', raw[:8])

    def pixel(x, y):
        i = len(raw) - w * h * 4 + (y * w + x) * 4
        return raw[i:i + 3]
    r, g, b = pixel(*BUTTON)
    if (r > 200 and g > 130 and b < 80) or (b > 180 and r < 120):
        return BUTTON
    if all(r < 120 and 150 <= g <= 210 and b > 230 for r, g, b in (pixel(470, ROAD_OK[1]), pixel(610, ROAD_OK[1]))):
        return ROAD_OK
    return None


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

    def tap(self, x, y):
        self.shell.stdin.write(f'input tap {x} {y}; echo done:$?\n')
        self.shell.stdin.flush()
        self.shell.stdout.readline()

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
    seats = [p.owner for p in state.players if deck <= {c.card for c in p.hand} | set(p.cycle)]
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
        for p in new.plays:
            if p.owner != agent.owner:
                log({'event': 'opponent', 'tick': p.tick, 'card': p.card})
        if pending and any(p.owner == agent.owner for p in new.plays):
            log({'event': 'landed', 'tick': state.tick, 'delay_ticks': state.tick - pending[1]})
            if pending[0] == WIZARD and args.shots:  # the hero ability button shows while our Wizard lives
                shot = Path(args.shots) / f'{time.strftime("%Y%m%d_%H%M%S")}_t{state.tick}.png'
                subprocess.Popen(['sh', '-c', f"sleep 2; adb -s {args.serial} exec-out screencap -p > '{shot}'"])
                log({'event': 'screenshot', 'tick': state.tick, 'path': str(shot)})
            pending = None
        if decided is not None and state.tick < decided + 5:
            continue
        decided = state.tick
        # The agent sees every play since its last decision, as in training.
        choice = agent.act(dataclasses.replace(state, plays=tuple(plays)))
        plays = []
        log({'event': 'decision', 'tick': state.tick, 'elixir': state.players[agent.owner].elixir,
             'crowns': state.crowns, 'choice': choice[0]})
        if pending and state.tick - pending[1] > 30:
            log({'event': 'unconfirmed', 'tick': state.tick, 'card': pending[0]})
            pending = None
        if choice[0] != 'card' or pending:
            continue  # ponytail: ability presses are ignored (no button calibration yet)
        _, slot, x, y = choice
        me = state.players[agent.owner]
        card = next((c for c in me.hand if c.slot == slot), None)
        if card is None or card.cost * 10000 > me.elixir + D.AFFORD_SLACK:
            continue  # the game waits for elixir only when the tap is almost affordable
        started = time.perf_counter()
        screen = taps.play(slot, x, y, agent.owner)
        pending = (card.card, state.tick)
        log({'event': 'tap', 'tick': state.tick, 'card': card.card, 'slot': slot, 'world': [x, y],
             'screen': screen, 'input_ms': round((time.perf_counter() - started) * 1000)})
    return state


def connect(args):
    engine = Engine(port=args.port)
    print('probe:', engine.request('live on'), flush=True)
    return engine


def restart_game(args, taps):
    """The game died. Bring it back online and on the home screen."""
    print('game lost; restarting it', flush=True)
    subprocess.run([sys.executable, '-m', 'native_engine.emulator', 'bootstrap', '--online'], check=True)
    time.sleep(10)
    taps.tap(*BATTLE_TAB)
    return connect(args)


def watch(args):
    """Self-play in the emulator engine with the decks of the first --games replays of a parquet part."""
    import pyarrow.parquet as pq
    from native_engine.build_cache import DECISION_TICKS, match_dict, prepare
    model, vocab = load(args.checkpoint)
    agents = [Agent(model, vocab, owner, decode=args.decode) for owner in (0, 1)]
    engine = (RenderedEngine if args.render else Engine)(port=args.port)
    pace = DECISION_TICKS / 20 if args.render else 0.0
    for row in pq.read_table(args.watch, columns=['payload_json']).slice(0, args.games).to_pylist():
        replay = prepare(json.loads(row['payload_json'])).replay
        c = replay.episode_config
        state = engine.reset(match=match_dict(replay, (c.deck0, c.deck1), c.seed))
        for agent in agents:
            agent.reset()
        tried, done = {0: 0, 1: 0}, {0: 0, 1: 0}
        while not state.finalized and state.tick < 7200:
            plays = []
            for agent in agents:
                choice = agent.act(state)
                if choice[0] == 'card':  # ponytail: ability choices are skipped
                    plays.append(Play(agent.owner, *choice[1:]))
                    tried[agent.owner] += 1
            started = time.monotonic()
            transition = engine.step(plays, DECISION_TICKS)
            time.sleep(max(0.0, pace - (time.monotonic() - started)))
            for play, ok in zip(plays, transition.executed):
                done[play.owner] += ok
            state = transition.state
        print(json.dumps({'result': state.result, 'crowns': state.crowns, 'tick': state.tick, 'plays': tried,
                          'executed': done}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--deck', help='card ids only our deck has (up to all 8), comma separated')
    parser.add_argument('--owner', type=int, choices=(0, 1), help='our native seat (mirror matches)')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--decode', choices=('gate', 'sample', 'argmax', 'policy'), default='gate')
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--serial', default='emulator-5554')
    parser.add_argument('--queue', action='store_true', help='press OK and Battle to start each game')
    parser.add_argument('--shots', help='a directory: screenshot 2 s after each of our Wizard plays lands')
    parser.add_argument('--watch', type=Path, help='a replay parquet part: self-play with its decks instead')
    parser.add_argument('--render', action='store_true', help='--watch: draw in the game UI at real-time speed')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.watch:
        return watch(args)
    if args.deck is None and args.owner is None:
        parser.error('give --deck (or --owner)')
    deck = {int(c) for c in args.deck.split(',')} if args.deck else None
    model, vocab = load(args.checkpoint)
    engine = connect(args)
    taps = Taps(args.serial)
    Path('runs/live').mkdir(parents=True, exist_ok=True)
    if args.shots:
        Path(args.shots).mkdir(parents=True, exist_ok=True)
    try:
        played, done = 0, set()
        while played < args.games:
            try:
                print('waiting for a battle...', flush=True)
                state, checked = None, 0.0
                while state is None or state.finalized or state.generation in done:
                    time.sleep(0.5)
                    state = capture(engine)
                    if args.queue and time.monotonic() - checked > 5:
                        checked = time.monotonic()
                        button = menu_button(args.serial)  # OK on results / trophy road, then Battle
                        if button:
                            taps.tap(*button)
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
            except (ConnectionError, OSError):
                engine, done = restart_game(args, taps), set()  # generations restart with the game
    finally:
        try:
            engine.request('live off')
        except OSError:
            pass


if __name__ == '__main__':
    main()
