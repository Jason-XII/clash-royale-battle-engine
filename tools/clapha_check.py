"""Check ClaphaEngine on the x86_64 node: threads per engine, the lean player read against clapha's full reader at
every decision of a few random games (22-tick delay), and the time of a decision step.

    cd ~/cr-engine && PYTHONPATH=. python tools/clapha_check.py ~/clapha-engine --games 5
"""
import argparse
import os
import random
import time

from native_engine import Play
from native_engine.clapha import ClaphaEngine
from il.arena import deck_pairs, make_match

parser = argparse.ArgumentParser()
parser.add_argument('clapha')
parser.add_argument('--games', type=int, default=5)
parser.add_argument('--pool', default=os.path.expanduser('~/cr-engine/data/deck_pool.json'))
args = parser.parse_args()
engine = ClaphaEngine(args.clapha)
print('threads in this engine process:', len(os.listdir('/proc/self/task')), flush=True)
rng = random.Random(0)
mismatches = decisions = 0
spent = observe_spent = 0.0
for game, pair in enumerate(deck_pairs(args.pool, None, args.games, 0)):
    state = engine.reset(match=make_match(pair, game + 1))
    while not state.finalized and state.tick < 7200:
        plays = []
        for p in state.players:
            ok = [h for h in p.hand if h.cost * 10000 <= p.elixir]
            if ok and rng.random() < 0.15:
                y = rng.randrange(1000, 14000) if p.owner == 0 else rng.randrange(18000, 31000)
                plays.append(Play(p.owner, rng.choice(ok).slot, rng.randrange(500, 17500), y))
        t = time.perf_counter()
        state = engine.step(plays, 5, delay=22).state
        spent += time.perf_counter() - t
        t = time.perf_counter()
        full = {p['owner']: p for p in engine.env.observe()['players']}  # clapha's own full read, as before
        observe_spent += time.perf_counter() - t
        for owner, (hand, cycle, elixir) in enumerate(engine._players()):
            ref = full[owner]
            same = (hand == {h['handIndex']: (h['deckSlot'], h['cardId']) for h in ref['hand']}
                    and list(cycle) == [c['cardId'] for c in ref['cycle']] and elixir == ref['elixirRaw'])
            if not same:
                mismatches += 1
                if mismatches <= 3:
                    print('MISMATCH tick', state.tick, 'owner', owner, hand, cycle, elixir, '| clapha:', ref['hand'], ref['cycle'], ref['elixirRaw'])
        decisions += 1
    print(f'game {game}: tick {state.tick}, result {state.result}, crowns {state.crowns}', flush=True)
print(f'{decisions} decisions, {mismatches} player-read mismatches')
print(f'engine.step (5 ticks + State): {1000 * spent / decisions:.3f} ms per decision; '
      f"clapha's full observe() alone: {1000 * observe_spent / decisions:.3f} ms")
os._exit(0)  # the engine segfaults in its exit-time destructors
