"""Speed of the clapha engine on one core, and a check of its State (run on the x86_64 node).

    python tools/clapha_bench.py ~/clapha-engine runs/il-002/best.pt --games 3
"""
import argparse
from collections import Counter
import time

import torch

from native_engine import Play
from native_engine.clapha import ClaphaEngine

parser = argparse.ArgumentParser()
parser.add_argument('clapha')
parser.add_argument('checkpoint')
parser.add_argument('--games', type=int, default=3)
args = parser.parse_args()
torch.set_num_threads(1)

started = time.perf_counter()
engine = ClaphaEngine(args.clapha)
print(f'boot {time.perf_counter() - started:.1f} s', flush=True)

# 1. raw engine: one tick at a time, nothing read
engine.reset(seed=1)
started, ticks = time.perf_counter(), 0
while ticks < 6000 and not engine.env.stalled:
    engine.env.step(1)
    ticks += 1
print(f'raw engine: {ticks / (time.perf_counter() - started):,.0f} ticks/s', flush=True)

# 2. a full State every 5 ticks (what the agent needs), no plays
state = engine.reset(seed=2)
started, decisions = time.perf_counter(), 0
while not state.finalized and state.tick < 7200:
    state = engine.step((), 5).state
    decisions += 1
spent = time.perf_counter() - started
print(f'with State every 5 ticks: {decisions / spent:,.0f} decisions/s = {5 * decisions / spent:,.0f} ticks/s '
      f'({1000 * spent / decisions:.2f} ms per decision)', flush=True)
print('sample state: tick', state.tick, 'players', [(p.owner, p.elixir, [(h.card, h.cost) for h in p.hand])
                                                   for p in state.players], flush=True)
print('sample entities:', state.entities[:3], flush=True)

# 3. self-play with the IL model: engine + State + features + model, one core
from il.agent import Agent, load  # noqa: E402
model, vocab = load(args.checkpoint)
agents = [Agent(model, vocab, owner) for owner in (0, 1)]
noh = Counter()
for game in range(args.games):
    state = engine.reset(seed=10 + game)
    for agent in agents:
        agent.reset()
    started, steps, tried, done = time.perf_counter(), 0, 0, 0
    while not state.finalized and state.tick < 7200:
        plays = [Play(a.owner, *c[1:]) for a in agents for c in [a.act(state)] if c[0] == 'card']
        transition = engine.step(plays, 5)
        tried, done, steps = tried + len(plays), done + sum(transition.executed), steps + 1
        state = transition.state
        for o in engine.env.observe()['objects'] if steps % 20 == 0 else ():
            if o['hp'] is None:
                noh[(o['cardId'], o['dataGlobalId'] // 1000000)] += 1
    spent = time.perf_counter() - started
    print(f'self-play game {game}: {spent:.1f} s, {steps} decisions ({1000 * spent / steps:.1f} ms each), '
          f'tick {state.tick}, result {state.result}, crowns {state.crowns}, plays {done}/{tried} executed', flush=True)
print('objects without hitpoints (card id, data class): count')
for key, n in noh.most_common(25):
    print(' ', key, n)
