"""Model A vs model B on clapha engines (x86_64 Linux), with the live deploy delay.

    PYTHONPATH=. python -m il.arena ~/clapha-engine runs/il-003/best.pt runs/il-002/best.pt --games 400 --procs 16

Decks are the two players' opening hand + cycle of cached human games (one cached game per arena game);
seats alternate so each model plays both sides of each deck pair equally often. Abilities are not played
(clapha does not support them yet).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time

import numpy as np

from . import data as D


def decks(cache, count, seed):
    """`count` deck pairs of cached human games, spread over shards: [[8 ids of owner 0], [8 ids of owner 1]]."""
    rng = np.random.default_rng(seed)
    shards = sorted(Path(cache).glob('part-*.npz'))
    pairs = []
    for shard in rng.permutation(len(shards))[:max(1, count // 200)]:  # ~200 games from each shard used
        with np.load(shards[shard]) as z:
            first = z['tick_off'][:-1]
            hand, cycle = z['hand'][first][..., 0], z['cycle'][first]
        pairs += [[[int(c) for c in list(hand[g, o]) + list(cycle[g, o]) if c >= 0] for o in (0, 1)]
                  for g in range(len(first))]
    return [pairs[i] for i in rng.choice(len(pairs), count, replace=count > len(pairs))]


def worker(index, args, jobs, out):
    log = os.open(f'runs/arena-logs/worker-{index}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    time.sleep(index // 4 * 8)  # boot 4 engines at a time
    import torch
    from native_engine import Play
    from native_engine.clapha import ClaphaEngine
    from .agent import Agent, load
    torch.set_num_threads(1)
    engine = ClaphaEngine(args.clapha)
    models = [load(args.a), load(args.b)]
    template = json.loads((Path(__file__).parents[1] / 'standard_match.json').read_text())
    for game, (deck0, deck1) in jobs:
        match = json.loads(json.dumps(template))
        match['battle']['deck0']['sp'] = [{'d': c} for c in deck0]
        match['battle']['deck1']['sp'] = [{'d': c} for c in deck1]
        match['rndSeed'] = game + 1
        a_owner = game % 2
        agents = [Agent(*models[0 if owner == a_owner else 1], owner, decode=args.decode) for owner in (0, 1)]
        state = engine.reset(match=match)
        started, plays = time.perf_counter(), [0, 0]
        while not state.finalized and state.tick < 7200:
            chosen = [Play(a.owner, *c[1:]) for a in agents for c in [a.act(state)] if c[0] == 'card']
            transition = engine.step(chosen, 5, delay=args.delay)
            for p, ok in zip(chosen, transition.executed):
                plays[p.owner] += ok
            state = transition.state
        winner = None if state.result not in (0, 1) else 'A' if state.result == a_owner else 'B'
        out.put(('game', index, dict(game=game, winner=winner, a_owner=a_owner,
                                     crowns_a=state.crowns[a_owner], crowns_b=state.crowns[1 - a_owner],
                                     plays_a=plays[a_owner], plays_b=plays[1 - a_owner], tick=state.tick,
                                     seconds=time.perf_counter() - started)))
    out.put(('done', index, None))
    out.close()
    out.join_thread()
    os._exit(0)  # the engine segfaults in its exit-time destructors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('a', help='checkpoint A')
    parser.add_argument('b', help='checkpoint B')
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--procs', type=int, default=16)
    parser.add_argument('--delay', type=int, default=D.DEPLOY_DELAY, help='decision -> landing ticks (0: instant)')
    parser.add_argument('--decode', choices=('gate', 'sample', 'argmax'), default='gate')
    parser.add_argument('--cache', default='data/cache')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='runs/arena.jsonl')
    args = parser.parse_args()
    os.makedirs('runs/arena-logs', exist_ok=True)
    pairs = list(enumerate(decks(args.cache, args.games, args.seed)))
    ctx = mp.get_context('fork')
    out = ctx.Queue()
    workers = [ctx.Process(target=worker, args=(i, args, pairs[i::args.procs], out)) for i in range(args.procs)]
    for w in workers:
        w.start()
    results, live = [], set(range(args.procs))
    with open(args.out, 'a') as f:
        while live:
            try:
                kind, index, value = out.get(timeout=5)
            except queue.Empty:
                for i in [i for i in live if workers[i].exitcode is not None]:
                    print(f'worker {i} died (exit {workers[i].exitcode}); see runs/arena-logs/worker-{i}.log', flush=True)
                    live.discard(i)
                continue
            if kind == 'done':
                live.discard(index)
                continue
            results.append(value)
            f.write(json.dumps(dict(value, a=args.a, b=args.b, delay=args.delay)) + '\n')
            if len(results) % 20 == 0:
                report(results, args)
    report(results, args)


def report(results, args):
    n = len(results)
    a = sum(r['winner'] == 'A' for r in results)
    b = sum(r['winner'] == 'B' for r in results)
    score = (a + 0.5 * (n - a - b)) / n
    half = 1.96 * math.sqrt(score * (1 - score) / n)
    mean = lambda k: sum(r[k] for r in results) / n
    print(f'{n} games: A {a} wins, B {b} wins, {n - a - b} draws | A scores {score:.1%} +/- {half:.1%} | '
          f'crowns A {mean("crowns_a"):.2f} B {mean("crowns_b"):.2f} | plays A {mean("plays_a"):.1f} '
          f'B {mean("plays_b"):.1f} | {mean("seconds"):.1f} s/game', flush=True)


if __name__ == '__main__':
    main()
