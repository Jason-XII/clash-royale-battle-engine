"""Model A vs model B on clapha engines (x86_64 Linux), with the live deploy delay.

    PYTHONPATH=. python -m il.arena ~/clapha-engine runs/il-003/best.pt runs/il-002/best.pt --games 400 --procs 16

Decks are the two players' opening hand + cycle of cached human games (one cached game per arena game),
or one fixed deck for both players with --deck (a mirror match); seats alternate so each model plays both
sides of each deck pair equally often. Abilities are not played
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


LEVEL = 16  # 75% of IL_Replay games: lvlcap = cardlvlmin = 16 (the rest 11); level 1 put towers far out of the data


def make_match(pair, seed):
    """A native match. `pair` is a deck pool entry (tools/build_deck_pool.py: cards with their evolution flags and
    the real game's levels) or two lists of 8 card ids (no evolutions, all at LEVEL). The template alone plays
    everything at level 1; the king tower level is separate (human games: 7728 hp at 16; level 1: 2400)."""
    match = json.loads((Path(__file__).parents[1] / 'standard_match.json').read_text())
    battle = match['battle']
    if isinstance(pair, dict):
        sp, cap, card_min, kt, exp = pair['decks'], pair['lvlcap'], pair['cardlvlmin'], pair['kt'], pair['exp']
    else:
        sp, cap, card_min, kt, exp = [[{'d': c} for c in d] for d in pair], LEVEL, LEVEL, [LEVEL] * 2, [LEVEL] * 2
    battle['lvlcap'], battle['cardlvlmin'] = cap, card_min
    for owner in (0, 1):
        battle[f'deck{owner}']['sp'] = sp[owner]
        battle[f'avatar{owner}']['expLevel'] = exp[owner]
        battle['hbd'][owner]['kt'] = kt[owner]
    match['rndSeed'] = seed
    return match


def deck_pairs(pool, cache, count, seed):
    """Deck pairs for `count` games: from the deck pool file when it exists (evolutions, real levels), else from
    the cache (card ids only, LEVEL, no evolutions)."""
    if pool and Path(pool).exists():
        entries = json.loads(Path(pool).read_text())
        rng = np.random.default_rng(seed)
        return [entries[i] for i in rng.choice(len(entries), count, replace=count > len(entries))]
    print(f'no deck pool at {pool}: cached decks without evolutions', flush=True)
    return decks(cache, count, seed)


def worker(index, args, jobs, out):
    log = os.open(f'{args.logs}/worker-{index}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
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
    for game, pair in jobs:
        match = make_match(pair, game + 1)
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


def mirror(deck_text):
    """A deck pool entry where both players use one deck at LEVEL: 8 comma-separated cards, id or id:el."""
    deck = [dict(zip(('d', 'el'), map(int, c.split(':')))) for c in deck_text.split(',')]
    assert len(deck) == 8, 'a deck has 8 cards'
    return {'decks': [deck, deck], 'lvlcap': LEVEL, 'cardlvlmin': LEVEL, 'kt': [LEVEL] * 2, 'exp': [LEVEL] * 2}


def run(clapha, a, b, pairs, procs=16, delay=D.DEPLOY_DELAY, decode='gate', out=None, report_every=20, logs='runs/arena-logs'):
    """Plays A against B on `pairs` [(game index, deck pair)] across `procs` engines; returns the game records."""
    from native_engine.clapha import raise_thread_limit
    raise_thread_limit()  # before any engine process starts, so every process gets it
    os.makedirs(logs, exist_ok=True)
    args = argparse.Namespace(clapha=clapha, a=a, b=b, delay=delay, decode=decode, logs=logs)
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    workers = [ctx.Process(target=worker, args=(i, args, pairs[i::procs], q)) for i in range(procs)]
    for w in workers:
        w.start()
    results, live = [], set(range(procs))
    f = open(out, 'a') if out else None
    while live:
        try:
            kind, index, value = q.get(timeout=5)
        except queue.Empty:
            for i in [i for i in live if workers[i].exitcode is not None]:
                print(f'worker {i} died (exit {workers[i].exitcode}); see {logs}/worker-{i}.log', flush=True)
                live.discard(i)
            continue
        if kind == 'done':
            live.discard(index)
            continue
        results.append(value)
        if f:
            f.write(json.dumps(dict(value, a=a, b=b, delay=delay)) + '\n')
            f.flush()
        if report_every and len(results) % report_every == 0:
            report(results)
    if f:
        f.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('a', help='checkpoint A')
    parser.add_argument('b', help='checkpoint B')
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--procs', type=int, default=16)
    parser.add_argument('--delay', type=int, default=D.DEPLOY_DELAY, help='decision -> landing ticks (0: instant)')
    parser.add_argument('--decode', choices=('gate', 'sample', 'argmax', 'policy'), default='gate',
                        help="policy: il.rl's training policy (wait with p(wait), else card x cell at its temperature)")
    parser.add_argument('--deck', help='8 comma-separated cards, id or id:el (el = evolution flag as in replays): '
                        'both players use this deck at level 16 (mirror match)')
    parser.add_argument('--pool', default='data/deck_pool.json', help='deck pairs as played (tools/build_deck_pool.py)')
    parser.add_argument('--cache', default='data/cache', help='deck pairs when there is no pool file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='runs/arena.jsonl')
    args = parser.parse_args()
    if args.deck:
        pairs = [(g, mirror(args.deck)) for g in range(args.games)]
    else:
        pairs = list(enumerate(deck_pairs(args.pool, args.cache, args.games, args.seed)))
    report(run(args.clapha, args.a, args.b, pairs, args.procs, args.delay, args.decode, args.out))


def summary(results):
    """Score of A with a 95% interval, and how the games went (both sides' activity included)."""
    n = len(results)
    a = sum(r['winner'] == 'A' for r in results)
    b = sum(r['winner'] == 'B' for r in results)
    score = (a + 0.5 * (n - a - b)) / n
    mean = lambda k: sum(r[k] for r in results) / n
    minutes = sum(max(r['tick'] - 90, 1) for r in results) / 1200
    ticks = sorted(r['tick'] for r in results)
    return dict(games=n, a_wins=a, b_wins=b, score=round(score, 4), ci95=round(1.96 * math.sqrt(score * (1 - score) / n), 4),
                crowns_a=round(mean('crowns_a'), 2), crowns_b=round(mean('crowns_b'), 2),
                plays_per_min_a=round(sum(r['plays_a'] for r in results) / minutes, 2),
                plays_per_min_b=round(sum(r['plays_b'] for r in results) / minutes, 2),
                median_tick=ticks[n // 2], three_crowns=round(sum(max(r['crowns_a'], r['crowns_b']) == 3 for r in results) / n, 3))


def report(results):
    s = summary(results)
    print(f"{s['games']} games: A {s['a_wins']} wins, B {s['b_wins']} wins | A scores {s['score']:.1%} +/- {s['ci95']:.1%} | "
          f"crowns A {s['crowns_a']} B {s['crowns_b']} | plays/min A {s['plays_per_min_a']} B {s['plays_per_min_b']} | "
          f"median tick {s['median_tick']}, 3-crowns {s['three_crowns']:.0%}", flush=True)


if __name__ == '__main__':
    main()
