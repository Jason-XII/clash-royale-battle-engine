"""Paired-seat native battles between two trained models (Linux, inside a Slurm allocation).

    python arena.py A.pt B.pt --engine ~/clapha-engine --pool data/deck_pool.json --games 20 --out runs/arena.jsonl
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from cr.engine import Engine, Game
from cr.policy import Agent, load


def evaluate(engine, models, pair, seed, a_owner, device='cpu', a_overlap=False):
    engine.overlap = [a_overlap and owner == a_owner for owner in (0, 1)]
    seats = [0 if owner == a_owner else 1 for owner in (0, 1)]
    game = Game(engine, pair, seed, {owner: models[index][1] for owner, index in enumerate(seats)})
    agents = [Agent(models[index][0], owner, device) for owner, index in enumerate(seats)]
    plays, refused = [0, 0], [0, 0]
    while not game.over:
        rows = game.views()
        submitted, transition = game.step({agent.owner: agent.act(rows[agent.owner]) for agent in agents})
        for action, ok in zip(submitted, transition.executed):
            refused[action.owner] += not ok
        for play in engine.played:
            plays[play['owner']] += 1
    state = game.state
    if not state.finalized:
        raise RuntimeError('native battle did not finalize before tick 7200')
    winner = None if state.result not in (0, 1) else 'A' if state.result == a_owner else 'B'
    return dict(winner=winner, a_owner=a_owner, crowns_a=state.crowns[a_owner], crowns_b=state.crowns[1-a_owner],
                plays_a=plays[a_owner], plays_b=plays[1-a_owner], refused_a=refused[a_owner], refused_b=refused[1-a_owner],
                tick=state.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True, help='deck pairs with actual levels and evolution flags')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--a-temperature', type=float, default=1.0, help="A's card and tile temperature")
    parser.add_argument('--b-temperature', type=float, default=1.0,
                        help="B's card and tile temperature (< 1: sharper; when to play stays sampled as trained)")
    parser.add_argument('--deck', type=Path, help="A always plays this deck (JSON, as in the pool), B the pair's second deck")
    parser.add_argument('--device', default='cpu', help='where both models run (cuda: big models; processes can share a GPU)')
    parser.add_argument('--a-overlap', action='store_true', help='A may tap a card while another is still in flight')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error('games must be positive and even for paired seats')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pool = json.loads(args.pool.read_text())
    if not pool:
        parser.error('deck pool is empty')
    deck = args.deck and json.loads(args.deck.read_text())
    models = [load(path, args.device)[:2] for path in (args.a, args.b)]
    models[0][0].temperature, models[1][0].temperature = args.a_temperature, args.b_temperature
    models[0][0].overlap = args.a_overlap  # the sampler too: it would wait while A's card is in flight
    engine = Engine(args.engine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.out.open('x') as stream:
        for pair_index, index in enumerate(rng.integers(len(pool), size=args.games // 2)):
            for seat in (0, 1):
                pair = pool[index]
                if deck:
                    pair = dict(pair, decks=[deck, pair['decks'][1]][::1 if seat == 0 else -1])
                result = dict(game=len(results), **evaluate(engine, models, pair, args.seed + pair_index, seat, args.device,
                                                               args.a_overlap))
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(json.dumps(result), flush=True)
    wins = sum(r['winner'] == 'A' for r in results)
    draws = sum(r['winner'] is None for r in results)
    print(json.dumps(dict(games=len(results), wins=wins, losses=len(results)-wins-draws, draws=draws,
                          score=(wins + 0.5 * draws) / len(results),
                          refused_per_game_a=sum(r['refused_a'] for r in results) / len(results),
                          refused_per_game_b=sum(r['refused_b'] for r in results) / len(results))), flush=True)
    sys.stdout.flush()
    os._exit(0)  # Native exit destructors are unsafe; all result files have been flushed.


if __name__ == '__main__':
    main()
