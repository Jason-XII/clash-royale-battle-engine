"""Evaluate PPO snapshots against a reference model in the arena (x86_64 node), with the live deploy delay.

    cd ~/cr-engine && nice -n 10 env PYTHONPATH=. python -m il.eval ~/clapha-engine ~/cr-engine/runs/ppo-general \
        --reference ~/cr-engine/runs/il-002/best.pt --latest

Every snapshot in <run>/pool not evaluated yet (and with --latest the current latest.pt) plays two arenas against
the reference: 'mixed' (the same real deck pairs for every snapshot) and 'ebarbs' (the user's deck, mirror). Both
sides decode with the PPO training policy. One line per snapshot and arena goes to <run>/eval.jsonl. The
reference's activity (plays_per_min_b) is reported too: a high score against a model that stopped playing proves
nothing (FirstLight's lesson).
"""
import argparse
import json
import os
from pathlib import Path

import torch

from .arena import deck_pairs, mirror, run, summary

# Elite Barbarians, Giant Skeleton, Battle Ram, Zappies, Wizard, Mother Witch, Barbarian Barrel, Vines, with the
# evolution setup 1759 of 1768 replays of this deck use (Wizard el=2, Battle Ram and Elite Barbarians el=1).
EBARBS = '26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('run', help='a PPO run directory (il.rl output)')
    parser.add_argument('--reference', required=True, help='checkpoint B, e.g. the IL model PPO started from')
    parser.add_argument('--games', type=int, default=200, help='per arena')
    parser.add_argument('--procs', type=int, default=6, help='engines (training uses the rest of the node)')
    parser.add_argument('--latest', action='store_true', help='also evaluate latest.pt')
    parser.add_argument('--pool', default=os.path.expanduser('~/cr-engine/data/deck_pool.json'))
    args = parser.parse_args()
    run_dir = Path(args.run).expanduser()
    log_path = run_dir / 'eval.jsonl'
    done = set()
    if log_path.exists():
        done = {(r['checkpoint'], r['arena']) for r in map(json.loads, log_path.read_text().splitlines())}
    todo = [(p.stem, p) for p in sorted((run_dir / 'pool').glob('*.pt'))]
    if args.latest:
        latest = run_dir / 'latest.pt'
        todo.append((f"latest@{torch.load(latest, map_location='cpu', weights_only=False)['update']}", latest))
    arenas = {'mixed': list(enumerate(deck_pairs(args.pool, None, args.games, seed=1))),
              'ebarbs': [(g, mirror(EBARBS)) for g in range(args.games)]}
    for label, path in todo:
        for arena, pairs in arenas.items():
            if (label, arena) in done:
                continue
            results = run(args.clapha, str(path), args.reference, pairs, args.procs, decode='policy',
                          report_every=0, logs=str(run_dir / 'eval-logs'))
            row = dict(checkpoint=label, arena=arena, reference=args.reference, **summary(results))
            with open(log_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"{label:>14} {arena:>6}: scores {row['score']:.1%} +/- {row['ci95']:.1%} vs reference | "
                  f"plays/min {row['plays_per_min_a']} (reference {row['plays_per_min_b']}) | "
                  f"median tick {row['median_tick']}, 3-crowns {row['three_crowns']:.0%}", flush=True)


if __name__ == '__main__':
    main()
