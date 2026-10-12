"""Watches a PPO run (il.rl): evaluates every new snapshot, keeps the best one, rolls the learner back when it degrades.

    sbatch tools/satellite.sbatch          # runs it next to the satellite's collectors (see the file for the settings)

Every snapshot in <run>/pool plays the panel (--panel name=checkpoint, e.g. the General and IL models) with the live
deploy delay and the training policy: --games per panel model, on the same real deck pairs every time, the snapshot
playing --deck (a specialist) or the pair's decks. One line per snapshot goes to <run>/autopilot.jsonl: the score
against each panel model and il.defense's stats of the snapshot's games.

- Promotion: a snapshot that scores better than <run>/best.pt against the panel, without playing worse than the first
  evaluated checkpoint on behaviour (big pushes answered, plays per game), is copied to <run>/best.pt.
- Rollback: after ROLLBACK_AFTER snapshots in a row scoring DROP below the best (or the learner's entropy under
  ENTROPY_FLOOR), <run>/rollback.json tells the learner to restart from best.pt at half its learning rate.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import defense
from .arena import deck_pairs, run, summary

DROP = 0.06  # below the best's panel score; ~1.2 x the 95% interval of a 400-game score
ROLLBACK_AFTER = 2
ENTROPY_FLOOR = 0.05
ANSWERED_SLACK, PLAYS_SLACK = 0.03, 0.8  # behaviour guards against the first evaluated checkpoint


def write(path, copy_from=None, text=None):
    """Atomically: the learner may read the file at any moment."""
    tmp = path.with_name(path.name + '.tmp')
    shutil.copy(copy_from, tmp) if copy_from else tmp.write_text(text)
    tmp.rename(path)


def evaluate(args, checkpoint, pairs):
    row = {'checkpoint': checkpoint.stem, 'time': time.time()}
    games = []
    for name, path in args.panel:
        results = run(args.clapha, str(checkpoint), path, pairs, args.procs, decode='policy', report_every=0,
                      logs=str(args.run / 'autopilot-logs'), a_deck=args.deck)
        row[name] = summary(results)['score']
        games += results
    row['score'] = float(np.mean([row[name] for name, _ in args.panel]))
    row['defense'] = defense.summary([g['defense_a'] for g in games])
    return row


def acceptable(row, base):
    d, b = row['defense'], base['defense']
    return (d['answered_big'] >= b['answered_big'] - ANSWERED_SLACK
            and d['plays_per_game'] >= PLAYS_SLACK * b['plays_per_game'])


def entropy(run_dir):
    lines = (run_dir / 'metrics.jsonl').read_text().splitlines()
    return json.loads(lines[-1]).get('entropy') if lines else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('run', type=Path, help='the il.rl output directory')
    parser.add_argument('--start', type=Path, required=True, help='the checkpoint the run started from (the first best)')
    parser.add_argument('--panel', action='append', required=True, help='name=checkpoint, repeatable')
    parser.add_argument('--deck', help="the snapshot's deck (8 cards, id or id:el), as il.arena --a-deck")
    parser.add_argument('--games', type=int, default=200, help='per panel model')
    parser.add_argument('--procs', type=int, default=16)
    parser.add_argument('--pool', default=str(Path.home() / 'cr-engine/data/deck_pool.json'))
    args = parser.parse_args()
    args.run = args.run.expanduser()
    args.panel = [p.split('=', 1) for p in args.panel]
    pairs = list(enumerate(deck_pairs(args.pool, None, args.games, seed=1)))
    log_path, best_path = args.run / 'autopilot.jsonl', args.run / 'best.pt'
    rows = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    if not rows:
        rows.append(dict(evaluate(args, args.start, pairs), promoted=True))
        write(best_path, copy_from=args.start)
        log_path.write_text(json.dumps(rows[0]) + '\n')
    base, bad = rows[0], 0
    print(f"start {base['checkpoint']}: panel score {base['score']:.3f}, {base['defense']}", flush=True)
    while True:
        done = {r['checkpoint'] for r in rows}
        todo = [p for p in sorted((args.run / 'pool').glob('*.pt')) if p.stem not in done]
        if not todo:
            time.sleep(60)
            continue
        best = max((r for r in rows if r.get('promoted')), key=lambda r: r['time'])
        row = evaluate(args, todo[0], pairs)
        row['promoted'] = row['score'] > best['score'] and acceptable(row, base)
        if row['promoted']:
            write(best_path, copy_from=todo[0])
        bad = 0 if row['score'] >= best['score'] - DROP else bad + 1
        h = entropy(args.run)
        if (bad >= ROLLBACK_AFTER or (h is not None and h < ENTROPY_FLOOR)) and not (args.run / 'rollback.json').exists():
            write(args.run / 'rollback.json', text=json.dumps({'checkpoint': str(best_path), 'lr_scale': 0.5}))
            row['rollback'] = True
            bad = 0
        rows.append(row)
        with open(log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(f"{row['checkpoint']}: panel {row['score']:.3f} (best {best['score']:.3f})"
              f"{' PROMOTED' if row['promoted'] else ''}{' ROLLBACK' if row.get('rollback') else ''} | "
              f"{ {n: row[n] for n, _ in args.panel} } | {row['defense']}", flush=True)


if __name__ == '__main__':
    main()
