"""Rebuild IL_Replay parts in the engine and store raw per-decision states (not features).

    python -m native_engine.build_cache data/IL_Replay/replays data/cache            # all parts
    python -m native_engine.build_cache data/IL_Replay/replays data/cache --parts 0 1 --limit 50

One compressed shard per 1000 replays of a parquet part (`part-000000-0.npz` ...); finished
shards are skipped, so the job resumes after an interruption. Arrays of all replays are concatenated; replay i
owns rows `off[i]:off[i+1]` of each table (see il/data.py for the reader).
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .engine import Engine
from .replay import rebuild

ENTITY_TYPES = ('troop', 'building', 'projectile', 'area')
# Per-entity columns: state row, native id, owner, object type id, x, y, hp, max hp,
# entity type, deploy ms, attack cooldown ms, target id. -1 = unknown/none.
ENTITY_COLUMNS = ('state', 'id', 'owner', 'object', 'x', 'y', 'hp', 'max_hp', 'type', 'deploy_ms',
                  'cooldown_ms', 'target')
# Human commands: execute tick, owner, card (-1 = ability), x, y, executed by native (0/1).
ACTION_COLUMNS = ('tick', 'owner', 'card', 'x', 'y', 'executed')


def _int(value):
    return -1 if value is None else value


def encode(result):
    """One rebuilt replay -> dict of numpy arrays (all int32/int64)."""
    states = result['states']
    T = len(states)
    tick = np.array([s.tick for s in states], np.int32)
    elixir = np.array([[p.elixir for p in s.players] for s in states], np.int32)
    crowns = np.array([s.crowns for s in states], np.int32)
    hand = np.full((T, 2, 4, 2), -1, np.int32)  # (card, cost) by hand slot
    cycle = np.full((T, 2, 4), -1, np.int32)
    entities, plays = [], []
    for t, s in enumerate(states):
        for o, p in enumerate(s.players):
            for c in p.hand:
                hand[t, o, c.slot] = (c.card, c.cost)
            cycle[t, o, :len(p.cycle)] = p.cycle[:4]
        for e in s.entities:
            entities.append((t, e.id, e.owner, e.card, e.x, e.y, _int(e.hp), _int(e.max_hp),
                             ENTITY_TYPES.index(e.entity_type), _int(e.deployment_remaining_ms),
                             _int(e.attack_cooldown_ms), _int(e.target_id)))
        plays.extend((t, p.tick, p.owner, p.card) for p in s.plays)
    actions = [(c.tick, c.owner, -1 if c.card is None else c.card, c.x, c.y, r.get('state') == 'succeeded')
               for c, r in zip(result['commands'], result['receipts'])]
    final = states[-1]
    return {
        'tick': tick, 'elixir': elixir, 'crowns': crowns, 'hand': hand, 'cycle': cycle,
        'entities': np.array(entities, np.int32).reshape(-1, len(ENTITY_COLUMNS)),
        'plays': np.array(plays, np.int32).reshape(-1, 4),
        'actions': np.array(actions, np.int32).reshape(-1, len(ACTION_COLUMNS)),
        # Per-replay scalars: native result (-1 draw/unknown), final crowns, fidelity flags.
        'meta': np.array([_int(final.result), *final.crowns, result['winner_matches'],
                          result['crowns_matches']], np.int32)[None],
    }


def write_shard(path, tags, replays):
    arrays = {'tags': np.array(tags)}
    for key in replays[0]:
        parts = [r[key] for r in replays]
        arrays[key] = np.concatenate(parts)
        arrays[key + '_off'] = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


SHARD_REPLAYS = 1000
START_ENGINE = Path(__file__).resolve().parent / 'tools' / 'start_engine.sh'


def rebuild_with_recovery(engine, payload, recoveries=2):
    """An OSError means the engine is gone (emulator crash, socket timeout), not a bad replay:
    restart the emulator and engine, then retry. Raises if recovery keeps failing."""
    for attempt in range(recoveries + 1):
        try:
            return rebuild(engine, payload)
        except OSError as error:
            engine.close()
            if attempt == recoveries:
                raise
            print(f'engine lost ({type(error).__name__}: {error}); restarting it', flush=True)
            subprocess.run([str(START_ENGINE)], check=True, capture_output=True)


def build_shard(engine, rows, out):
    tags, replays, failures = [], [], {}
    started = time.perf_counter()
    for row in rows:
        try:
            result = rebuild_with_recovery(engine, json.loads(row['payload_json']))
        except (OSError, subprocess.CalledProcessError):
            raise  # engine down: stop without writing a shard that silently misses replays
        except Exception as error:  # one bad replay must not stop the part; record why
            key = f'{type(error).__name__}: {str(error)[:80]}'
            failures[key] = failures.get(key, 0) + 1
            engine.close()  # drop a possibly half-read connection
            continue
        tags.append(result['tag'])
        replays.append(encode(result))
    if not replays:
        raise RuntimeError(f'no replay of {out.name} could be rebuilt: {failures}')
    write_shard(out, tags, replays)
    exact = sum(int(r['meta'][0, 3] and r['meta'][0, 4]) for r in replays)
    return {'shard': out.name, 'rows': len(rows), 'rebuilt': len(replays), 'exact': exact,
            'failures': failures, 'seconds': round(time.perf_counter() - started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replays', type=Path, help='directory of part-*.parquet')
    parser.add_argument('out', type=Path)
    parser.add_argument('--parts', type=int, nargs='*', help='part numbers (default: all)')
    parser.add_argument('--limit', type=int, help='replays per part, for smoke tests')
    parser.add_argument('--port', type=int, default=26789)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts = sorted(args.replays.glob('part-*.parquet'))
    if args.parts is not None:
        parts = [p for p in parts if int(p.stem.split('-')[1]) in args.parts]
    import pyarrow.parquet as pq
    engine = Engine(port=args.port)
    for parquet in parts:
        rows = None
        for k in range(-(-pq.ParquetFile(parquet).metadata.num_rows // SHARD_REPLAYS)):
            out = args.out / f'{parquet.stem}-{k}.npz'
            if out.exists():
                continue
            if rows is None:
                rows = pq.read_table(parquet, columns=['payload_json']).to_pylist()
            chunk = rows[k * SHARD_REPLAYS:(k + 1) * SHARD_REPLAYS][:args.limit]
            if not chunk:
                break
            report = build_shard(engine, chunk, out)
            print(json.dumps(report), flush=True)
            with open(args.out / 'build_log.jsonl', 'a') as log:
                log.write(json.dumps(report) + '\n')
            if args.limit:
                break


if __name__ == '__main__':
    main()
