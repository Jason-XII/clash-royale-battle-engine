"""Deck pairs of real IL_Replay games as played: each card with its evolution flag ('el'), plus the game's
level cap and king tower levels. Mac only (FirstLight's converter); copy the result to the training node.

    PYTHONPATH=. python tools/build_deck_pool.py data/IL_Replay/replays data/deck_pool.json --per-part 600
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from native_engine.replay import match_dict, prepare_collected_replay

parser = argparse.ArgumentParser()
parser.add_argument('replays', type=Path)
parser.add_argument('out', type=Path)
parser.add_argument('--per-part', type=int, default=600)
args = parser.parse_args()
pool = []
for part in sorted(args.replays.glob('part-*.parquet')):
    for row in pq.read_table(part, columns=['payload_json']).slice(0, args.per_part).to_pylist():
        try:
            replay = prepare_collected_replay(json.loads(row['payload_json'])).replay
        except Exception:  # ponytail: a replay the converter rejects is simply skipped
            continue
        b = match_dict(replay, (replay.episode_config.deck0, replay.episode_config.deck1), replay.episode_config.seed)['battle']
        pool.append({'decks': [b['deck0']['sp'], b['deck1']['sp']], 'lvlcap': b['lvlcap'], 'cardlvlmin': b['cardlvlmin'],
                     'kt': [b['hbd'][0]['kt'], b['hbd'][1]['kt']], 'exp': [b['avatar0']['expLevel'], b['avatar1']['expLevel']]})
    print(part.name, len(pool), flush=True)
args.out.write_text(json.dumps(pool, separators=(',', ':')))
print('wrote', len(pool), 'deck pairs to', args.out)
