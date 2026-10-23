"""The player views in an IL dataset whose deck is a given card set (23 Oct), for a specialist's imitation step
(settings.json "bc_index"): the deck is every card the view's hand and next cards show over the game (a replay's own
hand is reconstructed: cards not known yet are UNKNOWN), a match when all are in the set and at least 7 of the 8 show;
the outcome is its value target. NumPy only; run it in a job (it opens every archive).

    python deck_index.py DATASET OUT.json --cards HogRider,Musketeer,Cannon,Fireball,Log,Skeletons,IceGolemite,IceSpirits
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / 'assets'


def deck_of(job):
    root, entry, index = job
    with np.load(root / entry['file'], allow_pickle=False) as z:
        p = f"p{entry['owner']}_"
        cards = {int(c) for c in np.unique(np.concatenate([z[p + 'x_hand_card'].ravel(), z[p + 'x_cycle_card'].ravel()]))}
        outcome = float(z[p + 't_value'][0])
    return {index[c] for c in cards if c in index}, outcome


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--cards', required=True, help='card names, as in assets/replay_cards.json')
    p.add_argument('--procs', type=int, default=8)
    a = p.parse_args()
    names = {v['name']: int(k) for k, v in json.loads((ASSETS / 'replay_cards.json').read_text())['cards'].items()}
    want = {names[n] for n in a.cards.split(',')}
    manifest = json.loads((a.data / 'manifest.json').read_text())
    index = {n + 2: i for n, i in enumerate(sorted({int(i) for i in manifest['schema']['vocab']}))}  # as common.Vocab
    entries = manifest['entries']
    with Pool(a.procs) as pool:
        decks = pool.map(deck_of, [(a.data, e, index) for e in entries], chunksize=256)
    rows = [dict(e, outcome=o) for e, (d, o) in zip(entries, decks) if d <= want and len(d) >= len(want) - 1]
    out = dict(data=str(a.data), cards=sorted(want), views=len(entries), entries=rows,
               wins=sum(r['outcome'] > 0 for r in rows), losses=sum(r['outcome'] < 0 for r in rows),
               validation=sum(r['validation'] for r in rows), decisions=sum(r['decisions'] for r in rows))
    a.out.write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != 'entries'}, indent=1))


if __name__ == '__main__':
    main()
