"""Do games played in ClaphaEngine give a clapha-il model the inputs imitation learning gave it? One game of a checkpoint
against itself: the inputs each agent built while playing (PlayerView from ClaphaEngine.record and the commands issued so
far) against clapha_il.features.game_features on the same game stored as convert.py stores replays. Prints every input
that differs; the reconstructed hand may show UNKNOWN where the read hand is known (a replay's start order is guessed).

    PYTHONPATH=. python tools/check_clapha_inputs.py ~/clapha-engine RUN/latest.pt [--deck 26000043:1,...]
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clapha')
    ap.add_argument('checkpoint')
    ap.add_argument('--deck', default='26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026')
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    from native_engine import Play
    from native_engine.clapha import ClaphaEngine
    from il import policy
    from il.arena import make_match, mirror
    FE = policy.FE
    model, vocab, _ = policy.load(args.checkpoint)
    legal = policy.legal_fn()
    engine = ClaphaEngine(args.clapha)
    match = make_match(mirror(args.deck), args.seed)
    state = engine.reset(match=match)
    inputs = policy.Inputs(engine, match, {0: vocab, 1: vocab}, legal)
    agents = [policy.Agent(model, o) for o in (0, 1)]
    seen, records, queued, events = [[], []], [], [], []  # what each player's inputs were while playing
    original = inputs.engine.record
    inputs.engine.record = lambda: records.append(original()) or records[-1]  # keep every record the inputs read
    while not state.finalized and state.tick < 7200:
        rows = inputs(state)
        events += [(p.tick, p.owner, p.card) for p in state.plays]
        queued += [p for p in engine.pending if p.get('card') is not None and not any(p is q for q in queued)]
        for o in (0, 1):
            seen[o].append(rows[o])
        choices = [a.act(rows[a.owner]) for a in agents]
        plays = [Play(a.owner, *c[1:]) for a, c in zip(agents, choices) if c[0] == 'card']
        state = engine.step(plays, 5, delay=(23, 27)).state
    events += [(p.tick, p.owner, p.card) for p in state.plays]
    landed = []  # each play that left a hand, with the queued command it came from (its due tick is when it landed)
    for tick, owner, card in events:
        match_ = [p for p in queued if p['owner'] == owner and p['card'] == card and p['due'] <= tick
                  and not any(p is q for _, q in landed)]
        if match_:
            landed.append((match_[-1]['due'], match_[-1]))
    header = {'commands': [{'kind': 'card', 'owner': p['owner'], 'card_id': p['card'], 'x': p['x'], 'y': p['y'],
                            'tick': due, 'executed_tick': due} for due, p in landed],
              'episode': {f'deck{o}_form_availability': [c.get('el', 0) for c in match['battle'][f'deck{o}']['sp']]
                          for o in (0, 1)}, 'terminal': {'winner': state.result}}
    game = {'header': header, 'decisions': records}
    bad = 0
    for o in (0, 1):
        x, _, _ = FE.game_features(game, o, vocab, legal=legal)
        live = {k: np.stack([s[k] for s in seen[o]]) for k in seen[o][0]}
        for k in sorted(x):
            a, b = x[k], live[k]
            if k in ('hand_card', 'cycle_card'):  # the replay path hides what the start order leaves uncertain
                b = np.where(a == 1, 1, b)
            if k == 'hand_float':  # and flags it (column 4)
                a, b = np.delete(a, 4, -1), np.delete(b, 4, -1)
            diff = (a != b) if a.dtype.kind in 'biu' else ~np.isclose(a, b, atol=1e-6)
            rows = np.flatnonzero(diff.reshape(len(a), -1).any(1))
            if len(rows):
                bad += 1
                print(f'owner {o} {k}: {len(rows)} of {len(a)} decisions differ, first at tick {records[rows[0]]["tick"]}')
                for r in rows[:3]:
                    t = records[r]['tick']
                    near = [(p['due'], p['card'], 'landed' if any(p is q for _, q in landed) else 'never landed')
                            for p in queued if p['owner'] == o and abs(p['due'] - t) <= 40]
                    print(f'    tick {t}: replay {a[r].tolist()} played {b[r].tolist()}; own plays near: {near}')
        unknown = (x['hand_card'] == 1).mean()
        print(f'owner {o}: {len(seen[o])} decisions, {sum(1 for _, p in landed if p["owner"] == o)} plays landed, '
              f'reconstructed hand unknown {unknown:.1%}')
    print('INPUTS:', 'SAME' if not bad else f'{bad} inputs differ')
    sys.stdout.flush()
    import os
    os._exit(0)  # the engine segfaults in its exit-time destructors


if __name__ == '__main__':
    main()
