"""Who plays whom, and with which decks: the learner's opponents and their PFSP weights, each collector game's two
models, the loop's evaluation matches, and a specialist's deck seat."""
import json
from pathlib import Path

import numpy as np


def evaluation(run):
    """The loop's evaluation matches (RUN/eval.json, written by loop.py): [{'id', 'a': [name, path], 'b': [name,
    path], 'games'}], or [] (none, or caught mid-write)."""
    try:
        return json.loads((run / 'eval.json').read_text()).get('matches', [])
    except (OSError, ValueError):
        return []


def route_collector(opponent, opponents, rng, servers):
    """Partition the PFSP probability interval across servers, then route within the sampled opponent's span.

    Each server receives equal expected game traffic. With M frozen opponents and S servers there are at most
    M + S - 1 frozen-model/server pairs, instead of M*S; no shared manager or game-ending barrier is needed.
    """
    if opponent == 'current':
        return rng.randrange(servers)
    names = sorted(opponents)
    total = sum(opponents[name][1] for name in names)
    before = sum(opponents[name][1] for name in names[:names.index(opponent)])
    point = before + rng.random() * opponents[opponent][1]
    return min(servers - 1, int(point / total * servers))


def deck_seat(models, judged, rng):
    """The seat that plays a specialist's deck: a learning seat, or in an evaluation game (two frozen seats) the
    candidate, A, so a gate judges it with its deck."""
    if judged:
        return next(o for o in (0, 1) if models[o][0] == judged['a'][0])
    return rng.choice([o for o in (0, 1) if models[o][0] == 'current'])


def opponent_paths(out, init, s):
    """{name: checkpoint}: the IL model, the latest snapshots and the league (RUN/league: every champion loop.py
    promoted, kept for good). An exploiter (s.against, another run's directory) plays only that run's champion and
    follows its promotions."""
    if s.against:
        spec = json.loads((Path(s.against) / 'champion.json').read_text())
        return {spec['name']: spec['path']}
    return {'il': init, **{p.stem: str(p) for p in sorted((out / 'pool').glob('v*.pt'))[-s.pool_size:]},
            **{p.stem: str(p) for p in sorted((out / 'league').glob('*.pt'))}}


def pfsp(paths, results):
    """{name: (checkpoint, weight)} over `paths` ({name: checkpoint}): weight (1 - the learner's win rate)^2 + 0.05, the
    win rate from `results` ({name: recent outcomes, +-1}; 0.5 before any)."""
    win = {n: (np.mean(results[n]) + 1) / 2 if results[n] else 0.5 for n in paths}
    return {n: (path, (1 - win[n]) ** 2 + 0.05) for n, path in paths.items()}


def pick_models(opponents, matches, s, rng):
    """One game's seats [(name, checkpoint or None), (name, ...)] and the evaluation match it plays (or None). With
    s['eval_frac'] of the games a loop evaluation (`matches`: two frozen models, nobody learns); else a frozen opponent
    drawn by `opponents`' weights against the learner ('current'), or (s['self_play'] of them) the learner twice."""
    judged = rng.choices(matches, [m['games'] for m in matches])[0] \
        if matches and rng.random() < s['eval_frac'] else None
    names = list(opponents)
    if judged:
        models = [tuple(judged['a']), tuple(judged['b'])]
        if rng.random() < 0.5:
            models.reverse()
    elif names and rng.random() >= s['self_play']:
        name = rng.choices(names, [opponents[n][1] for n in names])[0]
        models = [('current', None), (name, opponents[name][0])]
        if rng.random() < 0.5:
            models.reverse()
    else:
        models = [('current', None)] * 2
    return models, judged
