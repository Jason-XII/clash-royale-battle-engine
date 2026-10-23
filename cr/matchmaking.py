"""Who plays whom, and with which decks: the learner's opponents and their weights (fixed shares, PFSP), each
collector game's two models, the loop's evaluation matches, and a specialist's seat and opponent decks."""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

FOCUS = 20         # a specialist's focus decks: the pool's most common card sets
HARD_BELOW = 0.55  # hard decks (FirstLight's rule): smoothed score (w + 2) / (n + 4) below this,
HARD_TOP = 128     # the most pressing ones, by priority (HARD_BELOW - score) x n / (n + 4), drawn by priority^2
RECENT = 32        # games per deck the scores keep (the newest)


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


def weights(paths, results, fixed=None):
    """{name: (checkpoint, weight)}: the opponents in `fixed` ({name: [checkpoint, share]}) at their shares of the games,
    the rest of the games by PFSP over the other `paths` ({name: checkpoint}): weight (1 - the learner's win rate)^2 +
    0.05, the win rate from `results` ({name: recent outcomes, +-1}; 0.5 before any), scaled to what the shares leave."""
    fixed = fixed or {}
    if not 0 <= sum(share for _, share in fixed.values()) <= 1:
        raise ValueError(f'opponent shares must add up to at most 1: {fixed}')
    rest = {n: path for n, path in paths.items() if n not in fixed}
    win = {n: (np.mean(results[n]) + 1) / 2 if results[n] else 0.5 for n in rest}
    pfsp = {n: (1 - win[n]) ** 2 + 0.05 for n in rest}
    scale = (1 - sum(share for _, share in fixed.values())) / sum(pfsp.values()) if fixed and pfsp else 1.0
    return {**{n: (path, share) for n, (path, share) in fixed.items()}, **{n: (rest[n], w * scale) for n, w in pfsp.items()}}


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


# ---- a specialist's opponent decks --------------------------------------------------------------------------------------
def deck_key(deck):
    """A deck's card set (forms aside), as one string."""
    return '-'.join(str(c) for c in sorted(c['d'] for c in deck))


class Decks:
    """The pool's decks by card set. pick(): a specialist's learning game draws its opponent's deck from the FOCUS most
    common card sets (uniformly), from the hard ones (`hard`: {key: weight}), or keeps the pair's own (each deck as often
    as humans played it), by the shares in `mix` ({'focus': share, 'hard': share}; the rest: the pair's own). A drawn
    set comes with one of its pool seats at random: its forms and its pair's levels, as humans had them."""

    def __init__(self, pool):
        self.pool, self.seats = pool, defaultdict(list)
        for i, pair in enumerate(pool):
            for side, deck in enumerate(pair['decks']):
                self.seats[deck_key(deck)].append((i, side))
        self.focus = sorted(self.seats, key=lambda k: (-len(self.seats[k]), k))[:FOCUS]

    def pick(self, pair, side, mix, hard, rng):
        """-> (the pair to play, its deck on `side` replaced as drawn, or `pair` as it is)."""
        u = rng.random()
        if u < mix.get('focus', 0):
            key = rng.choice(self.focus)
        elif u < mix.get('focus', 0) + mix.get('hard', 0) and hard:
            keys = sorted(hard)
            key = rng.choices(keys, [hard[k] for k in keys])[0]
        else:
            return pair
        i, theirs = rng.choice(self.seats[key])
        drawn = self.pool[i]
        return dict(drawn, decks=[drawn['decks'][theirs] if o == side else d for o, d in enumerate(drawn['decks'])])


def hard_decks(scores):
    """{key: weight} of the hard decks from `scores` ({key: recent scores of the specialist against it, 1 win, 0.5
    draw, 0 loss})."""
    priority = {}
    for key, xs in scores.items():
        n = len(xs)
        rate = (sum(xs) + 2) / (n + 4)
        if rate < HARD_BELOW:
            priority[key] = (HARD_BELOW - rate) * n / (n + 4)
    top = sorted(priority, key=lambda k: (-priority[k], k))[:HARD_TOP]
    return {k: priority[k] ** 2 for k in top}


class DeckScores:
    """The learner's record of a specialist's learning games by opponent deck: rank 0 adds every update's results and
    writes RUN/deck-scores.json ({key: recent scores}; read back on a restart) and RUN/hard-decks.json (hard_decks(),
    for the collectors)."""

    def __init__(self, run):
        self.run = Path(run)
        path = self.run / 'deck-scores.json'
        self.scores = defaultdict(list, json.loads(path.read_text()) if path.exists() else {})

    def add(self, results):
        for row in results:
            if 'deck' in row:
                self.scores[row['deck']] = (self.scores[row['deck']] + [row['deck_score']])[-RECENT:]

    def save(self):
        for name, value in (('deck-scores', self.scores), ('hard-decks', hard_decks(self.scores))):
            tmp = self.run / f'{name}.tmp'
            tmp.write_text(json.dumps(value))
            os.replace(tmp, self.run / f'{name}.json')


def read_hard(run):
    """RUN/hard-decks.json ({} before the learner wrote one, or caught mid-write)."""
    try:
        return json.loads((Path(run) / 'hard-decks.json').read_text())
    except (OSError, ValueError):
        return {}
