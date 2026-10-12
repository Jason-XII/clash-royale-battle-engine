"""How a player defends: enemy pushes into their half, whether they answered, what the pushes cost them.

    python -m il.defense data/cache --shards 8                      # human baseline (cache), all decks
    python -m il.defense data/cache --shards 40 --deck 26000043,26000020,26000036,26000052,26000017,26000083,28000015,28000026

A push is a stretch of time with enemy troops on the defender's side of the river (owner-relative y < RIVER),
ending once none has been there for GAP ticks; pushes of less than MIN_HP total troop hp (spirits, skeletons) are
ignored. It is answered when a play of the defender lands in its own half (or on the bridge), in the push's lane
(or the middle, CENTER of x = 9000), between ANSWER_EARLY ticks before the push and its end. Engine games (il.arena) and human games (the cache) are measured the same way.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RIVER = 15000
OWN_HALF = 17000  # plays landing below this (owner-relative) count as defensive: own half and the bridge
GAP = 40
MIN_HP = 1000
ANSWER_EARLY = 40  # a building or troop placed just before the push crosses still answers it
CENTER = 2500


def rel_y(y, owner):
    return y if owner == 0 else 32000 - y


def game(frames, plays):
    """frames: [(tick, rows)], rows an int array [n, 7]: owner, card (-1 tower), x, y, hp, max_hp, type (build_cache.ENTITY_TYPES: 0 troop, 1 building);
    plays: [(landing tick, owner, x, y)]. Returns per owner a dict of pushes (start, end, hp, damage, answered,
    latency), the tower hp lost, the starting tower hp and the share of plays landing in its own half."""
    out = []
    for o in (0, 1):
        own = [(t, x, y) for t, p, x, y in plays if p == o]
        landings = np.array([(t, x) for t, x, y in own if rel_y(y, o) < OWN_HALF], np.int64).reshape(-1, 2)
        pushes, current, last_seen, prev_hp, lost, start_hp = [], None, None, None, 0, None
        dmg_ticks = []
        for tick, rows in frames:
            towers = rows[(rows[:, 1] == -1) & (rows[:, 0] == o) & (rows[:, 6] == 1)]
            hp = int(np.maximum(towers[:, 4], 0).sum())
            if prev_hp is None:
                start_hp = hp
            elif hp < prev_hp:
                lost += prev_hp - hp
                dmg_ticks.append((tick, prev_hp - hp))
            prev_hp = hp
            enemy = rows[(rows[:, 0] == 1 - o) & (rows[:, 6] == 0) & (rows[:, 1] != -1)]
            threat = enemy[rel_y(enemy[:, 3], o) < RIVER]
            if len(threat):
                hps = np.maximum(threat[:, 5], threat[:, 4]).clip(0)
                if current is None or tick - last_seen > GAP:
                    if current is not None:
                        pushes.append(current)
                    current = dict(start=tick, end=tick, hp=0, left=0)
                current['end'], current['hp'] = tick, max(current['hp'], int(hps.sum()))
                current['left'] += int(hps[threat[:, 2] < 9000].sum()) - int(hps[threat[:, 2] >= 9000].sum())
                last_seen = tick
        if current is not None:
            pushes.append(current)
        dmg_ticks = np.array(dmg_ticks, np.int64).reshape(-1, 2)
        kept = []
        for p in pushes:
            if p['hp'] < MIN_HP:
                continue
            window = (dmg_ticks[:, 0] >= p['start']) & (dmg_ticks[:, 0] <= p['end'] + GAP)
            t, x = landings[:, 0], landings[:, 1]
            lane = np.abs(x - 9000) < CENTER
            lane |= (x < 9000) if p['left'] > 0 else (x >= 9000)
            answers = t[(t >= p['start'] - ANSWER_EARLY) & (t <= p['end']) & lane]
            kept.append(dict(p, damage=int(dmg_ticks[window, 1].sum()), answered=bool(len(answers)),
                             latency=int(max(answers.min() - p['start'], 0)) if len(answers) else None))
        out.append(dict(pushes=kept, lost=lost, start_hp=start_hp or 1,
                        own_half=len(landings) / max(len(own), 1), plays=len(own)))
    return out


def board_rows(state):
    """il.defense's frame rows: owner, card, x, y, hp, max hp, entity type."""
    from native_engine.build_cache import ENTITY_TYPES
    return np.array([(e.owner, e.card, e.x, e.y, -1 if e.hp is None else e.hp, -1 if e.max_hp is None else e.max_hp,
                      ENTITY_TYPES.index(e.entity_type)) for e in state.entities], np.int64).reshape(-1, 7)


def summary(games):
    """Per-player-game stats -> one line of aggregates: pushes per game, the share answered, damage per answered /
    unanswered push (as a share of the starting tower hp), tower hp lost per game, own-half play share."""
    pushes = [p | dict(start_hp=g['start_hp']) for g in games for p in g['pushes']]
    n = max(len(games), 1)
    answered = [p for p in pushes if p['answered']]
    unanswered = [p for p in pushes if not p['answered']]
    frac = lambda ps: float(np.mean([p['damage'] / p['start_hp'] for p in ps])) if ps else float('nan')
    big = [p for p in pushes if p['hp'] >= 3000]
    return dict(player_games=len(games), pushes_per_game=round(len(pushes) / n, 2),
                answered=round(len(answered) / max(len(pushes), 1), 3),
                answered_big=round(sum(p['answered'] for p in big) / max(len(big), 1), 3),
                latency_s=round(float(np.median([p['latency'] for p in answered])) / 20, 2) if answered else None,
                dmg_answered=round(frac(answered), 4), dmg_unanswered=round(frac(unanswered), 4),
                lost_per_game=round(float(np.mean([g['lost'] / g['start_hp'] for g in games])), 3),
                own_half_plays=round(float(np.mean([g['own_half'] for g in games])), 3),
                plays_per_game=round(float(np.mean([g['plays'] for g in games])), 1))


def cache_game(replay):
    """A cached human game -> game() input. Plays are the human commands the engine executed."""
    e = replay['entities']
    order = np.argsort(e[:, 0], kind='stable')
    e = e[order]
    bounds = np.searchsorted(e[:, 0], np.arange(len(replay['tick']) + 1))
    frames = [(int(t), e[bounds[i]:bounds[i + 1]][:, [2, 3, 4, 5, 6, 7, 8]]) for i, t in enumerate(replay['tick'])]
    a = replay['actions']
    plays = [(int(t), int(o), int(x), int(y)) for t, o, c, x, y, ok in a if ok and c != -1]
    return frames, plays


def live_game(path):
    """An il.live battle log -> (our seat, game() input). Our plays: each landed tap where it was tapped."""
    frames, plays, owner, tap = [], [], None, None
    for line in open(path):
        r = json.loads(line)
        if r['event'] == 'start':
            owner = r['owner']
        elif r['event'] == 'decision' and 'board' in r:
            frames.append((r['tick'], np.array(r['board'], np.int64).reshape(-1, 7)))
        elif r['event'] == 'tap' and r.get('card') is not None:
            tap = r['world']
        elif r['event'] == 'landed' and tap:
            plays.append((r['tick'], owner, *tap))
            tap = None
    return owner, (frames, plays)


def main():
    from .data import Shard
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', help='the cache directory, or with --live: il.live battle logs (runs/live/*.jsonl)', nargs='+')
    parser.add_argument('--live', action='store_true', help='measure our seat in live battles')
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--deck', help='only player-games with exactly these 8 cards (comma-separated ids)')
    args = parser.parse_args()
    if args.live:
        games = []
        for path in args.cache:
            owner, (frames, plays) = live_game(path)
            if frames:
                games.append(game(frames, plays)[owner])
        print(summary(games))
        return
    deck = set(map(int, args.deck.split(','))) if args.deck else None
    shards = sorted(Path(args.cache[0]).glob('part-*.npz'))
    shards = [shards[i] for i in np.linspace(0, len(shards) - 1, args.shards).astype(int)]
    games = []
    for path in shards:
        shard = Shard(path)
        for i in range(len(shard)):
            r = shard[i]
            if r['meta'][0, 3] != 1:  # rebuilt game did not end with the real winner: not trusted
                continue
            owners = [0, 1]
            if deck:
                owners = [o for o in (0, 1)
                          if set(r['hand'][0, o, :, 0].tolist()) | set(r['cycle'][0, o].tolist()) == deck]
                if not owners:
                    continue
            stats = game(*cache_game(r))
            games += [stats[o] for o in owners]
        print(path.name, summary(games), flush=True)


if __name__ == '__main__':
    main()
