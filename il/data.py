"""Cache reader, owner-relative features and action labels. numpy only: no engine, no torch.

A replay in the cache (built by build_cache.py) is a dict of raw arrays, one row per
decision state (every 5 ticks). `features(replay, owner, vocab)` turns it into the model
inputs of one player, rotated so that player always defends the bottom half.
"""
import json
from pathlib import Path

import numpy as np

from native_engine.engine import hold_slack, regen

W, H = 18, 32  # native tiles; 1 tile = 1000 world units
CELLS = W * H
WAIT, ABILITY = 0, 1
ACTIONS = 2 + 4 * CELLS  # wait, ability, (hand slot, cell)
MAX_ENTITIES = 64
ENTITY_FLOATS = 10
SCALARS = 6
REVEALED = 8
# A play lands ~22 ticks (1.1 s) after the tap (measured live: 21-25), so a human decided on the
# board ~22 ticks before the command's execution tick; labels use that board.
DEPLOY_DELAY = 22
# The game accepts a tap up to ~1 elixir short and lands the card once it is affordable; at the
# board 22 ticks before execution humans were short by <= 0.8 elixir in 99.5% of plays.
AFFORD_SLACK = 8000
# Inputs the IL cache lacks (zeros there; PPO learns them): our play in flight (4), the hero / champion ability (5),
# per hand slot "its next play is evolved" (4) and evolution progress (4), the opponent elixir estimate (1).
EXTRAS = 18
class OpponentElixir:
    """The opponent's elixir as a player tracks it: both start equal, it regenerates on the clock (capped at 10)
    and every card the opponent plays costs its static cost (unknown / variable: 4)."""

    def __init__(self):
        self.value, self.tick = None, None

    def update(self, tick, own_elixir, cards):
        from native_engine.clapha import COSTS
        self.value = float(own_elixir) if self.value is None else min(100000.0, self.value + regen(self.tick, tick))
        self.value = max(0.0, self.value - sum(COSTS.get(int(c), 4) * 10000 for c in cards))
        self.tick = tick
        return self.value


class Shard:
    """One cache shard; `shard[i]` is replay i as a dict of arrays."""

    def __init__(self, path):
        with np.load(path) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.tags = self.arrays.pop('tags')
        self.keys = [k for k in self.arrays if not k.endswith('_off')]

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, i):
        a = self.arrays
        return {k: a[k][a[k + '_off'][i]:a[k + '_off'][i + 1]] for k in self.keys}


def card_ids(replay):
    """Every native id the vocab must cover: object types, hand/cycle cards, played cards."""
    return np.concatenate([replay['entities'][:, 3], replay['hand'][..., 0].ravel(),
                           replay['cycle'].ravel(), replay['plays'][:, 3]])


def build_vocab(shard_paths):
    """id -> index; 0 = padding, 1 = unknown id (unseen at vocab time). Towers are object -1."""
    ids = set()
    for path in shard_paths:
        with np.load(path) as z:
            for key, column in (('entities', 3), ('plays', 3)):
                ids.update(np.unique(z[key][:, column]).tolist())
            ids.update(np.unique(z['hand'][..., 0]).tolist())
            ids.update(np.unique(z['cycle']).tolist())
    return {int(i): n + 2 for n, i in enumerate(sorted(ids))}


def load_vocab(path):
    return {int(k): v for k, v in json.loads(Path(path).read_text()).items()}


def lookup(vocab, ids, empty=None):
    """Native ids -> vocab indices; `empty` (e.g. -1 for a missing hand card) -> 0."""
    ids = np.asarray(ids)
    out = np.fromiter((vocab.get(int(i), 1) for i in ids.ravel()), np.int64, ids.size).reshape(ids.shape)
    return out if empty is None else np.where(ids == empty, 0, out)


def rotate(x, y, owner):
    """World position -> owner-relative world position (owner 1 is rotated 180 degrees)."""
    return (18000 - x, 32000 - y) if owner == 1 else (x, y)


def cell_of(x, y):
    return np.clip(np.asarray(y) // 1000, 0, H - 1) * W + np.clip(np.asarray(x) // 1000, 0, W - 1)


def cell_center(cell, owner):
    """Owner-relative cell index -> native world (x, y) at the cell center."""
    x, y = (cell % W) * 1000 + 500, (cell // W) * 1000 + 500
    return rotate(x, y, owner)


def entity_features(replay, owner, vocab, T):
    e = replay['entities'].astype(np.int64)
    st, eid, own, obj, x, y, hp, mhp, typ, dep, cd, tgt = e.T
    x, y = rotate(x, y, owner)
    # Displacement since the previous decision, matched by native entity id.
    key = st << 32 | eid
    order = np.argsort(key, kind='stable')
    prev = (st - 1) << 32 | eid
    pos = np.minimum(np.searchsorted(key[order], prev), max(len(key) - 1, 0))
    found = (key[order][pos] == prev) if len(key) else np.zeros(0, bool)
    j = order[pos] if len(key) else pos
    dx = np.where(found, x - x[j], 0)
    dy = np.where(found, y - y[j], 0)
    # Rows are grouped by state in capture order; keep the first MAX_ENTITIES of each state.
    slot = np.arange(len(st)) - np.searchsorted(st, st)
    keep = slot < MAX_ENTITIES
    st, slot = st[keep], slot[keep]
    floats = np.stack([
        x / 18000, y / 32000,
        np.where(mhp > 0, hp / np.maximum(mhp, 1), 0), np.log1p(np.maximum(mhp, 0)) / 16,
        dep > 0, np.clip(dep, 0, 5000) / 1000, np.clip(cd, 0, 5000) / 1000, tgt >= 0,
        np.clip(dx, -3000, 3000) / 1000, np.clip(dy, -3000, 3000) / 1000,
    ], 1)[keep].astype(np.float32)
    out = {
        'ent_obj': np.zeros((T, MAX_ENTITIES), np.int64),
        'ent_side': np.zeros((T, MAX_ENTITIES), np.int64),  # 0 pad, 1 own, 2 enemy
        'ent_type': np.zeros((T, MAX_ENTITIES), np.int64),  # 0 pad, 1 + build_cache.ENTITY_TYPES index
        'ent_cell': np.zeros((T, MAX_ENTITIES), np.int64),
        'ent_float': np.zeros((T, MAX_ENTITIES, ENTITY_FLOATS), np.float32),
    }
    out['ent_obj'][st, slot] = lookup(vocab, obj[keep])
    out['ent_side'][st, slot] = np.where(own[keep] == owner, 1, 2)
    out['ent_type'][st, slot] = typ[keep] + 1
    out['ent_cell'][st, slot] = cell_of(x[keep], y[keep])
    out['ent_float'][st, slot] = floats
    return out


def revealed_cards(replay, owner, vocab, T):
    """Opponent cards seen so far, most recent first, and how many opponent plays ago."""
    card = np.zeros((T, REVEALED), np.int64)
    age = np.zeros((T, REVEALED), np.float32)
    last = {}  # card id -> index of its latest opponent play
    plays = [p for p in replay['plays'] if p[2] != owner]
    k = 0
    for t in range(T):
        while k < len(plays) and plays[k][0] <= t:
            last[int(plays[k][3])] = k
            k += 1
        recent = sorted(last.items(), key=lambda item: -item[1])[:REVEALED]
        for i, (c, n) in enumerate(recent):
            card[t, i] = vocab.get(c, 1)
            age[t, i] = (k - 1 - n) / 8
    return {'rev_card': card, 'rev_age': age}


def labels(replay, owner):
    """Human action per decision: WAIT, ABILITY or 2 + slot * CELLS + cell, plus a loss mask.

    A command executing at tick c belongs to the last state with tick < c - DEPLOY_DELAY: the board
    the human saw when tapping.
    """
    tick, hand, elixir = replay['tick'], replay['hand'][:, owner], replay['elixir'][:, owner]
    T = len(tick)
    action = np.zeros(T, np.int64)
    mask = np.ones(T, bool)
    for c_tick, o, card, x, y, executed in replay['actions']:
        if o != owner:
            continue
        t = np.searchsorted(tick, c_tick - DEPLOY_DELAY) - 1
        if t < 0 or t >= T or action[t] != WAIT:
            continue  # ponytail: a 2nd action in one 250 ms window is dropped (~0.1% of windows)
        if not executed:
            mask[t] = False  # native rejected it; the state would not reflect the label
            continue
        if card == -1:
            action[t] = ABILITY
            continue
        slots = np.flatnonzero(hand[t, :, 0] == card)
        if len(slots) == 0 or hand[t, slots[0], 1] * 10000 > elixir[t] + AFFORD_SLACK:
            mask[t] = False
            continue
        action[t] = 2 + slots[0] * CELLS + cell_of(*rotate(x, y, owner))
    return action, mask


def features(replay, owner, vocab, online=False):
    """Model inputs for one player over a whole replay (T decision states). online: mask cards by the live client's hold
    rule (engine.hold_slack, what Observer does), not by the IL labels' fixed AFFORD_SLACK."""
    T = len(replay['tick'])
    tick = replay['tick'].astype(np.float32)
    hand = replay['hand'][:, owner]
    out = entity_features(replay, owner, vocab, T)
    out.update(revealed_cards(replay, owner, vocab, T))
    out['scalars'] = np.stack([
        replay['elixir'][:, owner] / 100000, (tick - 90) / 3600,
        replay['crowns'][:, owner] / 3, replay['crowns'][:, 1 - owner] / 3,
        tick >= 2490, tick >= 3690,  # double elixir, overtime (approximate boundaries)
    ], 1).astype(np.float32)
    out['hand_card'] = lookup(vocab, hand[..., 0], empty=-1)
    out['hand_cost'] = (np.maximum(hand[..., 1], 0) / 10).astype(np.float32)
    out['next_card'] = lookup(vocab, replay['cycle'][:, owner, 0], empty=-1)
    slack = np.array([hold_slack(int(k)) for k in replay['tick']])[:, None] if online else AFFORD_SLACK
    out['slot_ok'] = (hand[..., 0] != -1) & (hand[..., 1] * 10000 <= replay['elixir'][:, owner, None] + slack)
    out['cycle_card'] = lookup(vocab, replay['cycle'][:, owner, 1:4], empty=-1)
    out['pending_card'] = np.zeros(T, np.int64)  # no play in flight in the cache
    out['ability_ok'] = np.ones(T, bool)  # unknown in the cache: unmasked (IL labels include ability presses)
    other = 1 - owner
    out['opp_deck'] = lookup(vocab, np.concatenate([replay['hand'][:, other, :, 0], replay['cycle'][:, other]], 1),
                             empty=-1)
    extras = np.zeros((T, EXTRAS), np.float32)
    opponent, plays = OpponentElixir(), [p for p in replay['plays'] if p[2] != owner]
    k = 0
    for i in range(T):
        cards = []
        while k < len(plays) and plays[k][0] <= i:
            cards.append(plays[k][3])
            k += 1
        extras[i, -1] = opponent.update(int(replay['tick'][i]), replay['elixir'][0, owner], cards) / 100000
    out['extras'] = extras
    return out


def outcome(replay, owner):
    """+1 if `owner` won the rebuilt game, -1 if lost, 0 draw/unknown."""
    result = replay['meta'][0, 0]
    return 0.0 if result not in (0, 1) else (1.0 if result == owner else -1.0)


def faithful(replay):
    """The rebuilt game ended like the real one (winner and crowns)."""
    return bool(replay['meta'][0, 3] and replay['meta'][0, 4])


def sequence(replay, owner, vocab):
    """Everything one training sequence needs."""
    out = features(replay, owner, vocab)
    out['action'], out['label_mask'] = labels(replay, owner)
    out['value'] = np.full(len(replay['tick']), outcome(replay, owner), np.float32)
    return out


# ---- online: features from engine states (collectors, arena, live) -------------------------------------------------
_TYPE = {name: i for i, name in enumerate(('troop', 'building', 'projectile', 'area'))}  # build_cache.ENTITY_TYPES
_F32 = np.float32


def _int(value):
    return -1 if value is None else value


def _clip(a, low, high):
    """np.clip for integer arrays (the same values; np.clip's wrapper costs more than the work here)."""
    return np.minimum(np.maximum(a, low), high)


class InFlight:
    """A tap whose play (card) or ability press (card None) has not landed yet."""
    __slots__ = ('owner', 'card', 'cost', 'due')

    def __init__(self, owner, card, cost, due):
        self.owner, self.card, self.cost, self.due = owner, card, cost, due


class Observer:
    """Both seats' feature rows for one decision, built once per state: what `features` gives on a [previous, state]
    window with every play so far (build_cache.encode), byte for byte (test_il.FastObserve), ~3.7x faster.

        observer = Observer(vocab)            # one per game (it remembers the previous state and the plays so far)
        rows = observer.observe(state)        # [row of owner 0, row of owner 1]: {INPUT_KEYS: numpy array}

    Keep it in step with `features` when either changes.
    """

    def __init__(self, vocab, owners=(0, 1)):
        self.vocab, self.owners = vocab, tuple(owners)
        self.opponent = {o: OpponentElixir() for o in self.owners}
        self.first_elixir = None
        self.previous = None                   # {native id: (x, y)} of the previous state, first occurrence
        self.revealed = {o: {} for o in self.owners}   # opponent card -> index of its latest play (per seat)
        self.opp_plays = {o: 0 for o in self.owners}

    def observe(self, state, in_flight=()):
        """`in_flight`: InFlight plays / ability presses tapped but not landed yet (either player)."""
        vocab = self.vocab
        if self.first_elixir is None:
            self.first_elixir = [pl.elixir for pl in state.players]
        ents = state.entities
        n = len(ents)
        # The columns build_cache.encode keeps, through int32 as it stores them.
        e = np.array([(x.id, x.owner, x.card, x.x, x.y, _int(x.hp), _int(x.max_hp), _TYPE[x.entity_type],
                       _int(x.deployment_remaining_ms), _int(x.attack_cooldown_ms), _int(x.target_id)) for x in ents],
                     np.int32).reshape(-1, 11).astype(np.int64)
        eid, own, obj, x, y, hp, mhp, typ, dep, cd, tgt = e.T
        prev = self.previous
        if prev is None:  # the first state: Agent.observe's window is [state, state]
            first = {}
            for i, k in enumerate(eid.tolist()):
                first.setdefault(k, i)
            px, py, found = x[[first[k] for k in eid.tolist()]], y[[first[k] for k in eid.tolist()]], np.ones(n, bool)
        else:
            hits = [prev.get(k) for k in eid.tolist()]
            found = np.fromiter((h is not None for h in hits), bool, n)
            px = np.fromiter((h[0] if h is not None else 0 for h in hits), np.int64, n)
            py = np.fromiter((h[1] if h is not None else 0 for h in hits), np.int64, n)
        nxt = {}
        for k, a, b in zip(eid.tolist(), x.tolist(), y.tolist()):
            nxt.setdefault(k, (a, b))
        self.previous = nxt

        keep = min(n, MAX_ENTITIES)
        obj_ids = np.fromiter((vocab.get(int(i), 1) for i in obj[:keep].tolist()), np.int64, keep)
        # Owner-independent floats (as data.entity_features computes them, float64 then float32).
        hp_f = np.where(mhp > 0, hp / np.maximum(mhp, 1), 0)[:keep]
        mhp_f = (np.log1p(np.maximum(mhp, 0)) / 16)[:keep]
        dep_on = (dep > 0)[:keep]
        dep_f = (_clip(dep, 0, 5000) / 1000)[:keep]
        cd_f = (_clip(cd, 0, 5000) / 1000)[:keep]
        tgt_on = (tgt >= 0)[:keep]

        p = state.players
        tick = _F32(state.tick)
        rows = []
        for owner in self.owners:
            if owner == 1:
                rx, ry, qx, qy = 18000 - x, 32000 - y, 18000 - px, 32000 - py
            else:
                rx, ry, qx, qy = x, y, px, py
            dx = np.where(found, rx - qx, 0)
            dy = np.where(found, ry - qy, 0)
            floats = np.stack([rx[:keep] / 18000, ry[:keep] / 32000, hp_f, mhp_f, dep_on, dep_f, cd_f, tgt_on,
                               _clip(dx[:keep], -3000, 3000) / 1000, _clip(dy[:keep], -3000, 3000) / 1000],
                              1).astype(np.float32)
            row = {
                'ent_obj': np.zeros(MAX_ENTITIES, np.int64), 'ent_side': np.zeros(MAX_ENTITIES, np.int64),
                'ent_type': np.zeros(MAX_ENTITIES, np.int64), 'ent_cell': np.zeros(MAX_ENTITIES, np.int64),
                'ent_float': np.zeros((MAX_ENTITIES, ENTITY_FLOATS), np.float32)}
            row['ent_obj'][:keep] = obj_ids
            row['ent_side'][:keep] = np.where(own[:keep] == owner, 1, 2)
            row['ent_type'][:keep] = typ[:keep] + 1
            row['ent_cell'][:keep] = (_clip(ry[:keep] // 1000, 0, H - 1) * W
                                      + _clip(rx[:keep] // 1000, 0, W - 1))
            row['ent_float'][:keep] = floats

            # Opponent cards seen so far (data.revealed_cards on the accumulated plays).
            last = self.revealed[owner]
            for play in state.plays:
                if play.owner != owner:
                    last[int(play.card)] = self.opp_plays[owner]
                    self.opp_plays[owner] += 1
            k = self.opp_plays[owner]
            rev_card = np.zeros(REVEALED, np.int64)
            rev_age = np.zeros(REVEALED, np.float32)
            for i, (c, m) in enumerate(sorted(last.items(), key=lambda item: -item[1])[:REVEALED]):
                rev_card[i] = vocab.get(c, 1)
                rev_age[i] = (k - 1 - m) / 8
            row['rev_card'], row['rev_age'] = rev_card, rev_age

            me = p[owner]
            elixir = np.int32(me.elixir)
            row['scalars'] = np.array([elixir / 100000, (tick - 90) / 3600, np.int32(state.crowns[owner]) / 3,
                                       np.int32(state.crowns[1 - owner]) / 3, tick >= 2490, tick >= 3690],
                                      np.float64).astype(np.float32)
            hand = np.full((4, 2), -1, np.int32)
            for c in me.hand:
                hand[c.slot] = (c.card, c.cost)
            cards = np.fromiter((0 if c == -1 else vocab.get(c, 1) for c in hand[:, 0].tolist()), np.int64, 4)
            row['hand_card'] = cards
            row['hand_cost'] = (np.maximum(hand[:, 1], 0) / 10).astype(np.float32)
            cycle = np.full(4, -1, np.int32)
            cycle[:len(me.cycle)] = me.cycle[:4]
            nc = int(cycle[0])
            row['next_card'] = np.array(0 if nc == -1 else vocab.get(nc, 1), np.int64)
            row['slot_ok'] = (hand[:, 0] != -1) & (hand[:, 1] * 10000 <= elixir + hold_slack(state.tick))
            cycle_ids = cycle[1:4].tolist()
            row['cycle_card'] = np.fromiter((0 if c == -1 else vocab.get(c, 1) for c in cycle_ids), np.int64, 3)
            them = p[1 - owner]
            deck = np.full(8, -1, np.int32)
            for c in them.hand:
                deck[c.slot] = c.card
            deck[4:4 + min(4, len(them.cycle))] = them.cycle[:4]
            row['opp_deck'] = np.fromiter((0 if c == -1 else vocab.get(c, 1) for c in deck.tolist()), np.int64, 8)

            extras = np.zeros(EXTRAS, np.float32)
            mine = [f for f in in_flight if f.owner == owner]
            if mine:  # one action in flight per player (ClaphaEngine, il.live): nothing else can be played
                f = mine[0]
                extras[0:4] = (1, f.cost / 10, max(0, f.due - state.tick) / DEPLOY_DELAY,
                               (int(elixir) - f.cost * 10000) / 100000)
                row['slot_ok'] = np.zeros(4, bool)
                row['pending_card'] = np.array(0 if f.card is None else vocab.get(f.card, 1), np.int64)
            else:
                row['pending_card'] = np.array(0, np.int64)
            carriers = [a for a in me.abilities if a.units]
            ready = any(a.ready for a in me.abilities)
            if carriers:
                a = carriers[0]
                extras[4:9] = (1, ready, a.button == 10, 1 if a.charges < 0 else min(a.charges, 3) / 3,
                               a.cooldown_ms / a.cooldown_total_ms if a.cooldown_total_ms > 0 else 0)
            row['ability_ok'] = np.array(ready and not mine)
            evolution = {e.card: e for e in me.evolution if e.evolvable}
            for i, c in enumerate(hand[:, 0].tolist()):
                e = evolution.get(c)
                if e is not None:
                    extras[9 + i] = e.ready
                    extras[13 + i] = 1 - e.remaining / e.required if e.required > 0 else 0
            cards = [pl.card for pl in state.plays if pl.owner != owner]
            extras[-1] = self.opponent[owner].update(state.tick, self.first_elixir[owner], cards) / 100000
            row['extras'] = extras
            rows.append(row)
        return rows
