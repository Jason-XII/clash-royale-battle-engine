"""Cache reader, owner-relative features and action labels. numpy only: no engine, no torch.

A replay in the cache (built by build_cache.py) is a dict of raw arrays, one row per
decision state (every 5 ticks). `features(replay, owner, vocab)` turns it into the model
inputs of one player, rotated so that player always defends the bottom half.
"""
import json
from pathlib import Path

import numpy as np

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


def features(replay, owner, vocab):
    """Model inputs for one player over a whole replay (T decision states)."""
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
    out['slot_ok'] = (hand[..., 0] != -1) & (hand[..., 1] * 10000 <= replay['elixir'][:, owner, None] + AFFORD_SLACK)
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
