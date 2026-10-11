"""Both seats' feature rows for one decision, built once per state: what `Agent.observe` gives (byte for byte), without
re-encoding the previous state, the whole play history and a two-state replay window for every seat and decision.

    observer = Observer(vocab)            # one per game (it remembers the previous state and the plays so far)
    rows = observer.observe(state)        # [row of owner 0, row of owner 1]: {INPUT_KEYS: numpy array}

Every value is computed with the same numpy operations and dtypes as il.data.features on a [previous, state] window
(build_cache.encode), so the rows are identical; tools/bench_obs.py --check compares them on recorded games.
"""
import numpy as np

from . import data as D

_TYPE = {name: i for i, name in enumerate(('troop', 'building', 'projectile', 'area'))}  # build_cache.ENTITY_TYPES
_F32 = np.float32


def _int(value):
    return -1 if value is None else value


def _clip(a, low, high):
    """np.clip for integer arrays (the same values; np.clip's wrapper costs more than the work here)."""
    return np.minimum(np.maximum(a, low), high)


class Observer:
    def __init__(self, vocab, owners=(0, 1)):
        self.vocab, self.owners = vocab, tuple(owners)
        self.previous = None                   # {native id: (x, y)} of the previous state, first occurrence
        self.revealed = {o: {} for o in self.owners}   # opponent card -> index of its latest play (per seat)
        self.opp_plays = {o: 0 for o in self.owners}

    def observe(self, state):
        vocab = self.vocab
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

        keep = min(n, D.MAX_ENTITIES)
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
                'ent_obj': np.zeros(D.MAX_ENTITIES, np.int64), 'ent_side': np.zeros(D.MAX_ENTITIES, np.int64),
                'ent_type': np.zeros(D.MAX_ENTITIES, np.int64), 'ent_cell': np.zeros(D.MAX_ENTITIES, np.int64),
                'ent_float': np.zeros((D.MAX_ENTITIES, D.ENTITY_FLOATS), np.float32)}
            row['ent_obj'][:keep] = obj_ids
            row['ent_side'][:keep] = np.where(own[:keep] == owner, 1, 2)
            row['ent_type'][:keep] = typ[:keep] + 1
            row['ent_cell'][:keep] = (_clip(ry[:keep] // 1000, 0, D.H - 1) * D.W
                                      + _clip(rx[:keep] // 1000, 0, D.W - 1))
            row['ent_float'][:keep] = floats

            # Opponent cards seen so far (data.revealed_cards on the accumulated plays).
            last = self.revealed[owner]
            for play in state.plays:
                if play.owner != owner:
                    last[int(play.card)] = self.opp_plays[owner]
                    self.opp_plays[owner] += 1
            k = self.opp_plays[owner]
            rev_card = np.zeros(D.REVEALED, np.int64)
            rev_age = np.zeros(D.REVEALED, np.float32)
            for i, (c, m) in enumerate(sorted(last.items(), key=lambda item: -item[1])[:D.REVEALED]):
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
            row['slot_ok'] = (hand[:, 0] != -1) & (hand[:, 1] * 10000 <= elixir + D.AFFORD_SLACK)
            rows.append(row)
        return rows
