"""Imitation learning's loss and data: converted games -> chunks and padded batches -> the loss. train.py trains on
them; the PPO learner's imitation step (--bc-index) uses them too."""
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cr import observation as C

W, H, CELLS = C.W, C.H, C.CELLS
WAIT, PLAY, ABILITY1 = C.WAIT, C.PLAY, C.ABILITY1


# ---- the loss -----------------------------------------------------------------------------------------------------------
def loss_and_metrics(model, xs, ts, value_coef, next_coef):
    out = model(xs)
    live = ts['time'] & ts['ok']
    what = out['what'][live]
    target_what = ts['what'][live]
    finite = torch.isfinite(what.gather(1, target_what.unsqueeze(1))).squeeze(1)   # labels the masks allow
    zero = out['value'].new_zeros(())          # an empty loss (scores hold -inf for masked choices: x.sum() * 0 is NaN)
    loss_what = F.cross_entropy(what[finite], target_what[finite]) if finite.any() else zero
    play = live.reshape(-1) & (ts['what'].reshape(-1) == PLAY)
    which = out['which'].reshape(-1, 4)[play]
    target_which = ts['which'].reshape(-1)[play]
    which_ok = ts['which_ok'].reshape(-1)[play]
    loss_which = F.cross_entropy(which[which_ok], target_which[which_ok]) if which_ok.any() else zero
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in xs.items() if k in ('hand_card', 'legal')}
    if play.any():
        tiles = model.tiles({'core': out['core'][play], 'grid': out['grid'][play]},
                            {k: v[play] for k, v in flat.items()}, target_which)
        target_tile = ts['tile'].reshape(-1)[play]
        loss_tile = F.cross_entropy(tiles, target_tile)
    else:
        loss_tile = zero
    loss_value = F.mse_loss(out['value'][ts['time']], ts['value'][ts['time']])
    loss = loss_what + loss_which + loss_tile + value_coef * loss_value
    m = {}
    if next_coef:
        when_ok = ts['time']
        which_ok_next = when_ok & ts['next_which_ok']
        loss_when = F.cross_entropy(out['next_when'][when_ok], ts['next_when'][when_ok])
        loss_which_next = F.cross_entropy(out['next_which'][which_ok_next], ts['next_which'][which_ok_next]) \
            if which_ok_next.any() else zero
        loss = loss + next_coef * (loss_when + loss_which_next)
        m.update(next_when=loss_when.item(), next_which=loss_which_next.item())
    with torch.no_grad():
        is_play = target_what == PLAY
        p_play = what.float().softmax(-1)[:, PLAY]
        value_live, target_value = out['value'][ts['time']], ts['value'][ts['time']]
        decided = target_value != 0
        m.update(loss=loss.item(), what=loss_what.item(), which=loss_which.item(), tile=loss_tile.item(),
                 value=loss_value.item(), n=int(live.sum()), plays=int(is_play.sum()), p_play_sum=float(p_play.sum()),
                 value_ok=int(((value_live > 0) == (target_value > 0))[decided].sum()), value_n=int(decided.sum()))
        if play.any():
            top1 = tiles.argmax(-1)
            top5 = (tiles.topk(5, -1).indices == target_tile.unsqueeze(1)).any(1)
            dist_ = torch.sqrt(((top1 // W - target_tile // W) ** 2 + (top1 % W - target_tile % W) ** 2).float())
            m.update(card_ok=int((which.argmax(-1) == target_which)[which_ok].sum()), card_n=int(which_ok.sum()),
                     tile_top1=int((top1 == target_tile).sum()), tile_top5=int(top5.sum()), tile_dist_sum=float(dist_.sum()))
    return loss, m


def summarize(ms):
    s = {k: sum(m.get(k, 0) for m in ms) for k in ('n', 'plays', 'card_ok', 'tile_top1', 'tile_top5', 'p_play_sum',
                                                   'value_ok', 'value_n', 'tile_dist_sum', 'card_n')}
    out = {k: float(np.mean([m[k] for m in ms])) for k in ('loss', 'what', 'which', 'tile', 'value', 'next_when', 'next_which')
           if k in ms[0]}
    rate = s['plays'] / max(1, s['n'])
    guess = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate)) if 0 < rate < 1 else 0.0
    plays = max(1, s['plays'])
    out.update(timing_gain=1 - out['what'] / guess if guess else 0.0, card_acc=s['card_ok'] / max(1, s['card_n']),
               tile_top1=s['tile_top1'] / plays, tile_top5=s['tile_top5'] / plays, tile_dist=s['tile_dist_sum'] / plays,
               decisions=s['n'], p_play_mean=s['p_play_sum'] / max(1, s['n']), play_rate=rate,
               value_sign_acc=s['value_ok'] / max(1, s['value_n']))
    return out


class Objective(nn.Module):
    """The whole loss as one module, so DDP sees the tile head (called outside Policy.forward) too."""

    def __init__(self, policy, value_coef, next_coef):
        super().__init__()
        self.policy, self.value_coef, self.next_coef = policy, value_coef, next_coef

    def forward(self, xs, ts):
        return loss_and_metrics(self.policy, xs, ts, self.value_coef, self.next_coef)


# ---- the data -----------------------------------------------------------------------------------------------------------
def read(path, owner):
    """One player's arrays from a converted game (both players' views share one archive)."""
    with np.load(path, allow_pickle=False) as archive:
        prefix = f'p{owner}_'
        return {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}


def chunks(data, chunk, validation, rng):
    """One player's game -> chunks of `chunk` decisions (training: from a random offset), objects unpacked."""
    offsets = data['offsets']
    T = len(offsets) - 1
    for s in range(0 if validation or T <= chunk else rng.randrange(chunk), T, chunk):
        e = min(s + chunk, T)
        counts = np.diff(offsets[s:e + 1])
        n = max(1, int(counts.max()))
        x, t = {}, {}
        for name, v in data.items():
            k = name[2:]
            if name.startswith('t_'):
                t[k] = v[s:e]
            elif name.startswith('x_'):
                if k in C.OBJECT_KEYS:
                    a = np.zeros((e - s, n, *v.shape[1:]), v.dtype)
                    for i, count in enumerate(counts):
                        a[i, :count] = v[offsets[s + i]:offsets[s + i + 1]]
                elif k == 'legal':
                    a = np.unpackbits(v[s:e], axis=-1, count=4 * CELLS).reshape(e - s, 4, H, W).astype(bool)
                else:
                    a = v[s:e]
                x[k] = a.astype(np.int64) if a.dtype == np.int32 else a
        yield x, t


ABILITY_SHIFT = 4  # decisions (20 ticks)


def ability_labels(data):
    """--ability_labels 1 (22 Oct): each human ability press moves to the decision ~22 ticks before it executes, where
    card plays are labelled too (convert.py put presses at their execution tick: ~1.1 s late, while the arena and live
    delay a press like a card), and to the controller that is ready there when only the other one is (convert.py
    labelled every press controller 1: 42% of the presses in two-hero decks were dropped as not ready). A press whose
    button lit up only after that decision (7.8%: pressed the moment it was ready) goes to the first decision it is ready."""
    what, ready = data['t_what'].copy(), data['x_ability'][:, :, 1] > 0
    for i in np.flatnonzero(what >= ABILITY1):
        k = what[i] - ABILITY1
        what[i] = WAIT
        j = next((j for j in range(max(0, i - ABILITY_SHIFT), i + 1) if what[j] == WAIT and ready[j].any()), None)
        if j is not None:  # else no ready decision without a card play since the tap: dropped, as the loss would
            what[j] = ABILITY1 + (k if ready[j, k] else 1 - k)
    data['t_what'] = what
    return data


class Games(torch.utils.data.IterableDataset):
    """The player views of a dataset in a seeded random order (validation: in order), split over ranks and workers."""

    def __init__(self, root, entries, chunk, seed, validation=False, rank=0, world=1, fix_abilities=False):
        self.root, self.entries, self.chunk, self.seed = Path(root), entries, chunk, seed
        self.validation, self.rank, self.world, self.fix_abilities = validation, rank, world, fix_abilities

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed)
        entries = list(self.entries)
        if not self.validation:
            rng.shuffle(entries)
        rng = random.Random(self.seed * 1000 + self.rank * nw + wid)
        for e in entries[self.rank * nw + wid::self.world * nw]:
            data = read(self.root / e['file'], e['owner'])
            yield from chunks(ability_labels(data) if self.fix_abilities else data, self.chunk, self.validation, rng)


def collate(items):
    """Chunks -> a padded batch. The object axis is cut to the most objects any decision of the batch has (~12 on average
    of the 128 slots)."""
    L = max(len(t['what']) for _, t in items)
    O = max(1, max(int((x['obj_side'] > 0).sum(-1).max()) for x, _ in items))

    def pad(a, k):
        if k in C.OBJECT_KEYS:
            a = a[:, :O]
        out = np.zeros((L, O, *a.shape[2:]) if k in C.OBJECT_KEYS else (L, *a.shape[1:]), a.dtype)
        out[(slice(0, len(a)), *map(slice, a.shape[1:]))] = a
        return out
    xs = {k: torch.from_numpy(np.stack([pad(x[k], k) for x, _ in items])) for k in items[0][0]}
    ts = {k: torch.from_numpy(np.stack([pad(t[k], k) for _, t in items])) for k in items[0][1]}
    ts['time'] = torch.from_numpy(np.stack([np.arange(L) < len(t['what']) for _, t in items]))
    return xs, ts
