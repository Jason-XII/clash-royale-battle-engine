"""Imitation learning on a converted dataset: the policy network, its loss, the data loader and the training loop, plus
the inference helpers play.py, arena.py and ppo.py use.

    python train.py --data DATASET --out RUN                                  # one GPU (or CPU: --device cpu)
    torchrun --standalone --nproc_per_node=3 train.py --data DATASET --out RUN   # three GPUs, --batch is global

Rerunning the same command resumes from RUN/latest.pt with the settings saved there.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

import common as C

W, H, CELLS = C.W, C.H, C.CELLS
WAIT, PLAY, ABILITY1, ABILITY2 = C.WAIT, C.PLAY, C.ABILITY1, C.ABILITY2


# ---- the network --------------------------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int
    card_dim: int = 32
    object_dim: int = 32
    attention_layers: int = 1
    attention_heads: int = 4
    conv_channels: int = 48
    conv_layers: int = 4
    core: int = 256
    board: int = 0  # channels of a coarse 8 x 4 board map fed to the core (0 = off: the core sees only the mean and max)
    effect_fields: int = C.EFFECT_FIELDS
    next_play: bool = True
    # architecture options (21 Oct ablations; the defaults are the network every checkpoint so far has)
    res_blocks: int = 0    # 1: the board tower is a stem conv + conv_layers // 2 pre-activation residual blocks (GroupNorm)
    flight_grid: int = 0   # 1: plays in flight (ours and the opponent's) are drawn on the board as objects too
    tile_blocks: int = 0   # residual blocks between the card-conditioned FiLM and the tile scorer
    board_rows: int = 8    # the coarse board map's size (with board > 0)
    board_cols: int = 4
    res_norm: int = 1      # 0: residual blocks without GroupNorm, each branch scaled by a learned gate from 0 (SkipInit)
    dist_bias: int = 0     # 1: object attention biased by the distance between the two objects (per head, per tile)
    card_stats: int = 0    # > 0 (22 Oct): every card and unit id also gets its fixed stats from the game data (this
                           # many fields, assets/card_stats.json: range, targets, hp, damage, speed, splash, ...)


class ResBlock(nn.Module):
    """x + conv(relu(norm(conv(relu(norm(x)))))): a pre-activation residual block. norm=False (21 Oct): no GroupNorm
    (with its fp32 casts under autocast it took ~38% of a serving step) and the branch scaled by a gate that starts at 0."""

    def __init__(self, ch, norm=True):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ch) if norm else nn.Identity(), nn.Conv2d(ch, ch, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, ch) if norm else nn.Identity(), nn.Conv2d(ch, ch, 3, padding=1)
        self.gate = None if norm else nn.Parameter(torch.zeros(()))

    def forward(self, x):
        y = self.c2(F.relu(self.n2(self.c1(F.relu(self.n1(x))))))
        return x + (y if self.gate is None else self.gate * y)


class CardEmbedding(nn.Embedding):
    """The card embedding plus a learned map of each id's fixed stats (`table`, from the game data). The map starts at
    zero, so the embedding starts as a plain one; no bias, so padding (id 0, zero stats) stays zero."""

    def __init__(self, n, d, table):
        super().__init__(n, d, padding_idx=0)
        self.register_buffer('table', torch.as_tensor(table).float())
        self.stat = nn.Linear(self.table.shape[1], d, bias=False)
        nn.init.zeros_(self.stat.weight)

    def forward(self, ids):
        return super().forward(ids) + self.stat(self.table[ids])


def mlp(*widths):
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class Policy(nn.Module):
    def __init__(self, config: Config, effect_table=None, card_table=None):
        super().__init__()
        c = self.config = config
        d = c.card_dim
        if c.card_stats:  # without a table (train.load): zeros, filled by the checkpoint's buffer
            table = torch.zeros(c.vocab_size, c.card_stats) if card_table is None else torch.as_tensor(card_table)
            self.card = CardEmbedding(c.vocab_size, d, table)
        else:
            self.card = nn.Embedding(c.vocab_size, d, padding_idx=0)
        self.side = nn.Embedding(3, d, padding_idx=0)
        self.kind = nn.Embedding(6, d, padding_idx=0)
        # the effects' property vectors: fixed (from the game data), indexed like the vocabulary
        table = torch.zeros(c.vocab_size, c.effect_fields) if effect_table is None else torch.as_tensor(effect_table)
        self.register_buffer('effect_table', table.float())
        self.effect = mlp(d + 1 + c.effect_fields, d, d)
        self.object = mlp(4 * d + C.OBJ_FLOATS, c.object_dim, c.object_dim)
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(c.object_dim, c.attention_heads, 2 * c.object_dim, dropout=0.0, batch_first=True,
                                       norm_first=True), c.attention_layers) if c.attention_layers else None
        ch = c.conv_channels
        if c.res_blocks:  # convs keeps the stem (channels-last layout code expects convs[0])
            self.convs = nn.ModuleList([nn.Conv2d(c.object_dim + 2, ch, 3, padding=1)])
            self.blocks = nn.ModuleList([ResBlock(ch, c.res_norm) for _ in range(c.conv_layers // 2)])
            self.tower_out = nn.GroupNorm(8, ch) if c.res_norm else nn.Identity()
        else:
            self.convs = nn.ModuleList([nn.Conv2d(c.object_dim + 2, ch, 3, padding=1)] +
                                       [nn.Conv2d(ch, ch, 3, padding=1) for _ in range(c.conv_layers - 1)])
        self.flight = mlp(2 * d + 4, c.object_dim, c.object_dim) if c.flight_grid else None
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        self.register_buffer('coords', torch.stack([xs, ys])[None], persistent=False)
        hand_in = 4 * (2 * d + C.HAND_FLOATS)
        rest_in = (4 * d) + 8 + (d + 4) + 2 * (d + 4) + 8 * (d + 3) + 2 * d + C.SCALARS
        summary_in = 2 * ch + 2 * c.object_dim + hand_in + rest_in + c.board * c.board_rows * c.board_cols
        self.board = nn.Sequential(nn.Conv2d(ch, c.board, 1), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((c.board_rows, c.board_cols))) if c.board else None
        self.summary_norm = nn.LayerNorm(summary_in)
        self.summary = mlp(summary_in, c.core, c.core)
        self.core_norm = nn.LayerNorm(c.core)
        self.lstm = nn.LSTM(c.core, c.core, batch_first=True)
        self.what = nn.Linear(c.core, 4)                       # wait / play / ability 1 / ability 2
        self.which = mlp(c.core + 2 * d + C.HAND_FLOATS, 256, 1)   # score of each hand card
        self.film = nn.Linear(c.core + d, 2 * ch)              # the picture conditioned on the chosen card
        self.tile = nn.Conv2d(ch, 1, 1)
        self.tile_blocks = nn.ModuleList([ResBlock(ch, c.res_norm) for _ in range(c.tile_blocks)])
        # ponytail: distance only (FirstLight also biases by relation type and extent); add those if this helps
        self.dist_bias = nn.Embedding(16, c.attention_heads) if c.dist_bias and c.attention_layers else None
        if self.dist_bias is not None:
            nn.init.zeros_(self.dist_bias.weight)  # starts as plain attention
        self.value = mlp(c.core + C.SCALARS, 256, 1)
        if c.next_play:
            self.next_when = nn.Linear(c.core, len(C.NEXT_EDGES) + 1)
            self.next_which = mlp(c.core + 2 * d + C.HAND_FLOATS, 256, 1)

    def _objects(self, x, N):
        c = self.config
        card, data = self.card(x['obj_card']), self.card(x['obj_data'])
        eff = torch.cat([self.card(x['eff_id']), x['eff_left'].unsqueeze(-1), self.effect_table[x['eff_id']]], -1)
        eff = (self.effect(eff) * (x['eff_id'] > 0).unsqueeze(-1)).sum(-2)
        o = torch.cat([card, data + self.side(x['obj_side']) + self.kind(x['obj_kind']), eff,
                       torch.zeros_like(card), x['obj_float']], -1)
        o = self.object(o.reshape(N, -1, o.shape[-1]))
        present = (x['obj_side'] > 0).reshape(N, -1)
        if self.attention is not None:
            # a row with no object at all (a padded time step) would have every key masked: NaN; leave it unmasked
            pad = ~present & present.any(1, keepdim=True)
            if self.dist_bias is None:
                o = self.attention(o, src_key_padding_mask=pad)
            else:  # [N * heads, L, L]: a bias for each distance in whole tiles (0 .. 15+), padding -inf
                xy = torch.stack([x['obj_float'][..., 0].reshape(N, -1) * W, x['obj_float'][..., 1].reshape(N, -1) * H], -1)
                tiles_apart = (xy[:, :, None] - xy[:, None]).float().norm(dim=-1).long().clamp(max=15)
                bias = self.dist_bias(tiles_apart).permute(0, 3, 1, 2).to(o.dtype)
                bias = bias.masked_fill(pad[:, None, None, :], float('-inf'))
                o = self.attention(o, mask=bias.reshape(-1, *bias.shape[2:]))
        o = o * present.unsqueeze(-1)
        # place on the tile picture (this player's view: x, y are the first two floats)
        fx = x['obj_float'][..., 0].reshape(N, -1)
        fy = x['obj_float'][..., 1].reshape(N, -1)
        cells = ((fy * H).long().clamp(0, H - 1) * W + (fx * W).long().clamp(0, W - 1))
        grid = o.new_zeros(N, CELLS, c.object_dim)
        grid.scatter_add_(1, cells.unsqueeze(-1).expand(-1, -1, c.object_dim), o)
        if self.flight is not None:  # plays in flight where they will land: ours (1) and the opponent's (2)
            cards = torch.cat([x['own_flight_card'].reshape(N, 1), x['opp_flight_card'].reshape(N, 2)], 1)
            floats = torch.cat([x['own_flight_float'].reshape(N, 1, 4), x['opp_flight_float'].reshape(N, 2, 4)], 1)
            side = torch.tensor([1, 2, 2], device=cards.device).expand(N, 3)
            f = self.flight(torch.cat([self.card(cards), self.side(side), floats], -1)) * (cards > 0).unsqueeze(-1)
            spots = (floats[..., 1] * H).long().clamp(0, H - 1) * W + (floats[..., 0] * W).long().clamp(0, W - 1)
            grid.scatter_add_(1, spots.unsqueeze(-1).expand(-1, -1, c.object_dim), f.to(grid.dtype))
        grid = grid.transpose(1, 2).reshape(N, c.object_dim, H, W)
        grid = torch.cat([grid, self.coords.expand(N, -1, -1, -1)], 1)
        if c.res_blocks:
            grid = self.convs[0](grid)
            for block in self.blocks:
                grid = block(grid)
            grid = F.relu(self.tower_out(grid))
        else:
            for conv in self.convs:
                grid = F.relu(conv(grid))
        count = present.sum(1, keepdim=True).clamp(min=1)
        pooled_objects = torch.cat([o.sum(1) / count, o.masked_fill(~present.unsqueeze(-1), -1e4).amax(1).clamp(min=-1e3)], -1)
        return grid, pooled_objects

    def forward(self, x, state=None):
        """x: dict of [B, T, ...] tensors. -> dict: what [B, T, 4], which [B, T, 4], grid [B*T, C, H, W] (for tile
        scores), core [B*T, core], value [B, T], state."""
        B, T = x['scalars'].shape[:2]
        N = B * T
        flat = {k: v.reshape(N, *v.shape[2:]) for k, v in x.items()}
        grid, pooled_objects = self._objects(flat, N)
        hand = torch.cat([self.card(flat['hand_card']), self.card(flat['mirror_card']), flat['hand_float']], -1)
        rest = torch.cat([
            self.card(flat['cycle_card']).flatten(1), flat['ability'].flatten(1),
            torch.cat([self.card(flat['own_flight_card']), flat['own_flight_float']], -1),
            torch.cat([self.card(flat['opp_flight_card']), flat['opp_flight_float']], -1).flatten(1),
            torch.cat([self.card(flat['opp_seen_card']), flat['opp_seen_float']], -1).flatten(1),
            self.card(flat['opp_hero_card']).flatten(1), flat['scalars']], -1)
        summary = [grid.mean((2, 3)), grid.amax((2, 3)), pooled_objects, hand.flatten(1), rest]
        if self.board is not None:
            summary.append(self.board(grid).flatten(1))
        summary = torch.cat(summary, -1)
        summary = self.core_norm(F.relu(self.summary(self.summary_norm(summary)))).reshape(B, T, -1)
        core, state = self.lstm(summary, state)
        flat_core = core.reshape(N, -1)
        core_hand = torch.cat([flat_core.unsqueeze(1).expand(-1, 4, -1), hand], -1)
        empty = flat['hand_card'] == 0
        which = self.which(core_hand).squeeze(-1).masked_fill(empty, float('-inf'))
        what = self.what(core)
        ability_ready = flat['ability'][:, :, 1] > 0                              # [N, 2]
        what = what.masked_fill(torch.cat([torch.zeros_like(ability_ready[:, :1]), empty.all(-1, keepdim=True),
                                           ~ability_ready], -1).reshape(B, T, 4), float('-inf'))
        value = self.value(torch.cat([core, x['scalars']], -1)).squeeze(-1)
        out = {'what': what, 'which': which.reshape(B, T, 4), 'grid': grid, 'core': flat_core, 'value': value,
               'state': state, 'hand': hand}
        if self.config.next_play:
            out['next_when'] = self.next_when(core)
            out['next_which'] = self.next_which(core_hand).squeeze(-1).masked_fill(empty, float('-inf')).reshape(B, T, 4)
        return out

    def tiles(self, out, x, slot):
        """Tile scores [N, CELLS] for the card in hand slot `slot` [N] (the picture conditioned on that card), with
        illegal tiles -inf when x has legal masks."""
        N = out['core'].shape[0]
        card = self.card(x['hand_card'].reshape(N, 4).gather(1, slot.unsqueeze(1)).squeeze(1))
        # in fp32 even under autocast: the scores drift together (softmax ignores a shared offset) into the hundreds or
        # thousands, where bf16 rounds neighbouring tiles to the same score (2026-10-16: 85% of plays tied, loss 2.8 -> 5)
        with torch.autocast(out['core'].device.type, enabled=False):
            scale, shift = self.film(torch.cat([out['core'].float(), card.float()], -1)).chunk(2, -1)
            g = F.relu(out['grid'].float() * (1 + scale[..., None, None]) + shift[..., None, None])
            for block in self.tile_blocks:
                g = block(g)
            scores = self.tile(g).reshape(N, CELLS)
        if 'legal' in x:
            legal = x['legal'].reshape(N, 4, CELLS).gather(1, slot.view(N, 1, 1).expand(-1, 1, CELLS)).squeeze(1)
            scores = scores.masked_fill(~legal, float('-inf'))
        return scores


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


class Games(torch.utils.data.IterableDataset):
    """The player views of a dataset in a seeded random order (validation: in order), split over ranks and workers."""

    def __init__(self, root, entries, chunk, seed, validation=False, rank=0, world=1):
        self.root, self.entries, self.chunk, self.seed = Path(root), entries, chunk, seed
        self.validation, self.rank, self.world = validation, rank, world

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed)
        entries = list(self.entries)
        if not self.validation:
            rng.shuffle(entries)
        rng = random.Random(self.seed * 1000 + self.rank * nw + wid)
        for e in entries[self.rank * nw + wid::self.world * nw]:
            yield from chunks(read(self.root / e['file'], e['owner']), self.chunk, self.validation, rng)


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


# ---- training -----------------------------------------------------------------------------------------------------------
DEFAULTS = dict(epochs=4.0, batch=32, chunk=128, workers=4, lr=3e-4, final_lr=1.0, weight_decay=0.01, warmup=500, value_coef=0.5,
                next_play=0.5, card_dim=32, object_dim=32, attention_layers=1, conv_channels=48, conv_layers=4, core=256, board=0,
                res_blocks=0, flight_grid=0, tile_blocks=0, board_rows=8, board_cols=4, res_norm=1, dist_bias=0,
                card_stats=0, max_train=0, max_val=0,
                seed=0, log_every=50, checkpoint_every=1000, device='cuda')
# final_lr: the learning rate falls linearly to lr x final_lr by the last step (1.0 = constant)
# medium model (3.4M, 2026-10-16): --card_dim 64 --object_dim 96 --attention_layers 3 --conv_channels 96 --conv_layers 6
# --core 384
# large model (8.0M, 2026-10-18): --card_dim 64 --object_dim 160 --attention_layers 6 --conv_channels 160 --conv_layers 12
# --core 448 --board 32


def train(data, out, settings):
    out = Path(out)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = settings['device'] == 'cuda'
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if cuda else torch.device('cpu')
    torch.set_num_threads(2)
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(minutes=15))
    manifest = json.loads((Path(data) / 'manifest.json').read_text())
    vocab = C.Vocab(manifest['schema']['vocab'])
    train_entries = [e for e in manifest['entries'] if not e['validation']]
    val_entries = [e for e in manifest['entries'] if e['validation']]
    if settings.get('max_train'):  # ablations: the same random subset for every run with the same seed
        train_entries = random.Random(settings['seed']).sample(train_entries, min(settings['max_train'], len(train_entries)))
    if settings.get('max_val'):
        val_entries = random.Random(settings['seed'] + 1).sample(val_entries, min(settings['max_val'], len(val_entries)))
    batch = max(1, settings['batch'] // world)
    # ponytail: an epoch is an estimate (random chunk offsets vary the count a little); exact epoch ends are not kept
    total = int(settings['epochs'] * sum(math.ceil(e['decisions'] / settings['chunk']) for e in train_entries) // settings['batch'])

    torch.manual_seed(settings['seed'])
    stats = C.card_matrix(vocab) if settings.get('card_stats') else None  # --card_stats 1: the table's fields
    config = Config(vocab_size=len(vocab), next_play=settings['next_play'] > 0, **{k: settings[k] for k in
                    ('card_dim', 'object_dim', 'attention_layers', 'conv_channels', 'conv_layers', 'core')},
                    board=settings.get('board', 0), card_stats=0 if stats is None else stats.shape[1],
                    **{k: settings.get(k, v) for k, v in dict(res_blocks=0, flight_grid=0, tile_blocks=0, board_rows=8,
                                                              board_cols=4, res_norm=1, dist_bias=0).items()})
    policy = Policy(config, C.effect_matrix(vocab), stats).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'], fused=cuda)
    step = 0
    if (out / 'latest.pt').exists():
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        policy.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']
    if cuda:  # measured faster on A100s (2026-10-14)
        for conv in policy.convs:
            conv.to(memory_format=torch.channels_last)
        policy.convs[0].register_forward_pre_hook(lambda m, a: (a[0].contiguous(memory_format=torch.channels_last),))
        policy.forward = torch.compile(policy.forward, dynamic=True)
    objective = Objective(policy, settings['value_coef'], settings['next_play'])
    if world > 1:
        objective = nn.parallel.DistributedDataParallel(objective, device_ids=[device.index] if cuda else None,
                                                        find_unused_parameters=True)
    autocast = (lambda: torch.autocast('cuda', torch.bfloat16)) if cuda else nullcontext
    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None

    def write(row):
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)

    def checkpoint():
        if rank == 0:
            tmp = out / 'latest.tmp'
            torch.save(dict(config=asdict(config), model=policy.state_dict(), optimizer=optimizer.state_dict(),
                            step=step, settings=settings, vocab=sorted(vocab.index)), tmp)
            tmp.replace(out / 'latest.pt')

    def loader(entries, seed, validation=False, rank=0, world=1):
        return torch.utils.data.DataLoader(Games(data, entries, settings['chunk'], seed, validation, rank, world),
                                           batch_size=batch, num_workers=settings['workers'], collate_fn=collate,
                                           pin_memory=cuda)

    def validate():
        if rank == 0:
            policy.eval()
            with torch.no_grad(), autocast():
                rows = [loss_and_metrics(policy, *({k: v.to(device) for k, v in g.items()} for g in b),
                                         settings['value_coef'], settings['next_play'])[1]
                        for b in loader(val_entries, 0, validation=True)]
            policy.train()
            write(dict(step=step, split='val', **summarize(rows)))
        if world > 1:
            dist.barrier()

    started, window = time.monotonic(), []
    while step < total:
        # a pass over the data; a resumed run starts a fresh order (seeded by the step it resumes at)
        with nn.parallel.distributed.Join([objective]) if world > 1 else nullcontext():
            for xs, ts in loader(train_entries, settings['seed'] * 1_000_003 + step, rank=rank, world=world):
                xs, ts = ({k: v.to(device, non_blocking=True) for k, v in g.items()} for g in (xs, ts))
                for group in optimizer.param_groups:
                    group['lr'] = settings['lr'] * min(1.0, (step + 1) / max(1, settings['warmup'])) * \
                        (1 - (1 - settings.get('final_lr', 1.0)) * step / total)
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    loss, metrics = objective(xs, ts)
                if not math.isfinite(metrics['loss']):
                    raise FloatingPointError(f'nonfinite loss at step {step}; latest.pt is unchanged')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                step += 1
                window.append(metrics)
                if rank == 0 and step % settings['log_every'] == 0:
                    write(dict(step=step, split='train', seconds=round(time.monotonic() - started), **summarize(window)))
                    window = []
                if step % settings['checkpoint_every'] == 0:
                    checkpoint()
                if step >= total:
                    break
        if world > 1:  # a rank whose share ran out first joined early (Join kept its model in sync, not its count)
            count = torch.tensor(step, device=device)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            step = int(count)
        checkpoint()
        validate()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='a converted dataset (with manifest.json)')
    parser.add_argument('--out', type=Path, required=True, help='the run directory')
    for key, value in DEFAULTS.items():
        parser.add_argument('--' + key, type=type(value), default=value)
    args = parser.parse_args()
    settings = {k: getattr(args, k) for k in DEFAULTS}
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / 'settings.json'
    if saved.exists():
        settings = json.loads(saved.read_text())  # a resumed run keeps its settings
    elif int(os.environ.get('RANK', 0)) == 0:
        saved.write_text(json.dumps(settings, indent=1))
    train(args.data, args.out, settings)


# ---- inference (play.py, arena.py, ppo.py) -------------------------------------------------------------------------------
CARD = 4  # actions: WAIT, PLAY (never sampled as such), ABILITY1, ABILITY2, then CARD + hand slot * CELLS + cell
ABILITY_EXPLORE = 0.01  # self-play: P(ability) >= this while one is ready (old PPO stopped pressing it at all)


def load(checkpoint, device='cpu'):
    """-> (model in eval mode, its vocabulary, the saved dict)."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Policy(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    return model.to(device).eval(), C.Vocab(saved['vocab']), saved


def pack(x):
    """One decision's inputs -> fewer bytes: present objects only (they come first), int32 ids, legal tiles as bits."""
    n = int((x['obj_side'] > 0).sum())
    out = {}
    for k, v in x.items():
        if k in C.OBJECT_KEYS:
            v = v[:n]
        out[k] = np.packbits(v) if k == 'legal' else v.astype(np.int32) if v.dtype == np.int64 else v
    return out


def pad_stack(arrays):
    """Arrays of one rank -> one array with a leading axis, each padded with zeros to the largest along every axis."""
    shape = np.max([a.shape for a in arrays], 0)
    out = np.zeros((len(arrays), *shape), arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[(i, *map(slice, a.shape))] = a
    return out


def tensors(arrays, device):
    """Stacked packed inputs (any leading axes, e.g. [B, T]) -> model inputs on `device`: int64 ids, legal tiles unpacked."""
    out = {}
    for k, v in arrays.items():
        if k == 'legal':
            v = np.unpackbits(v, -1)[..., :4 * CELLS].reshape(*v.shape[:-1], 4, H, W).astype(bool)
        elif v.dtype == np.int32:
            v = v.astype(np.int64)
        out[k] = torch.from_numpy(v).to(device, non_blocking=True)
    return out


def _log_softmax(logits):
    """log_softmax over the last axis; a row with nothing allowed (all -inf) stays -inf instead of NaN."""
    logits = logits.float()
    ok = torch.isfinite(logits).any(-1, keepdim=True)
    safe = torch.where(ok, logits, torch.zeros_like(logits)).log_softmax(-1)
    return safe.masked_fill(~torch.isfinite(logits), float('-inf'))


def what_logp(what, x, explore=0.0, which=None):
    """log p(what) [N, 4]: only waiting while a play of ours is in flight, and P(ability) >= explore while one is ready.
    With `which` (the card head's logits): a play of a card the client would refuse (not playable()) becomes a wait, as it
    does live, instead of moving to the cards it can pay for; p(play, card) of every playable card is unchanged."""
    busy = (x['own_flight_card'] > 0).unsqueeze(-1) & (torch.arange(4, device=what.device) != WAIT)
    logp = _log_softmax(what.float().masked_fill(busy, float('-inf')))
    if which is not None and 'scalars' in x:
        base = _log_softmax(which.float().masked_fill(~x['legal'].flatten(2).any(-1), float('-inf')))
        ok = playable(x)  # log P(a playable card | play); -1e9, not -inf, inside: logsumexp's gradient over all -inf is NaN
        keep = torch.where(ok.any(-1), base.masked_fill(~ok, -1e9).logsumexp(-1), float('-inf'))
        lost = logp[:, PLAY] + (-torch.expm1(keep)).clamp(min=1e-30).log()  # log P(play, then a refused card)
        logp = torch.cat([torch.logaddexp(logp[:, WAIT], lost)[:, None], (logp[:, PLAY] + keep)[:, None],
                          logp[:, ABILITY1:]], -1)
    if explore:
        ready = torch.isfinite(logp[:, ABILITY1:])
        n = ready.sum(-1, keepdim=True)
        floor = torch.cat([torch.zeros_like(logp[:, :ABILITY1]), ready / n.clamp(min=1)], -1)
        mixed = (1 - explore) * logp.exp() + explore * floor
        mixed = mixed.clamp(min=1e-30).log().masked_fill(mixed == 0, float('-inf'))  # log(0)'s gradient would be NaN
        logp = torch.where(n > 0, mixed, logp)
    return logp


HOLD_ELIXIR = C.HOLD_TICKS / 56  # elixir regenerated during the client's hold at 1x (C.REGEN: 1 per 56 ticks)


def playable(x):
    """[N, 4]: hand slots the client would play now: a legal tile, and the cost paid within the client's hold
    (C.HOLD_TICKS; a tap still short after it is refused, as arena.Engine.step and the live client do). From inputs every
    model already has: hand_float cost (Mirror's +1 included), scalars elixir_own and the elixir-rate one-hot."""
    legal = x['legal'].flatten(2).any(-1)
    if 'hand_float' not in x or 'scalars' not in x:
        return legal
    s = x['scalars'].float()
    elixir = 10 * s[:, 0]
    rate = s[:, 2] + 2 * s[:, 3] + 3 * s[:, 4]
    cost = 10 * x['hand_float'][..., 0].float()
    return legal & (cost <= (elixir + HOLD_ELIXIR * rate + 1e-3).unsqueeze(-1))


def which_logp(which, x):
    """log p(hand slot | play) [N, 4]: only playable() cards (no legal tile now, e.g. Mirror before any play, or not paid
    for within the client's hold: the engine and the live client refuse those taps)."""
    return _log_softmax(which.float().masked_fill(~playable(x), float('-inf')))


def tile_logp(model, out, x, slot, rows=None):
    """log p(cell | slot) [n, CELLS] for the decisions `rows` (default all) of a forward pass `out` and its flat inputs x."""
    sub = out if rows is None else {'core': out['core'][rows], 'grid': out['grid'][rows]}
    sx = {k: x[k] if rows is None else x[k][rows] for k in ('hand_card', 'legal')}
    return _log_softmax(model.tiles(sub, sx, slot))


def flat(out, x):
    """A forward pass on [B, T] inputs -> its heads and inputs as [B * T, ...]."""
    N = out['core'].shape[0]
    return ({'what': out['what'].reshape(N, 4), 'which': out['which'].reshape(N, 4), 'core': out['core'],
             'grid': out['grid']}, {k: v.reshape(N, *v.shape[2:]) for k, v in x.items()})


def _gumbel_argmax(logp):
    return (logp - torch.empty_like(logp).exponential_().log()).argmax(-1)


def sharpen(logp, t, mix=0.0):
    """Log-probs at temperature t, renormalised; a row with nothing legal stays all -inf. mix > 0: that share of the
    untempered distribution mixed in (the learner explores around the sharpened policy it is deployed as)."""
    if t == 1.0:
        return logp
    if t <= 0:  # greedy: all of the probability on the best legal choice (a row with nothing legal stays all -inf)
        best = torch.full_like(logp, float('-inf')).scatter(-1, logp.argmax(-1, keepdim=True), 0.0)
        return torch.where(torch.isfinite(logp).any(-1, keepdim=True), best, logp)
    sharp = logp / t
    sharp = sharp - sharp.logsumexp(-1, keepdim=True)
    sharp = torch.where(sharp.isnan(), torch.full_like(sharp, float('-inf')), sharp)
    if not mix:
        return sharp
    legal = sharp > float('-inf')  # masked (-inf or the dtype's min); finite inputs only, or the backward gives NaN
    a, b = (torch.where(legal, v, torch.zeros_like(v)) for v in (sharp + math.log1p(-mix), logp + math.log(mix)))
    return torch.where(legal, torch.logaddexp(a, b), torch.full_like(logp, float('-inf')))


def sample(model, out, x, explore=0.0):
    """One action per decision of a forward pass on [N, 1] inputs -> (action [N], its log-prob [N])."""
    out, x = flat(out, x)
    mx = x if getattr(model, 'elixir_mask', True) else {k: x[k] for k in ('legal', 'own_flight_card')}  # off: as before PR 11
    what_lp, which_lp = what_logp(out['what'], mx, explore, out['which']), which_logp(out['which'], mx)
    t, mix = getattr(model, 'temperature', 1.0), getattr(model, 'mix', 0.0)  # < 1: card and tile sharper; when to play stays
    which_lp = sharpen(which_lp, t, mix)
    what, slot = _gumbel_argmax(what_lp), _gumbel_argmax(which_lp)
    # tiles for every row, played or not: the decision servers capture this in a CUDA graph (no nonzero(), fixed shapes)
    tile_lp = sharpen(tile_logp(model, out, x, slot), t, mix)
    cell_ = _gumbel_argmax(tile_lp)
    play = what == PLAY
    pick = lambda lp, i: lp.gather(1, i.unsqueeze(1)).squeeze(1)  # noqa: E731
    logp = pick(what_lp, what) + torch.where(play, pick(which_lp, slot) + pick(tile_lp, cell_), torch.zeros_like(what_lp[:, 0]))
    return torch.where(play, CARD + slot * CELLS + cell_, what), logp


def split(action):
    """Actions -> (what, hand slot, cell, played a card)."""
    play = action >= CARD
    card = (action - CARD).clamp(min=0)
    return torch.where(play, torch.full_like(action, PLAY), action), card // CELLS, card % CELLS, play


def decode(action, owner):
    """An action -> ('wait',) | ('ability',) | ('card', hand slot, x, y) in world units."""
    if action == WAIT:
        return ('wait',)
    if action in (ABILITY1, ABILITY2):
        return ('ability',)  # the first ready controller is pressed (decks with two are rare)
    row, col = divmod((action - CARD) % CELLS, W)
    x, y = C.rotate(col * 1000 + 500, row * 1000 + 500, owner)
    return ('card', (action - CARD) // CELLS, x, y)


class Agent:
    """One player: act(inputs) -> decode()'s action, sampled from the policy. `inputs`: one PlayerView.step()."""

    def __init__(self, model, owner, device='cpu'):
        self.model, self.owner, self.device, self.lstm, self.value, self.p_what = model, owner, device, None, 0.0, None

    @torch.no_grad()
    def act(self, inputs):
        x = tensors({k: v[None, None] for k, v in pack(inputs).items()}, self.device)
        out = self.model(x, self.lstm)
        self.lstm = out['state']
        self.value = float(out['value'].reshape(-1)[0])  # the IL value head: expected outcome, -1 loss .. +1 win
        self.p_what = out['what'].float().softmax(-1).reshape(-1).tolist()  # wait / play / ability 1 / ability 2
        action, _ = sample(self.model, out, x)
        return decode(int(action[0]), self.owner)


if __name__ == '__main__':
    main()
