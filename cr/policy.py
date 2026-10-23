"""The policy network and how it plays: loading a checkpoint, packing inputs, sampling actions, decoding them."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cr import observation as C

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



# ---- playing -------------------------------------------------------------------------------------------------------------
CARD = 4  # actions: WAIT, PLAY (never sampled as such), ABILITY1, ABILITY2, then CARD + hand slot * CELLS + cell
ABILITY_EXPLORE = 0.01  # self-play: P(ability) >= this while one is ready (old PPO stopped pressing it at all)


def load(checkpoint, device='cpu'):
    """-> (model in eval mode, its vocabulary, the saved dict). Checkpoints hold plain dicts (config, weights, vocabulary),
    so they load whatever module the code lives in."""
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


def what_logp(what, x, explore=0.0, which=None, overlap=False):
    """log p(what) [N, 4]: only waiting while a play of ours is in flight (overlap: only no ability press, as
    arena.Engine.overlap allows another card), and P(ability) >= explore while one is ready.
    With `which` (the card head's logits): a play of a card the client would refuse (not playable()) becomes a wait, as it
    does live, instead of moving to the cards it can pay for; p(play, card) of every playable card is unchanged."""
    busy = (x['own_flight_card'] > 0).unsqueeze(-1) & (torch.arange(4, device=what.device) >= (ABILITY1 if overlap else PLAY))
    logp = _log_softmax(what.float().masked_fill(busy, float('-inf')))
    if which is not None and 'scalars' in x:
        base = _log_softmax(which.float().masked_fill(~x['legal'].flatten(2).any(-1), float('-inf')))
        ok = playable(x, overlap)  # log P(a playable card | play); -1e9, not -inf, inside: logsumexp's gradient over all -inf is NaN
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


def playable(x, overlap=False):
    """[N, 4]: hand slots the client would play now: a legal tile, and the cost paid within the client's hold
    (C.HOLD_TICKS; a tap still short after it is refused, as arena.Engine.step and the live client do). From inputs every
    model already has: hand_float cost (Mirror's +1 included), scalars elixir_own and the elixir-rate one-hot. overlap:
    with a card in flight, not its slot (it leaves the hand when it lands) and its cost reserved, as Engine.step does."""
    legal = x['legal'].flatten(2).any(-1)
    if 'live_playable' in x:
        # The host has reconciled ALL queued cards and abilities (live: cr.live_budget; the simulator:
        # Engine.playable for an overlapping player). Do not subtract own_flight a second
        # time or reapply the older one-flight affordability approximation.
        return legal & x['live_playable'].bool()
    if 'hand_float' not in x or 'scalars' not in x:
        return legal
    s = x['scalars'].float()
    elixir = 10 * s[:, 0]
    if overlap:
        flying = x['own_flight_card'].reshape(-1)
        legal = legal & ~((x['hand_card'].reshape(-1, 4) == flying[:, None]) & (flying[:, None] > 0))
        elixir = elixir - 10 * x['own_flight_float'].reshape(-1, 4)[:, 3].float() * (flying > 0)
    rate = s[:, 2] + 2 * s[:, 3] + 3 * s[:, 4]
    cost = 10 * x['hand_float'][..., 0].float()
    return legal & (cost <= (elixir + HOLD_ELIXIR * rate + 1e-3).unsqueeze(-1))


def which_logp(which, x, overlap=False):
    """log p(hand slot | play) [N, 4]: only playable() cards (no legal tile now, e.g. Mirror before any play, or not paid
    for within the client's hold: the engine and the live client refuse those taps)."""
    return _log_softmax(which.float().masked_fill(~playable(x, overlap), float('-inf')))


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
    overlap = getattr(model, 'overlap', False)  # arena.py --a-overlap: may tap a card while another is in flight
    what_lp, which_lp = what_logp(out['what'], x, explore, out['which'], overlap), which_logp(out['which'], x, overlap)
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
    """An action -> ('wait',) | ('ability', controller 0 or 1) | ('card', hand slot, x, y) in world units."""
    if action == WAIT:
        return ('wait',)
    if action in (ABILITY1, ABILITY2):
        return ('ability', action - ABILITY1)
    row, col = divmod((action - CARD) % CELLS, W)
    x, y = C.rotate(col * 1000 + 500, row * 1000 + 500, owner)
    return ('card', (action - CARD) // CELLS, x, y)


class Agent:
    """One player: act(inputs) -> decode()'s action, sampled from the policy. `inputs`: one PlayerView.step()."""

    def __init__(self, model, owner, device='cpu'):
        self.model, self.owner, self.device, self.lstm, self.value, self.p_what = model, owner, device, None, 0.0, None

    @torch.no_grad()
    def act(self, inputs, *, live_playable=None, live_abilities=None):
        x = tensors({k: v[None, None] for k, v in pack(inputs).items()}, self.device)
        out = self.model(x, self.lstm)
        self.lstm = out['state']
        # Live-only action masks are attached AFTER the network forward pass:
        # checkpoint shapes, observations and the training sampler are unchanged.
        if live_playable is not None:
            x['live_playable'] = torch.as_tensor(live_playable, device=self.device).reshape(1, 1, 4)
        if live_abilities is not None:
            allowed = torch.as_tensor([True, True, *live_abilities], device=self.device)
            out = {**out, 'what': out['what'].masked_fill(~allowed, float('-inf'))}
        self.value = float(out['value'].reshape(-1)[0])  # the IL value head: expected outcome, -1 loss .. +1 win
        self.p_what = out['what'].float().softmax(-1).reshape(-1).tolist()  # wait / play / ability 1 / ability 2
        action, _ = sample(self.model, out, x)
        return decode(int(action[0]), self.owner)


def convolution_layout(model):
    """Match IL's convolution layout on CUDA; keep the fp32 tile head unchanged."""
    for conv in [*model.convs, *(c for b in getattr(model, 'blocks', ()) for c in (b.c1, b.c2))]:  # + residual blocks
        conv.to(memory_format=torch.channels_last)
    model.convs[0].register_forward_pre_hook(
        lambda m, a: (a[0].contiguous(memory_format=torch.channels_last),))

