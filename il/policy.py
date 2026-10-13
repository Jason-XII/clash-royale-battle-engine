"""How a clapha-il model plays (github ClashAlpha/clapha-il; ~/clapha-il): load a checkpoint, build its inputs with
clapha_il.features.PlayerView from ClaphaEngine.record(), and the policy over its heads: what to do (wait / play / ability
1 / ability 2), which hand card, which tile for that card. Self-play (il.rl, il.actors), the arena and live play all play
through here, so the arena measures the policy PPO trains.

An action is one integer: WAIT 0, ABILITY1 2, ABILITY2 3 (the model's `what` classes), CARD + slot * CELLS + cell for a
card (cell owner-relative, as the model's tiles). Inputs travel and are stored packed (pack): present objects only, int32
ids, legal tiles as bits; `tensors` turns stacked packed rows back into model inputs.

Needs clapha-il (CLAPHA_IL, default ~/clapha-il) and FirstLight's native_runner (FIRSTLIGHT, ~/firstlight-cr: legal tiles).
"""
import os
from pathlib import Path
import sys

import numpy as np
import torch

CLAPHA_IL = Path(os.environ.get('CLAPHA_IL', Path.home() / 'clapha-il'))
FIRSTLIGHT = Path(os.environ.get('FIRSTLIGHT', Path.home() / 'firstlight-cr'))
for _path in (str(FIRSTLIGHT.parent), str(FIRSTLIGHT), str(CLAPHA_IL)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from clapha_il import features as FE  # noqa: E402
from clapha_il import model as M  # noqa: E402

WAIT, PLAY, ABILITY1, ABILITY2 = M.WAIT, M.PLAY, M.ABILITY1, M.ABILITY2
CARD = 4
PLAY_DELAY = (23, 27)  # decision -> landing ticks, drawn per play: a replay's play is labelled 23-27 ticks before it lands
CELLS = FE.CELLS
ABILITY_EXPLORE = 0.01  # self-play: P(ability) >= this while one is ready (old PPO stopped pressing it at all)
OBJECT_KEYS = ('obj_card', 'obj_data', 'obj_side', 'obj_kind', 'obj_float', 'eff_id', 'eff_left')


# ---- checkpoints ---------------------------------------------------------------------------------------------------------
def load(checkpoint, device='cpu'):
    """-> (model in eval mode, its vocabulary, the saved dict)."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = M.Policy(M.Config(**saved['config']))
    model.load_state_dict(saved['model'])
    vocab = FE.Vocab(saved['vocab']) if 'vocab' in saved else FE.Vocab.load(Path(checkpoint).parent / 'vocab.json')
    return model.to(device).eval(), vocab, saved


def is_clapha_il(checkpoint):
    """A clapha-il checkpoint (its config has object_dim), not one of the old il.model's."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return 'object_dim' in saved.get('config', {})


def legal_fn():
    """The legal tiles of a card (32 x 18 booleans), as clapha-il's train.py builds them, cached."""
    from clapha_il.legal_tiles import legal_rows
    cache = {}

    def legal(card, owner, lanes, last):
        key = (card, owner, lanes, last if card == FE.MIRROR else None)
        if key not in cache:
            cache[key] = np.array(legal_rows(card, owner, lanes=lanes, last_card=last), bool)
        return cache[key]
    return legal


# ---- inputs ----------------------------------------------------------------------------------------------------------------
class Commands:
    """Card plays issued so far as (execution tick, command), as a converted game's header holds them: landed plays from
    the states (the tick they left the hand), plays in flight from the engine's queue (their landing tick)."""

    def __init__(self):
        self.landed = []

    def update(self, state, engine):
        self.landed += [(p.tick, {'owner': p.owner, 'card_id': p.card}) for p in state.plays]
        flying = [(p['due'], {'owner': p['owner'], 'card_id': p['card'], 'x': p['x'], 'y': p['y']})
                  for p in engine.pending if p.get('card') is not None]
        return self.landed + flying


class Inputs:
    """The inputs of the players in `vocabs` ({owner: vocabulary}), decision by decision, in one ClaphaEngine game: the
    raw record once per decision, a PlayerView per player (the code the converted replays went through for IL)."""

    def __init__(self, engine, match, vocabs, legal):
        record = engine.record()
        self.engine, self.commands = engine, Commands()
        self.views = {}
        for owner, vocab in vocabs.items():
            other = 1 - owner
            forms = [c.get('el', 0) for c in match['battle'][f'deck{other}']['sp']]  # 1 evolution, 2 hero (as replays)
            heroes = FE.opponent_heroes([record], {'episode': {f'deck{other}_form_availability': forms}}, owner)
            self.views[owner] = FE.PlayerView(owner, vocab, heroes, legal)

    def __call__(self, state):
        record, issued = self.engine.record(), self.commands.update(state, self.engine)
        return {owner: view.step(record, issued) for owner, view in self.views.items()}


def pack(x):
    """One decision's inputs -> fewer bytes (pipes, trajectories): present objects only (they come first), int32 ids,
    legal tiles as bits."""
    n = int((x['obj_side'] > 0).sum())
    out = {}
    for k, v in x.items():
        if k in OBJECT_KEYS:
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


def stack(rows):
    """Packed rows -> one array per input [n, ...], objects padded to the most any row has (padding: no object)."""
    return {k: pad_stack([r[k] for r in rows]) for k in rows[0]}


def tensors(arrays, device):
    """Stacked packed inputs (any leading axes, e.g. [B, T]) -> model inputs on `device`: int64 ids, legal tiles unpacked."""
    out = {}
    for k, v in arrays.items():
        if k == 'legal':
            v = np.unpackbits(v, -1)[..., :4 * CELLS].reshape(*v.shape[:-1], 4, FE.H, FE.W).astype(bool)
        elif v.dtype == np.int32:
            v = v.astype(np.int64)
        out[k] = torch.from_numpy(v).to(device, non_blocking=True)
    return out


# ---- the policy ------------------------------------------------------------------------------------------------------------
def _log_softmax(logits):
    """log_softmax over the last axis; a row with nothing allowed (all -inf) stays -inf instead of NaN (in value and
    gradient)."""
    logits = logits.float()
    ok = torch.isfinite(logits).any(-1, keepdim=True)
    safe = torch.where(ok, logits, torch.zeros_like(logits)).log_softmax(-1)
    return safe.masked_fill(~torch.isfinite(logits), float('-inf'))


def what_logp(what, x, explore=0.0):
    """log p(what) [N, 4] from the what head [N, 4] and inputs x ([N, ...]): only waiting while a play of ours is in flight
    (the engine takes one at a time, as the live client does), and P(ability) >= explore while one is ready."""
    busy = (x['own_flight_card'] > 0).unsqueeze(-1) & (torch.arange(4, device=what.device) != WAIT)
    logp = _log_softmax(what.float().masked_fill(busy, float('-inf')))
    if explore:
        ready = torch.isfinite(logp[:, ABILITY1:])
        n = ready.sum(-1, keepdim=True)
        floor = torch.cat([torch.zeros_like(logp[:, :ABILITY1]), ready / n.clamp(min=1)], -1)
        mixed = (1 - explore) * logp.exp() + explore * floor
        mixed = mixed.clamp(min=1e-30).log().masked_fill(mixed == 0, float('-inf'))  # log(0)'s gradient would be NaN
        logp = torch.where(n > 0, mixed, logp)
    return logp


def which_logp(which, x):
    """log p(hand slot | play) [N, 4]: a card with no legal tile now (Mirror before any play) cannot be chosen."""
    return _log_softmax(which.float().masked_fill(~x['legal'].flatten(2).any(-1), float('-inf')))


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


def sample(model, out, x, explore=0.0):
    """One action per decision of a forward pass on [N, 1] inputs -> (action [N], its log-prob [N])."""
    out, x = flat(out, x)
    what_lp, which_lp = what_logp(out['what'], x, explore), which_logp(out['which'], x)
    what, slot = _gumbel_argmax(what_lp), _gumbel_argmax(which_lp)
    tile_lp = tile_logp(model, out, x, slot)
    cell = _gumbel_argmax(tile_lp)
    play = what == PLAY
    pick = lambda lp, i: lp.gather(1, i.unsqueeze(1)).squeeze(1)  # noqa: E731
    logp = pick(what_lp, what) + torch.where(play, pick(which_lp, slot) + pick(tile_lp, cell), torch.zeros_like(what_lp[:, 0]))
    return torch.where(play, CARD + slot * CELLS + cell, what), logp


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
        return ('ability',)  # ClaphaEngine presses the first ready controller (decks with two are rare)
    row, col = divmod((action - CARD) % CELLS, FE.W)
    x, y = FE.rotate(col * 1000 + 500, row * 1000 + 500, owner)
    return ('card', (action - CARD) // CELLS, x, y)


class Agent:
    """One player: act(inputs) -> decode()'s action, sampled from the policy as self-play plays it (without the ability
    exploration). `inputs`: this player's row of Inputs."""

    def __init__(self, model, owner, device='cpu'):
        self.model, self.owner, self.device, self.lstm = model, owner, device, None

    @torch.no_grad()
    def act(self, inputs):
        x = tensors({k: v[None, None] for k, v in pack(inputs).items()}, self.device)
        out = self.model(x, self.lstm)
        self.lstm = out['state']
        action, _ = sample(self.model, out, x)
        return decode(int(action[0]), self.owner)
