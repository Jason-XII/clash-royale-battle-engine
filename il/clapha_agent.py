"""Plays a clapha-il checkpoint (github ClashAlpha/clapha-il: objects with unit states and effects, legal tiles; heads
what / which card / tile) in ClaphaEngine. Each decision's raw record (ClaphaEngine.record) and the card commands issued
so far go through clapha_il.features.PlayerView: the code the converted replays went through for imitation learning.

Needs clapha-il (CLAPHA_IL, default ~/clapha-il) and FirstLight's native_runner (FIRSTLIGHT, ~/firstlight-cr: the
legal-tile masks) on the machine.
"""
import os
from pathlib import Path
import sys

import numpy as np
import torch

CLAPHA_IL = Path(os.environ.get('CLAPHA_IL', Path.home() / 'clapha-il'))
FIRSTLIGHT = Path(os.environ.get('FIRSTLIGHT', Path.home() / 'firstlight-cr'))


def _modules():
    for p in (str(FIRSTLIGHT.parent), str(FIRSTLIGHT), str(CLAPHA_IL)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from clapha_il import features, model
    return features, model


def is_clapha_il(checkpoint):
    """A clapha-il checkpoint (train.py: config with object_dim, vocab.json next to it), not one of il.model's."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return 'object_dim' in saved.get('config', {})


def load(checkpoint, device='cpu'):
    FE, M = _modules()
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = M.Policy(M.Config(**saved['config'])).to(device).eval()
    model.load_state_dict(saved['model'])
    return model, FE.Vocab.load(Path(checkpoint).parent / 'vocab.json')


def legal_fn():
    """The legal tiles of a card (32 x 18 booleans), as train.py builds them: FirstLight's placement check, cached."""
    FE, _ = _modules()
    from clapha_il.legal_tiles import legal_rows
    cache = {}

    def legal(card, owner, lanes, last):
        key = (card, owner, lanes, last if card == FE.MIRROR else None)
        if key not in cache:
            cache[key] = np.array(legal_rows(card, owner, lanes=lanes, last_card=last), bool)
        return cache[key]
    return legal


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


class Agent:
    """One player. act(record, commands) -> ('wait',) | ('ability',) | ('card', hand slot, x, y) in world units. Wait /
    play / ability and the card are sampled from the model's heads (its top choice would almost never play: waiting is
    most of what humans do), the tile too unless tile='argmax'."""

    def __init__(self, model, vocab, owner, match, record, legal, tile='sample', device='cpu'):
        FE, M = _modules()
        self.FE, self.M, self.model, self.owner, self.tile, self.device = FE, M, model, owner, tile, device
        other = 1 - owner
        forms = [c.get('el', 0) for c in match['battle'][f'deck{other}']['sp']]  # 1 evolution, 2 hero (as the replays)
        heroes = FE.opponent_heroes([record], {'episode': {f'deck{other}_form_availability': forms}}, owner)
        self.view = FE.PlayerView(owner, vocab, heroes, legal)
        self.lstm = None

    @torch.no_grad()
    def act(self, record, commands):
        FE, M = self.FE, self.M
        x = self.view.step(record, commands)
        xs = {k: torch.as_tensor(v)[None, None].to(self.device) for k, v in x.items()}
        out = self.model(xs, self.lstm)
        self.lstm = out['state']
        what = int(torch.distributions.Categorical(logits=out['what'][0, 0].float()).sample())
        if what == M.WAIT:
            return ('wait',)
        if what in (M.ABILITY1, M.ABILITY2):
            return ('ability',)
        slot = int(torch.distributions.Categorical(logits=out['which'][0, 0].float()).sample())
        flat = {k: xs[k].reshape(1, *xs[k].shape[2:]) for k in ('hand_card', 'legal') if k in xs}
        tiles = self.model.tiles(out, flat, torch.tensor([slot], device=self.device))[0].float()
        cell = int(tiles.argmax()) if self.tile == 'argmax' else int(torch.distributions.Categorical(logits=tiles).sample())
        row, col = divmod(cell, FE.W)
        x_world, y_world = FE.rotate(col * 1000 + 500, row * 1000 + 500, self.owner)
        return ('card', slot, x_world, y_world)
