"""Run a trained policy in the native engine, using exactly the training feature code."""
import numpy as np
import torch

from . import data as D
from .model import INPUT_KEYS, Policy


def load(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Policy(**saved['config']).to(device).eval()
    model.load_state_dict(saved['model'])
    return model, saved['vocab']


class Agent:
    """One player. `act(state)` returns ('wait',), ('ability',) or ('card', slot, x, y) in world units."""

    def __init__(self, model, vocab, owner, sample=True, device='cpu'):
        self.model, self.vocab, self.owner, self.sample, self.device = model, vocab, owner, sample, device
        self.reset()

    def reset(self):
        self.lstm, self.previous, self.plays = None, None, []

    def observe(self, state):
        """Feature row for `state`; features only need the previous state and all plays so far."""
        from native_engine.build_cache import encode  # Mac side only
        window = [self.previous or state, state]
        raw = encode({'states': window, 'commands': [], 'receipts': [], 'winner_matches': 0, 'crowns_matches': 0})
        self.plays += [(1, p.tick, p.owner, p.card) for p in state.plays]  # state.plays: new since last capture
        raw['plays'] = np.array(self.plays, np.int32).reshape(-1, 4)
        self.previous = state
        out = D.features(raw, self.owner, self.vocab)
        return {k: torch.as_tensor(out[k][1:2])[None].to(self.device) for k in INPUT_KEYS}

    @torch.no_grad()
    def act(self, state):
        logits, value, self.lstm = self.model(self.observe(state), self.lstm)
        logits = logits[0, 0].float()
        action = int(torch.distributions.Categorical(logits=logits).sample() if self.sample else logits.argmax())
        if action == D.WAIT:
            return ('wait',)
        if action == D.ABILITY:
            return ('ability',)
        slot, cell = divmod(action - 2, D.CELLS)
        x, y = D.cell_center(cell, self.owner)
        return ('card', slot, int(x), int(y))
