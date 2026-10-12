"""Small recurrent policy: entities scattered onto the 32x18 tile grid, a conv tower, one LSTM,
and a joint (wait | ability | hand slot x cell) action head (~0.9M parameters); `Agent` plays a checkpoint in any engine."""
import torch
from torch import nn
import torch.nn.functional as F

from . import data as D
from .data import ACTIONS, CELLS, ENTITY_FLOATS, H, MAX_ENTITIES, REVEALED, SCALARS, W

INPUT_KEYS = ('ent_obj', 'ent_side', 'ent_type', 'ent_cell', 'ent_float', 'rev_card', 'rev_age',
              'scalars', 'hand_card', 'hand_cost', 'next_card', 'slot_ok')


def mlp(*widths):
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class Policy(nn.Module):
    def __init__(self, vocab_size, card_dim=32, entity_dim=32, channels=48, core=256, conv_layers=4):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, card_dim=card_dim, entity_dim=entity_dim,
                           channels=channels, core=core, conv_layers=conv_layers)
        self.card = nn.Embedding(vocab_size, card_dim, padding_idx=0)
        self.side = nn.Embedding(3, card_dim, padding_idx=0)
        self.kind = nn.Embedding(5, card_dim, padding_idx=0)
        self.entity = mlp(card_dim + ENTITY_FLOATS, 64, entity_dim)
        convs = [nn.Conv2d(entity_dim + 2, channels, 3, padding=1)]
        convs += [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(conv_layers - 1)]
        self.convs = nn.ModuleList(convs)
        # Fixed coordinate channels so convolutions know where the river and towers are.
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        self.register_buffer('coords', torch.stack([xs, ys])[None], persistent=False)
        global_in = 2 * channels + SCALARS + 5 * (card_dim + 1) + REVEALED * (card_dim + 1)
        # Both norms keep the LSTM input at unit scale: unnormalised pooled grid features (~10-20)
        # saturated the LSTM in the first IL run, freezing its state and the value head.
        self.global_norm = nn.LayerNorm(global_in)
        self.globals = mlp(global_in, core, core)
        self.lstm_norm = nn.LayerNorm(core)
        self.lstm = nn.LSTM(core, core, batch_first=True)
        self.wait_ability = nn.Linear(core, 2)
        self.film = nn.Linear(core + card_dim, 2 * channels)
        self.cell = nn.Conv2d(channels, 1, 1)
        self.value = mlp(core + SCALARS, 128, 1)  # scalars (crowns, time, elixir) read directly

    def initial_state(self, batch, device=None):
        zeros = torch.zeros(1, batch, self.config['core'], device=device)
        return zeros, zeros.clone()

    def forward(self, x, state=None):
        """x: dict of [B, T, ...] tensors (INPUT_KEYS). Returns masked logits [B, T, ACTIONS],
        value [B, T] and the new LSTM state."""
        B, T = x['scalars'].shape[:2]
        N = B * T
        # Entities -> per-cell sums on the grid.
        ent = self.card(x['ent_obj']) + self.side(x['ent_side']) + self.kind(x['ent_type'])
        ent = self.entity(torch.cat([ent, x['ent_float']], -1))  # [B, T, E, D]
        ent = ent * (x['ent_side'] > 0).unsqueeze(-1)
        D = ent.shape[-1]
        grid = ent.new_zeros(N, CELLS, D)
        grid.scatter_add_(1, x['ent_cell'].reshape(N, MAX_ENTITIES, 1).expand(-1, -1, D), ent.reshape(N, -1, D))
        grid = grid.transpose(1, 2).reshape(N, D, H, W)
        grid = torch.cat([grid, self.coords.expand(N, -1, -1, -1)], 1)
        for conv in self.convs:
            grid = F.relu(conv(grid))
        C = grid.shape[1]
        pooled = torch.cat([grid.mean((2, 3)), grid.amax((2, 3))], 1).reshape(B, T, 2 * C)
        hand = torch.cat([self.card(x['hand_card']), x['hand_cost'].unsqueeze(-1)], -1)  # [B, T, 4, d+1]
        nxt = torch.cat([self.card(x['next_card']), torch.zeros_like(x['next_card'], dtype=hand.dtype).unsqueeze(-1)], -1)
        rev = torch.cat([self.card(x['rev_card']), x['rev_age'].unsqueeze(-1)], -1)
        g = torch.cat([pooled, x['scalars'], hand.flatten(2), nxt, rev.flatten(2)], -1)
        core, state = self.lstm(self.lstm_norm(F.relu(self.globals(self.global_norm(g)))), state)  # [B, T, core]
        # Per hand slot: condition the grid on (core, card) and score every cell.
        film = self.film(torch.cat([core.unsqueeze(2).expand(-1, -1, 4, -1), self.card(x['hand_card'])], -1))
        scale, shift = film.reshape(N, 4, 2 * C, 1, 1).chunk(2, 2)
        cells = self.cell(F.relu(grid.unsqueeze(1) * (1 + scale) + shift).reshape(N * 4, C, H, W))
        cells = cells.reshape(B, T, 4, CELLS).masked_fill(~x['slot_ok'].unsqueeze(-1), float('-inf'))
        logits = torch.cat([self.wait_ability(core), cells.flatten(2)], -1)
        value = self.value(torch.cat([core, x['scalars']], -1)).squeeze(-1)
        return logits, value, state


def decode(action):
    """Flat action -> ('wait',) | ('ability',) | ('card', slot, owner-relative cell)."""
    if action == 0:
        return ('wait',)
    if action == 1:
        return ('ability',)
    return ('card', (action - 2) // CELLS, (action - 2) % CELLS)


assert ACTIONS == 2 + 4 * CELLS


def load(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Policy(**saved['config']).to(device).eval()
    model.load_state_dict(saved['model'])
    return model, saved['vocab']


class Agent:
    """One player. `act(state)` returns ('wait',), ('ability',) or ('card', slot, x, y) in world units.

    decode: 'gate'   - play with probability 1 - p(wait), then the best card/cell/ability (default)
            'policy' - il.rl's PPO policy: wait with p(wait), else card x cell sampled at its temperature
            'sample' - sample the whole joint action
            'argmax' - joint argmax; almost never plays, because a decision to play is spread
                       over hundreds of card x cell options while WAIT is a single option
    """

    def __init__(self, model, vocab, owner, decode='gate', device='cpu'):
        self.model, self.vocab, self.owner, self.decode, self.device = model, vocab, owner, decode, device
        self.reset()

    def reset(self):
        from .data import Observer
        self.lstm, self.observer = None, Observer(self.vocab, (self.owner,))

    def observe(self, state):
        """Feature row for `state` ([1, 1, ...] tensors); the observer keeps the previous state and the plays so far."""
        row = self.observer.observe(state)[0]
        return {k: torch.as_tensor(row[k])[None, None].to(self.device) for k in INPUT_KEYS}

    @torch.no_grad()
    def act(self, state):
        logits, value, self.lstm = self.model(self.observe(state), self.lstm)
        logits = logits[0, 0].float()
        self.last_logits = logits  # for logging (il.live)
        if self.decode == 'policy':  # il.rl's training policy
            from .rl import distribution, sample
            action = int(sample(distribution(logits[None]))[0])
        elif self.decode == 'sample':
            action = int(torch.distributions.Categorical(logits=logits).sample())
        elif self.decode == 'argmax':
            action = int(logits.argmax())
        else:
            play = torch.rand(()) > logits.softmax(-1)[D.WAIT]
            action = 1 + int(logits[1:].argmax()) if play else D.WAIT
        if action == D.WAIT:
            return ('wait',)
        if action == D.ABILITY:
            return ('ability',)
        slot, cell = divmod(action - 2, D.CELLS)
        x, y = D.cell_center(cell, self.owner)
        return ('card', slot, int(x), int(y))
