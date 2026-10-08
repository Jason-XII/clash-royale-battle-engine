"""Small recurrent policy: entities scattered onto the 32x18 tile grid, a conv tower, one LSTM,
and a joint (wait | ability | hand slot x cell) action head. About 1M parameters."""
import torch
from torch import nn
import torch.nn.functional as F

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
        self.globals = mlp(global_in, core, core)
        self.lstm = nn.LSTM(core, core, batch_first=True)
        self.wait_ability = nn.Linear(core, 2)
        self.film = nn.Linear(core + card_dim, 2 * channels)
        self.cell = nn.Conv2d(channels, 1, 1)
        self.value = mlp(core, 128, 1)

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
        core, state = self.lstm(F.relu(self.globals(g)), state)  # [B, T, core]
        # Per hand slot: condition the grid on (core, card) and score every cell.
        film = self.film(torch.cat([core.unsqueeze(2).expand(-1, -1, 4, -1), self.card(x['hand_card'])], -1))
        scale, shift = film.reshape(N, 4, 2 * C, 1, 1).chunk(2, 2)
        cells = self.cell(F.relu(grid.unsqueeze(1) * (1 + scale) + shift).reshape(N * 4, C, H, W))
        cells = cells.reshape(B, T, 4, CELLS).masked_fill(~x['slot_ok'].unsqueeze(-1), float('-inf'))
        logits = torch.cat([self.wait_ability(core), cells.flatten(2)], -1)
        return logits, self.value(core).squeeze(-1), state


def decode(action):
    """Flat action -> ('wait',) | ('ability',) | ('card', slot, owner-relative cell)."""
    if action == 0:
        return ('wait',)
    if action == 1:
        return ('ability',)
    return ('card', (action - 2) // CELLS, (action - 2) % CELLS)


assert ACTIONS == 2 + 4 * CELLS
