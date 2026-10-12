"""Small recurrent policy: entities scattered onto the 32x18 tile grid, a conv tower, one LSTM,
and a joint (wait | ability | hand slot x cell) action head (~0.9M parameters); `Agent` plays a checkpoint in any engine."""
import torch
from torch import nn
import torch.nn.functional as F

from . import data as D
from .data import ACTIONS, CELLS, ENTITY_FLOATS, EXTRAS, H, MAX_ENTITIES, REVEALED, SCALARS, W

INPUT_KEYS = ('ent_obj', 'ent_side', 'ent_type', 'ent_cell', 'ent_float', 'rev_card', 'rev_age',
              'scalars', 'hand_card', 'hand_cost', 'next_card', 'slot_ok',
              'extras', 'cycle_card', 'pending_card', 'ability_ok', 'opp_deck')


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
        # Added after IL (zero-initialised, so an older checkpoint plays exactly as before until PPO trains them):
        # extras + our next three cycle cards + the card in flight, into the LSTM input; and the opponent's whole
        # deck, into the value only (training-time information: it reduces the deck-matchup noise in advantages
        # and never reaches a decision).
        self.extras = nn.Linear(EXTRAS + 4 * card_dim, core)
        self.matchup = mlp(core + card_dim, 64, 1)
        for layer in (self.extras, self.matchup[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def initial_state(self, batch, device=None):
        zeros = torch.zeros(1, batch, self.config['core'], device=device)
        return zeros, zeros.clone()

    def forward(self, x, state=None, valid=None):
        """x: dict of [B, T, ...] tensors (INPUT_KEYS). Returns masked logits [B, T, ACTIONS],
        value [B, T] and the new LSTM state.

        valid: optional [B, T] bool (the PPO update's time mask). The per-step parts (entities, grid, convs, cell
        scores) then run on the valid steps only and padded steps get what a padded step gives anyway (no playable
        card: every cell -inf; ability_ok is False there); valid steps' values are the same as without it. Only the LSTM
        and the small per-step layers (added inputs, wait / ability, value) see every step."""
        B, T = x['scalars'].shape[:2]
        if valid is None:
            grid, g = self._steps(x, B, T)
            core, state = self.lstm(self._lstm_input(g, x), state)  # [B, T, core]
            cells = self._cells(grid, core, x, B, T)
        else:
            idx = valid.reshape(-1).nonzero().squeeze(1)  # valid steps, as rows of the flattened [B * T]
            n = len(idx)
            rows = {k: v.reshape(B * T, *v.shape[2:]).index_select(0, idx).unsqueeze(1) for k, v in x.items()}
            grid, g = self._steps(rows, n, 1)  # [n, ...]: each valid step as a sequence of one
            g = g.new_zeros(B * T, g.shape[-1]).index_copy(0, idx, g[:, 0]).reshape(B, T, -1)
            core, state = self.lstm(self._lstm_input(g, x), state)
            picked = core.reshape(B * T, -1).index_select(0, idx).unsqueeze(1)
            part = self._cells(grid, picked, rows, n, 1)[:, 0]
            cells = part.new_full((B * T, 4, CELLS), float('-inf')).index_copy(0, idx, part).reshape(B, T, 4, CELLS)
        wait_ability = self.wait_ability(core)
        ability = wait_ability[..., 1:].masked_fill(~x['ability_ok'].unsqueeze(-1), float('-inf'))
        logits = torch.cat([wait_ability[..., :1], ability, cells.flatten(2)], -1)
        deck = self.card(x['opp_deck']).mean(2)
        value = (self.value(torch.cat([core, x['scalars']], -1)) + self.matchup(torch.cat([core, deck], -1))).squeeze(-1)
        return logits, value, state

    def _steps(self, x, B, T):
        """Per-step part before the LSTM: the conv grid [B * T, C, H, W] and the LSTM's input [B, T, ...]."""
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
        return grid, torch.cat([pooled, x['scalars'], hand.flatten(2), nxt, rev.flatten(2)], -1)

    def _lstm_input(self, g, x):
        """[B, T, ...]: the per-step summary plus the inputs added after IL (zero-initialised layer)."""
        added = torch.cat([x['extras'], self.card(x['cycle_card']).flatten(2), self.card(x['pending_card'])], -1)
        return self.lstm_norm(F.relu(self.globals(self.global_norm(g))) + self.extras(added))

    def _cells(self, grid, core, x, B, T):
        """Per hand slot: condition the grid on (core, card) and score every cell -> [B, T, 4, CELLS], masked."""
        N, C = B * T, grid.shape[1]
        film = self.film(torch.cat([core.unsqueeze(2).expand(-1, -1, 4, -1), self.card(x['hand_card'])], -1))
        scale, shift = film.reshape(N, 4, 2 * C, 1, 1).chunk(2, 2)
        cells = self.cell(F.relu(grid.unsqueeze(1) * (1 + scale) + shift).reshape(N * 4, C, H, W))
        return cells.reshape(B, T, 4, CELLS).masked_fill(~x['slot_ok'].unsqueeze(-1), float('-inf'))


def decode(action):
    """Flat action -> ('wait',) | ('ability',) | ('card', slot, owner-relative cell)."""
    if action == 0:
        return ('wait',)
    if action == 1:
        return ('ability',)
    return ('card', (action - 2) // CELLS, (action - 2) % CELLS)


assert ACTIONS == 2 + 4 * CELLS


def load_weights(model, weights):
    """A checkpoint from before the extras / matchup layers loads with them at their zero initialisation."""
    missing, unexpected = model.load_state_dict(weights, strict=False)
    added = ('extras.', 'matchup.')
    if unexpected or any(not k.startswith(added) for k in missing):
        raise RuntimeError(f'checkpoint does not fit the model: missing {missing}, unexpected {unexpected}')


def load(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Policy(**saved['config']).to(device).eval()
    load_weights(model, saved['model'])
    return model, saved['vocab']


class Agent:
    """One player. `act(state)` returns ('wait',), ('ability',) or ('card', slot, x, y) in world units.

    decode: 'gate'   - play with probability 1 - p(wait), then the best card/cell/ability (default)
            'policy' - il.rl's PPO policy: wait with p(wait), else (ability | card x cell) sampled at its temperature
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

    def observe(self, state, in_flight=()):
        """Feature row for `state` ([1, 1, ...] tensors); the observer keeps the previous state and the plays so far.
        `in_flight`: il.data.InFlight taps not landed yet."""
        row = self.observer.observe(state, in_flight)[0]
        return {k: torch.as_tensor(row[k])[None, None].to(self.device) for k in INPUT_KEYS}

    @torch.no_grad()
    def act(self, state, in_flight=()):
        logits, value, self.lstm = self.model(self.observe(state, in_flight), self.lstm)
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
