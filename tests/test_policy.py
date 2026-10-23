"""The network, imitation training and resume, and how the policy samples (masks, temperature, abilities, overlap)."""
import json
import math

import numpy as np
import pytest
import torch

from cr import imitation, observation as C, policy
from synthetic import arrays, dataset
import train


def test_training_runs_resumes_and_loads_for_play(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    settings = dict(train.DEFAULTS, epochs=2, batch=2, chunk=3, workers=0, card_dim=4, object_dim=4, conv_channels=4,
                    conv_layers=1, core=8, attention_layers=0, board=2, log_every=1, checkpoint_every=2, device='cpu')
    out = tmp_path / 'run'
    out.mkdir()
    train.train(root, out, settings)
    first = torch.load(out / 'latest.pt', weights_only=False)
    # 2 epochs x (3 + 4 chunks of the 7- and 11-decision training games, both players) / batch 2
    assert first['step'] == 2 * 2 * (3 + 4) // 2 == 14
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').open()]
    assert rows[-1]['split'] == 'val' and np.isfinite(rows[-1]['loss'])
    settings['epochs'] = 4  # continuing a finished run: only the extra steps
    train.train(root, out, settings)
    assert torch.load(out / 'latest.pt', weights_only=False)['step'] == 28
    model, vocab, _ = policy.load(out / 'latest.pt')
    x, _ = arrays(1, 0)
    action = policy.Agent(model, 0).act({k: v[0] for k, v in x.items()})
    assert action[0] in ('wait', 'card', 'ability')


@pytest.mark.parametrize('option', [{}, dict(res_blocks=1), dict(flight_grid=1), dict(tile_blocks=2),
                                    dict(board=2, board_rows=4, board_cols=3), dict(res_blocks=1, tile_blocks=1, res_norm=0),
                                    dict(attention_layers=2, dist_bias=1), dict(card_stats=3)])
def test_architecture_options_train_and_the_default_is_unchanged(tmp_path, monkeypatch, option):
    root = dataset(tmp_path / 'data', monkeypatch)
    entries = json.loads((root / 'manifest.json').read_text())['entries']
    xs, ts = imitation.collate(list(imitation.Games(root, entries, 6, 0, validation=True))[:3])
    torch.manual_seed(0)
    config = dict(vocab_size=len(C.Vocab([26000000])), card_dim=4, object_dim=8, conv_channels=8, conv_layers=4, core=8, attention_layers=0)
    stats = torch.rand(config['vocab_size'], option['card_stats']) if option.get('card_stats') else None
    model = policy.Policy(policy.Config(**{**config, **option}), card_table=stats)
    if option.get('card_stats'):  # starts as the plain embedding; a checkpoint carries its table
        assert torch.equal(model.card(torch.tensor([0, 2])), model.card.weight[[0, 2]])
        fresh = policy.Policy(policy.Config(**{**config, **option}))  # as policy.load builds it
        fresh.load_state_dict(model.state_dict())
        assert torch.equal(fresh.card.table, stats)
    if not option:  # the parameters every saved checkpoint has: no new keys, same shapes
        old = {'convs.0.weight', 'convs.3.weight', 'tile.weight', 'film.weight', 'lstm.weight_ih_l0'}
        assert old <= set(model.state_dict()) and not any(k.split('.')[0] in ('blocks', 'tower_out', 'flight', 'tile_blocks')
                                                         for k in model.state_dict())
    xs['own_flight_card'][:, :2], xs['own_flight_float'][:, :2, :2] = 1, 0.5  # something in flight
    for k in C.OBJECT_KEYS:  # a second object 9 tiles away (the toy games have one), so attention has a choice
        xs[k] = torch.cat([xs[k], xs[k]], 2)
    xs['obj_float'][:, :, 1, 0] += 0.5
    loss, m = imitation.loss_and_metrics(model, xs, ts, 0.5, 0.5)
    loss.backward()
    assert math.isfinite(loss.item()) and all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
    if option.get('flight_grid'):
        assert model.flight[0].weight.grad.abs().sum() > 0, 'plays in flight reach the board'
    if option.get('res_norm') == 0:
        assert all(b.gate.grad.abs() > 0 for b in [*model.blocks, *model.tile_blocks]), 'the gates learn'
    if option.get('dist_bias'):
        assert model.dist_bias.weight.grad.abs().sum() > 0, 'the distance bias learns'
    if option.get('card_stats'):
        assert model.card.stat.weight.grad.abs().sum() > 0, 'the stats reach the network'


def test_unaffordable_cards_are_not_offered_and_held_taps_are_in_flight():
    legal = torch.ones(1, 4, C.H, C.W, dtype=torch.bool)
    hf = torch.zeros(1, 4, C.HAND_FLOATS)
    hf[0, :, 0] = torch.tensor([3, 4, 5, 9]) / 10          # costs 3, 4, 5, 9
    sc = torch.zeros(1, C.SCALARS)
    sc[0, 0], sc[0, 2] = 0.40, 1.0                           # 4.0 elixir, 1x rate: the hold adds ~0.41
    x = {'legal': legal, 'hand_float': hf, 'scalars': sc, 'own_flight_card': torch.zeros(1, dtype=torch.long)}
    assert policy.playable(x).tolist() == [[True, True, False, False]]
    which = policy.which_logp(torch.zeros(1, 4), x)
    assert torch.isinf(which[0, 2:]).all() and torch.isfinite(which[0, :2]).all()
    # a refused card's share of PLAY goes to WAIT (as live: the tap does nothing); playable cards keep p(play, card)
    what, which_logits = torch.zeros(1, 4, requires_grad=True), torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
    lp = policy.what_logp(what, x, 0.0, which_logits)
    p_card = torch.softmax(which_logits[0].detach(), -1)                # costs 3 and 4 playable: 2 / (2 + 2e)
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.))
    assert torch.allclose(lp[0, policy.PLAY].exp(), 0.25 * p_card[:2].sum())
    assert torch.allclose(lp[0, policy.WAIT].exp(), 0.25 + 0.25 * p_card[2:].sum())
    joint = lp[0, policy.PLAY] + policy.which_logp(which_logits, x)[0, :2]
    assert torch.allclose(joint.exp(), 0.25 * p_card[:2]), 'p(play, card) unchanged for cards the client takes'
    sc[0, 0] = 0.0                                           # nothing affordable: playing is not an option
    lp = policy.what_logp(what, x, 0.0, which_logits)
    assert torch.isinf(lp[0, policy.PLAY]) and torch.allclose(lp[0, policy.WAIT].exp(), torch.tensor(.5))
    (lp[torch.isfinite(lp)].sum() + policy.which_logp(which_logits, x)[torch.isfinite(policy.which_logp(which_logits, x))].sum()).backward()
    assert torch.isfinite(what.grad).all() and torch.isfinite(which_logits.grad).all(), 'no NaN gradient'
    # PlayerView's own-flight rule: a held tap (tap known) is in flight from the tap, not only from landing - 22
    landed = [(200, {'owner': 0, 'card_id': 26000021, 'x': 3500, 'y': 10500, 'tap': 150})]
    assert [tt for tt, c in landed if c.get('tap', tt - C.DEPLOY_DELAY) <= 160 < tt] == [200]
    assert not [tt for tt, c in landed if tt - C.DEPLOY_DELAY <= 160 < tt]


def test_ability_labels_move_to_the_tap_and_the_ready_controller():
    what = np.zeros(25, np.int64)
    what[[2, 10, 15, 24]] = policy.ABILITY1  # convert.py labels every press controller 1, at its execution tick
    what[11] = policy.PLAY
    ability = np.zeros((25, 2, 4), np.float32)
    ability[:, 0, 1] = 1
    ability[6, 0, 1], ability[6, 1, 1] = 0, 1  # at decision 6 only the second controller is ready
    ability[20:23, 0, 1] = 0                    # the button lights up at 23
    out = imitation.ability_labels({'t_what': what.copy(), 'x_ability': ability})['t_what']
    assert out[6] == policy.ABILITY2 and out[10] == policy.WAIT, '4 decisions earlier, on the ready controller'
    assert out[0] == out[12] == out[23] == policy.ABILITY1 and out[11] == policy.PLAY, \
        'else the first ready decision after: the first decision, past a card play, once the button lit up'
    assert (out >= policy.ABILITY1).sum() == 4


def test_sharpened_log_probs_stay_normalised_and_masked():
    inf = float('-inf')
    lp = torch.tensor([[0.0, -1.0, inf], [inf, inf, inf]]).log_softmax(-1).nan_to_num(nan=inf)
    assert torch.equal(policy.sharpen(lp, 1.0), lp)
    cold = policy.sharpen(lp, 0.05)
    assert torch.allclose(cold[0].exp().sum(), torch.tensor(1.0)) and cold[0, 0].exp() > 0.99 and cold[0, 2] == inf
    assert torch.isinf(cold[1]).all(), 'nothing legal stays nothing legal'
    greedy = policy.sharpen(lp, 0)
    assert greedy[0, 0] == 0 and torch.isinf(greedy[0, 1:]).all() and torch.isinf(greedy[1]).all(), 'T 0 = the best legal choice'
    raw = lp.clone().requires_grad_()
    mixed = policy.sharpen(raw, 0.1, 0.1)
    assert torch.allclose(mixed[0].exp().sum(), torch.tensor(1.0)) and mixed[0, 2] == inf and torch.isinf(mixed[1]).all()
    assert mixed[0, 1] >= lp[0, 1] + math.log(0.1) - 1e-6, 'every legal action keeps at least mix x its own probability'
    mixed[0, :2].sum().backward()
    assert torch.isfinite(raw.grad[0]).all(), 'masked entries must not turn the gradient into NaN'


def test_overlap_lets_the_sampler_tap_a_second_card():
    legal = torch.ones(1, 4, C.H, C.W, dtype=torch.bool)
    hf = torch.zeros(1, 4, C.HAND_FLOATS)
    hf[0, :, 0] = 0.3                                        # every card costs 3
    sc = torch.zeros(1, C.SCALARS)
    sc[0, 0], sc[0, 2] = 0.65, 1.0                           # 6.5 elixir at 1x
    x = {'legal': legal, 'hand_float': hf, 'scalars': sc, 'hand_card': torch.tensor([[5, 6, 7, 8]]),
         'own_flight_card': torch.tensor([6]), 'own_flight_float': torch.tensor([[0.5, 0.2, 0.5, 0.3]])}
    what = torch.zeros(1, 4)
    assert torch.isinf(policy.what_logp(what, x)[0, 1:]).all(), 'as before: only waiting while a card is in flight'
    lp = policy.what_logp(what, x, 0.0, torch.zeros(1, 4), overlap=True)
    assert torch.isfinite(lp[0, :2]).all() and torch.isinf(lp[0, 2:]).all(), 'a card yes, an ability press no'
    assert policy.playable(x, overlap=True).tolist() == [[True, False, True, True]], 'not the slot in flight'
    sc[0, 0] = 0.5                                           # 5 - 3 reserved + the hold's ~0.4 < 3
    assert not policy.playable(x, overlap=True).any() and policy.playable(x).all()
