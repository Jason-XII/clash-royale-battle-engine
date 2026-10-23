"""Exercise the real sampler and the live loop's wiring, without taps on a device."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import play
from cr import observation as C, policy
from cr.live_budget import LiveBudget, Pending
from synthetic import arrays


def test_real_agent_applies_live_masks_without_changing_network_inputs():
    x, _ = arrays(1, 0)
    inputs = {k: v[0].copy() for k, v in x.items()}
    inputs['own_flight_card'] = np.array(6, np.int64)
    inputs['hand_card'] = np.array([5, 6, 7, 8], np.int64)
    inputs['hand_float'][:, 0] = 0.3
    before = {k: v.copy() for k, v in inputs.items()}

    class Model:
        overlap, temperature = True, 0

        def __call__(self, x, state):
            assert 'live_playable' not in x, 'host bookkeeping must not become a new checkpoint feature'
            return dict(what=torch.tensor([[[-100., 10., 30., 30.]]]),
                        which=torch.tensor([[[-100., -100., 100., -100.]]]),
                        core=torch.zeros(1, 1, 2), grid=torch.zeros(1, 1, 1, C.H, C.W),
                        value=torch.zeros(1, 1), state=None)

        def tiles(self, out, x, slot):
            return torch.zeros(len(slot), C.CELLS)

    agent = policy.Agent(Model(), 0)
    # Even with zero current elixir, the timeline's positive decision must not
    # be vetoed by the old raw-balance/one-flight test in playable().
    assert agent.act(inputs, live_playable=[False, False, True, False],
                     live_abilities=[False, False])[:2] == ('card', 2)
    assert agent.act(inputs, live_playable=[False] * 4, live_abilities=[False] * 2) == ('wait',)
    assert all(np.array_equal(inputs[k], v) for k, v in before.items())


def test_live_commands_keep_two_reservations_visible_without_duplicate_ack():
    plays = play.LivePlays()
    budget = LiveBudget(0, overlap=True)
    budget.pending = [Pending('card', 3, 100, 122, 0, 26, 3500, 10000),
                      Pending('card', 2, 105, 127, 1, 27, 4500, 10000)]
    plays.flying = [(122, dict(owner=0, card_id=26, x=3500, y=10000))]
    commands = plays.commands(reservations=budget.commands(), tick=110)
    assert len(commands) == 2 and commands[0][1]['tap'] == 100
    assert [c['card_id'] for _, c in commands] == [26, 27]
    # A missing/late ACK must not erase our local tap from the model input.
    plays.flying = []
    assert all(due > 130 for due, _ in plays.commands(reservations=budget.commands(), tick=130))


def test_held_card_blocks_overlapping_tap_until_issued():
    budget = LiveBudget(0, overlap=True)
    budget.pending.append(Pending('card', 3, 100, 140, 0, 26))
    assert budget.card_due(10, 105, 1, 27, 2) is None
    assert budget.card_due(10, 119, 1, 27, 2) is not None


@pytest.mark.parametrize('scenario', ['unfunded_zap', 'regeneration_funds_zap', 'two_cards'])
def test_battle_loop_reserves_and_checks_before_real_tap_calls(monkeypatch, scenario):
    start = 3000 if scenario == 'regeneration_funds_zap' else 100
    elixir = {'unfunded_zap': 20000, 'regeneration_funds_zap': 23000, 'two_cards': 100000}[scenario]

    def frame(tick, *, battle=1):
        return dict(game_tick=tick, coherent=True, failure='none', queue_ok=True, commands=[], battle_active=True,
                    chain=dict(battle=battle), entities=[], players=[
                        dict(side=o, deck_card_ids=[26, 27, 28, 29, 30], hand_deck_indices=[0, 1, 2, 3],
                             deck_form_flags=[0] * 5, cycle_deck_indices=[4], elixir_raw=elixir,
                             abilities=[dict(controller_slot=1, button=2, cooldown_ms=0, charges=-1, units=[100])]
                             if o == 0 else []) for o in (0, 1)])

    class Records:
        reset = lambda self: None

        def __call__(self, f):
            return dict(tick=f['game_tick'], objects=[], lean=dict(state=dict(players=[play._player(p) for p in f['players']]),
                        players=[dict(owner=o, abilityRuntime=[]) for o in (0, 1)]))

    inputs = dict(hand_float=np.array([[.2] + [0.] * (C.HAND_FLOATS - 1)] * 4, np.float32))
    monkeypatch.setattr(C, 'opponent_heroes', lambda *a: [])
    monkeypatch.setattr(C, 'cached_legal', lambda: None)
    monkeypatch.setattr(C, 'PlayerView', lambda *a: SimpleNamespace(step=lambda *a: inputs))
    calls = []

    class Agent:
        value, p_what = 0., [0.] * 4

        def __init__(self, *a):
            pass

        def act(self, inputs, **masks):
            calls.append(masks)
            if len(calls) == 1:
                assert masks['live_abilities'] == [True, False]
                return ('card', 0, 3500, 10000) if scenario == 'two_cards' else ('ability', 0)
            assert masks['live_abilities'] == [False, False]
            if scenario == 'unfunded_zap':
                assert masks['live_playable'] == [False] * 4
                return ('wait',)
            assert masks['live_playable'] == ([False, True, True, True] if scenario == 'two_cards' else [True] * 4)
            return ('card', 1, 4500, 10000)

    monkeypatch.setattr(play, 'Agent', Agent)
    tapped = []
    def card_tap(*a):
        if scenario == 'unfunded_zap':
            raise AssertionError('unfunded card was tapped')
        tapped.append(('card', *a))
        return (100, 200)

    taps = SimpleNamespace(tap=lambda *a: tapped.append(a), play=card_tap)
    play.play_battle(iter([frame(start + 5), frame(start + 10, battle=2)]), frame(start), SimpleNamespace(), {}, Records(),
                     0, taps, lambda entry: None, overlap=True, ability_costs=(1, 2))
    assert len(calls) == 2
    if scenario == 'unfunded_zap':
        assert tapped == [play.ABILITY_BUTTON]
    elif scenario == 'two_cards':
        assert [t[1] for t in tapped] == [0, 1], 'two different cards tapped five ticks (250 ms) apart'
    else:
        assert tapped[0] == play.ABILITY_BUTTON and tapped[1][0] == 'card'
