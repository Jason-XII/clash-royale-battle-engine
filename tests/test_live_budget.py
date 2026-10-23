"""Live affordability regressions; no emulator or native engine is required."""
import math

import pytest

from cr import observation as C
from cr.live_budget import LiveBudget, Pending, balance_at, payment_tick


def card(cost=3, tap=100, due=122, slot=0, cid=26000000):
    return Pending('card', cost, tap, due, slot, cid, 3500, 10000)


def frame(tick, hand=(0, 1, 2, 3), commands=(), button=2, charges=-1):
    return dict(game_tick=tick, failure='none', coherent=True, queue_ok=True, commands=list(commands),
                players=[dict(side=0, deck_card_ids=[26000000, 26000001, 26000002, 26000003, 26000004],
                              hand_deck_indices=list(hand), abilities=[dict(controller_slot=1, button=button,
                                                                          charges=charges)])])


def test_ability_then_zap_cannot_spend_the_same_elixir_twice():
    pending = [Pending('ability', 1, 100, 122, 0)]
    assert payment_tick(2, 105, pending, 2) is None
    assert payment_tick(3, 105, pending, 2) == 127


def test_regeneration_can_fund_zap_before_its_payment_without_delaying_the_tap():
    # At 2x, 28 ticks regenerate exactly one elixir. A prior ability is due
    # in 5 ticks and a card in 10; Zap can be tapped at tick 3000 and lands 3022.
    # 5.25 + 22/28 - 3 - 1 - 2 >= 0. Neither prior action paid yet.
    pending = [card(due=3010), Pending('ability', 1, 2990, 3005, 0)]
    assert payment_tick(5.25, 3000, pending, 2) == 3022
    assert payment_tick(5.0, 3000, pending, 2) is None


def test_an_affordable_tap_does_not_gain_an_imaginary_extra_hold():
    # The visible bar pays for Zap now, so the client issues NOW. It won't
    # voluntarily hold it another 22 ticks to accommodate our reservations.
    pending = [card(due=3010), Pending('ability', 1, 2990, 3005, 0)]
    assert payment_tick(5, 3000, pending, 2) is None


def test_client_hold_boundary_and_regeneration_rate_transition():
    assert payment_tick(3 - C.hold_slack(100) / 10000, 100, [], 3) == 144
    assert payment_tick(3 - C.hold_slack(100) / 10000 - 0.001, 100, [], 3) is None
    assert payment_tick(3 - C.hold_slack(2390) / 10000, 2390, [], 3) == 2434
    assert payment_tick(0, 100, [], 1, ability=True) is None  # ability hold is unverified


def test_cap_and_payment_order_do_not_create_free_regeneration():
    assert balance_at(10, 100, [(156, 3), (212, 9)]) is None  # regen before 156 was leaked
    assert balance_at(10, 100, [(100, 3), (156, 8)]) == pytest.approx(0)
    assert balance_at(3, 100, [(110, 2), (110, 2)]) is None
    assert balance_at(3, 100, [(110, 2), (110, 2)]) == balance_at(3, 100, [(110, 4)])


def test_two_cards_are_opt_in_and_third_or_duplicate_slot_is_blocked():
    budget = LiveBudget(0, overlap=True)
    budget.pending.append(card())
    assert budget.card_due(10, 105, 1, 26000001, 3) == 127
    assert budget.card_due(10, 105, 0, 26000000, 3) is None
    budget.pending.append(card(tap=105, due=127, slot=1, cid=26000001))
    assert budget.card_due(10, 110, 2, 26000002, 1) is None
    serial = LiveBudget(0)
    serial.pending.append(card())
    assert serial.card_due(10, 105, 1, 26000001, 1) is None


def test_two_pending_costs_count_once_and_only_until_payment_is_observed():
    budget = LiveBudget(0, overlap=True)
    budget.pending.append(card())
    ack = dict(owner=0, card_id=26000000, sequence=7, issue_tick=104, execute_tick=125)
    budget.update(frame(110, commands=[ack]))
    assert len(budget.pending) == 1 and budget.pending[0].due == 125
    budget.update(frame(125, hand=(-1, 1, 2, 3)))
    assert len(budget.pending) == 1, 'empty slot does not prove the cost was paid'
    budget.update(frame(126, hand=(4, 1, 2, 3)))
    assert budget.pending == []
    assert budget.card_due(2, 126, 1, 26000001, 2) is not None, 'new balance must not be debited again'


def test_late_ack_and_stale_ack_and_incoherent_frames():
    budget = LiveBudget(0, overlap=True)
    budget.pending.append(card())
    stale = dict(owner=0, card_id=26000000, sequence=1, issue_tick=50, execute_tick=140)
    budget.update(frame(110, commands=[stale]))
    assert budget.pending[0].due == 122
    bad = frame(125, hand=(4, 1, 2, 3)); bad['coherent'] = False
    budget.update(bad)
    assert len(budget.pending) == 1
    budget.update(frame(123))
    assert len(budget.pending) == 1, 'estimated due passing does not release the reservation'
    with pytest.raises(RuntimeError, match='unconfirmed'):
        budget.update(frame(155))


def test_external_own_queue_is_reserved_but_opponents_queue_is_not():
    budget = LiveBudget(0, overlap=True)
    cmd = dict(owner=1, card_id=26000001, sequence=1, issue_tick=100, execute_tick=122)
    budget.update(frame(110, commands=[cmd]))
    assert budget.pending == []
    budget.update(frame(110, commands=[dict(cmd, owner=0)]))
    assert len(budget.pending) == 1 and budget.pending[0].cost == C.COSTS[26000001]


def test_ability_is_reserved_until_same_frame_reports_execution():
    budget = LiveBudget(0, ability_costs=(1, 2))
    due = budget.ability_due(3, 100, 0)
    assert due == 122
    budget.pending.append(Pending('ability', 1, 100, due, 0, charges=2))
    assert budget.ability_due(10, 105, 0) is None
    budget.update(frame(105, charges=2))
    assert len(budget.pending) == 1
    budget.update(frame(122, button=8, charges=1))
    assert budget.pending == []
    assert LiveBudget(0).ability_due(10, 100, 0) is None, 'unknown cost is not zero'


@pytest.mark.parametrize('cost', [-1, 11, math.inf, math.nan])
def test_invalid_costs_fail_before_a_tap(cost):
    with pytest.raises(ValueError):
        LiveBudget(0, ability_costs=(cost, 1))
    with pytest.raises(ValueError):
        payment_tick(10, 100, [], cost)
