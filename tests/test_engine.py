"""The simulator wrapper's play rules (no native engine: plays are only accepted or refused here)."""
from types import SimpleNamespace

from cr import engine, policy


def test_a_pending_ability_press_blocks_only_another_press():
    hand = tuple(engine.HandCard(i, 26000000 + i, 3, 0, 26000000 + i) for i in range(4))
    ready = engine.Ability(2, 0, 0, 1, 1, (5000010,))
    state = engine.State(1, 500, False, None, (0, 0), (engine.Player(0, 100000, hand, (), (ready,)),
                                                      engine.Player(1, 100000, hand, ())), (), (), 1)
    sim = engine.Engine.__new__(engine.Engine)  # no native engine: plays are only accepted or refused here
    sim.state, sim.pending, sim.overlap = state, [], [False, False]
    sim._release, sim._capture, sim.env = (lambda: []), (lambda: state), SimpleNamespace(step=lambda n: None)
    press, card, other = engine.Play(0, engine.ABILITY, 0, 0), engine.Play(0, 1, 9500, 5500), engine.Play(0, 2, 9500, 5500)
    assert sim.step([press], 5, delay=22).executed == (True,)
    assert sim.step([press, card], 5, delay=22).executed == (False, True), 'a card goes while the press is on its way'
    assert sim.step([other], 5, delay=22).executed == (False,), 'but not while another card is'


def test_a_press_executes_the_controller_the_model_chose():
    assert policy.decode(policy.ABILITY2, 0) == ('ability', 1)
    queued = []
    rows = [dict(controllerSlot=k + 1, buttonState=2, remainingCooldownMs=0, remainingChargesRaw=1, championObjectIds=[unit])
            for k, unit in enumerate((111, 222))]
    reader = SimpleNamespace(logic=lambda b: None, player=lambda logic, owner: None, abilities=lambda player: rows)
    sim = engine.Engine.__new__(engine.Engine)
    sim.env = SimpleNamespace(reader=reader, battle=None,
                              queue_ability_action_at=lambda command: queued.append(command.secondary_index))
    assert sim._activate(0, 1) and queued == [222]
    rows[1]['buttonState'] = 6  # used
    assert not sim._activate(0, 1) and queued == [222], 'a used button does nothing, though the other is ready'


def test_overlap_takes_two_cards_in_flight_and_the_sampler_mask_is_the_engines_rule():
    # with overlap a player may have CARDS_IN_FLIGHT cards on their way, never the slot already in flight; the mask
    # Game.views() gives its sampler (Engine.playable) is exactly what step() then accepts (23 Oct: the model's inputs
    # show one card in flight, so without it the sampler re-tapped the second: 88 of 94 refusals)
    hand = tuple(engine.HandCard(i, 26000000 + i, 1, 0, 26000000 + i) for i in range(4))
    state = engine.State(1, 500, False, None, (0, 0), (engine.Player(0, 100000, hand, ()),
                                                      engine.Player(1, 100000, hand, ())), (), (), 1)
    sim = engine.Engine.__new__(engine.Engine)
    sim.state, sim.pending, sim.overlap = state, [], [True, False]
    sim._release, sim._capture, sim.env = (lambda: []), (lambda: state), SimpleNamespace(step=lambda n: None)
    tap = lambda owner, slot: engine.Play(owner, slot, 9500, 5500)  # noqa: E731
    assert sim.playable(0).tolist() == [True] * 4
    assert sim.step([tap(0, 0)], 5, delay=22).executed == (True,)
    assert sim.playable(0).tolist() == [False, True, True, True], 'not the slot in flight'
    assert sim.step([tap(0, 0)], 5, delay=22).executed == (False,)
    assert sim.step([tap(0, 1)], 5, delay=22).executed == (True,)
    assert sim.playable(0).tolist() == [False] * 4, 'two cards in flight: no third'
    assert sim.step([tap(0, 2)], 5, delay=22).executed == (False,)
    assert sim.step([tap(1, 0), tap(1, 1)], 5, delay=22).executed == (True, False), 'without overlap: one at a time'
    assert sim.playable(1).tolist() == [False] * 4
