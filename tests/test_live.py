"""Live play's bookkeeping: plays confirmed by the hand, screen coordinates."""
import play


def frame(tick, commands=(), hand=(0, 1, 2, 3)):
    return dict(game_tick=tick, coherent=True, failure='none', queue_ok=True, commands=list(commands),
                players=[dict(side=1, deck_card_ids=[26, 27, 28, 29, 30], hand_deck_indices=list(hand))])


def command(sequence=1, card=26, due=121):
    return dict(owner=1, sequence=sequence, card_id=card, issue_tick=due - 21, execute_tick=due, x=3500, y=25000)


def test_live_plays_are_confirmed_by_the_hand_not_the_queue():
    plays = play.LivePlays()
    assert plays.update(frame(110, [command()])) == []
    assert plays.commands()[0][0] == 121 and plays.commands()[0][1]['first_seen_tick'] == 110
    landed = plays.update(frame(121, [], hand=(4, 1, 2, 3)))
    assert len(landed) == 1 and landed[0][0] == 121 and landed[0][1]['x'] == 3500
    assert plays.flying == [] and len(plays.update(frame(122, [], hand=(4, 1, 2, 3)))) == 1
    # an incoherent frame confirms nothing; a hand change with no queued command has no coordinates
    other = play.LivePlays()
    other.update(frame(110, [command()]))
    bad = frame(121, [], hand=(4, 1, 2, 3))
    bad['coherent'] = False
    assert other.update(bad) == []
    blind = play.LivePlays()
    blind.update(frame(110))
    assert 'x' not in blind.update(frame(121, [], hand=(4, 1, 2, 3)))[0][1]
    # two cards at once, and our own just-sent tap, are not counted twice
    two = play.LivePlays()
    two.update(frame(110, [command(), command(2, 27, 122)]))
    assert len(two.commands(dict(card=26, due=121, x=3500, y=25000), 1)) == 2
    assert [c['card_id'] for _, c in two.update(frame(123, [], hand=(4, -1, 2, 3)))] == [26, 27]


def test_world_to_screen_puts_our_side_at_the_bottom():
    assert play.world_screen(9000, 3000, 0)[1] > play.world_screen(9000, 29000, 0)[1]   # owner 0's king is at y 3000
    assert play.world_screen(9000, 29000, 1)[1] > play.world_screen(9000, 3000, 1)[1]
