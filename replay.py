"""Rebuild human replays (IL_Replay parquet) in the native engine.

    python -m native_engine.replay data/IL_Replay/replays/part-000000.parquet --count 5
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

# ponytail: borrow FirstLight's verified replay converter (deal selection, side flips,
# forms, tower troops, ability hints) instead of porting ~2k lines. Pure Python, no engine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'firstlight-cr'))
from native_runner.contracts import ActionKind  # noqa: E402
from native_runner.match_factory import MatchConfig  # noqa: E402
from native_runner.royaleapi_replay import (  # noqa: E402
    _replay_card_plays, _special_deal_target, prepare_collected_replay)

from .engine import Engine  # noqa: E402

DECISION_TICKS = 5


@dataclass(frozen=True)
class Command:
    tick: int  # First state tick that includes the command.
    owner: int
    card: int | None  # None for an ability activation.
    x: int = 0
    y: int = 0
    hints: tuple[str, ...] = ()


def match_dict(replay, decks, seed):
    """Native match inputs; mirrors FirstLight's replay_viewer._match_config_from_replay.
    `decks` are per-owner card orders (the deal depends on the order)."""
    c, t = replay.episode_config, replay.episode_config.tags
    forms = [dict(zip(d, t[f'deck{o}_form_availability'])) for o, d in enumerate((c.deck0, c.deck1))]
    return MatchConfig(
        deck0=decks[0], deck1=decks[1], seed=seed, game_mode=c.game_mode, arena=c.arena,
        location=t['location'], level_cap=t['level_cap'], minimum_card_level=t['minimum_card_level'],
        deck0_form_availability=tuple(forms[0][card] for card in decks[0]),
        deck1_form_availability=tuple(forms[1][card] for card in decks[1]),
        tower_troop0_id=t['tower_troop0_id'], tower_troop1_id=t['tower_troop1_id'],
        king_tower_level=t['king_tower_level'], end_tick=t['end_tick'],
    ).to_replay_dict()


def dealt(state, owner):
    player = state.players[owner]
    return tuple(c.card for c in sorted(player.hand, key=lambda c: c.slot)), tuple(player.cycle)


def reset_with_deal(engine, prepared, seeds=8):
    """Reset so each owner's opening hand and queue equal a replay-compatible deal.

    For a given seed the native deal is a fixed permutation of deck slots: observe where
    each slot lands and reorder the deck to put the wanted cards there. Cards that may not
    start in hand (Elixir Collector, ...) are dealt to a seed-dependent queue position;
    keep them there and pick another deal that fits the public plays, or try the next seed.
    Mirrors FirstLight's calibrate_collected_replay_deal, without the renderer.
    """
    replay = prepared.replay
    tags = replay.episode_config.tags
    plays = _replay_card_plays(replay)
    for seed in [replay.episode_config.seed, *range(1, seeds)]:
        decks = [tuple(replay.episode_config.deck0), tuple(replay.episode_config.deck1)]
        want = [(prepared.opening_cards[o], prepared.queue_cards[o]) for o in (0, 1)]
        for _ in range(3):
            state = engine.reset(match=match_dict(replay, decks, seed))
            wrong = [o for o in (0, 1) if dealt(state, o) != want[o]]
            if not wrong:
                return state
            for o in wrong:
                hand, cycle = dealt(state, o)
                omitted = tags[f'deck{o}_omit_from_starting_hand_ids']
                if omitted:
                    want[o] = _special_deal_target(
                        replay_tag=prepared.replay_tag, owner=o, deck=decks[o], plays=plays[o],
                        omitted_ids=omitted, observed_opening=hand, observed_queue=cycle)
                    if want[o] is None:
                        break
                slots = [decks[o].index(card) for card in hand + cycle]
                order = [None] * 8
                for slot, card in zip(slots, want[o][0] + want[o][1]):
                    order[slot] = card
                decks[o] = tuple(order)
            else:
                continue
            break  # no compatible deal with this seed
    raise RuntimeError('could not reproduce the replay deal')


def replay_commands(replay):
    """Timed card/ability commands; mirrors FirstLight's replay_viewer._direct_render_commands."""
    commands = []
    for operation in replay.operations:
        for action in operation.actions:
            tick = operation.start_native_tick + max(1, action.execute_offset_ticks or 1)
            if action.kind is ActionKind.PLAY_CARD:
                (cx, cy), (ox, oy) = action.target_grid, action.subcell_offset or (0, 0)
                x = cx * 1000 + 500 + round(ox * 1000)
                y = cy * 1000 + 500 + round(oy * 1000)
                commands.append(Command(tick, action.owner, action.card_id,
                                        min(max(x, 0), 17_999), min(max(y, 0), 31_999)))
            elif action.kind is ActionKind.ACTIVATE_ABILITY:
                m = action.metadata
                # Several ability cards in one deck: let native pick the unique Ready one.
                hints = () if len(m.get('ability_source_keys', ())) > 1 else tuple(m.get('ability_runtime_hints', ()))
                commands.append(Command(tick, action.owner, None, hints=hints))
    return sorted(commands, key=lambda c: c.tick)


def rebuild(engine, payload):
    """Play one replay to its end; return per-decision states, commands, receipts and fidelity."""
    prepared = prepare_collected_replay(payload)
    replay = prepared.replay
    state = reset_with_deal(engine, prepared)
    commands = replay_commands(replay)
    sequences = engine.schedule([Engine.ability_command(c.owner, c.hints, c.tick) if c.card is None
                                 else Engine.card_command(c.owner, c.card, c.x, c.y, c.tick) for c in commands])
    end = replay.end_native_tick
    states = [state, *engine.run(DECISION_TICKS, (end - state.tick) // DECISION_TICKS)]
    if not states[-1].finalized and states[-1].tick < end:
        states.append(engine.advance(end - states[-1].tick).state)
    state = states[-1]
    receipts = engine.schedule_status(sequences)
    expected_crowns = {p['owner']: p['crowns'] for p in replay.terminal['players']}
    return {
        'tag': prepared.replay_tag,
        'states': states,
        'commands': commands,
        'receipts': receipts,
        'executed': sum(r.get('state') == 'succeeded' for r in receipts),
        'winner_matches': state.result == replay.terminal['winner'],
        'crowns_matches': state.crowns == (expected_crowns[0], expected_crowns[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet', type=Path)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--port', type=int, default=26789)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(args.parquet, columns=['payload_json']).slice(0, args.count).to_pylist()
    engine = Engine(port=args.port)
    totals = Counter()
    errors = Counter()
    for row in rows:
        started = time.perf_counter()
        try:
            r = rebuild(engine, json.loads(row['payload_json']))
        except Exception as error:  # report and continue: one bad replay must not stop the batch
            print(f'FAILED {type(error).__name__}: {error}')
            errors[str(error)[:70]] += 1
            continue
        totals.update(games=1, commands=len(r['commands']), executed=r['executed'],
                      winner=r['winner_matches'], exact=r['winner_matches'] and r['crowns_matches'],
                      all_executed=r['executed'] == len(r['commands']))
        failed = [x for x in r['receipts'] if x.get('state') != 'succeeded']
        print(f"{r['tag'][:12]} ticks={r['states'][-1].tick} decisions={len(r['states'])} "
              f"executed={r['executed']}/{len(r['commands'])} "
              f"winner={r['winner_matches']} crowns={r['crowns_matches']} "
              f"{time.perf_counter() - started:.1f}s")
        for x in failed[:3]:
            print('   rejected:', {k: x.get(k) for k in ('kind', 'cardId', 'executeTick', 'state', 'error')})
    print(f"\n{len(rows)} replays, {sum(errors.values())} failed to start: {dict(errors)}")
    g = max(totals['games'], 1)
    print(f"actions executed {totals['executed']}/{totals['commands']}; games with every action executed "
          f"{totals['all_executed'] / g:.1%}; winner matches {totals['winner'] / g:.1%}; "
          f"winner+crowns match {totals['exact'] / g:.1%}")


if __name__ == '__main__':
    main()
