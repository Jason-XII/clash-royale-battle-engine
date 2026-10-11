"""Human replays (IL_Replay parquet) in the emulator engine (Mac): rebuild them, cache their raw per-decision states
(not features) for imitation learning, and extract the deck pool PPO plays with.

    python -m native_engine.build_cache data/IL_Replay/replays data/cache            # all parts
    python -m native_engine.build_cache data/IL_Replay/replays data/cache --parts 0 1 --limit 50
    python -m native_engine.build_cache data/IL_Replay/replays data/deck_pool.json --deck-pool 600   # per part
    python -m native_engine.build_cache data/IL_Replay/replays/part-000000.parquet - --check 5         # fidelity

One compressed shard per 1000 replays of a parquet part (`part-000000-0.npz` ...); finished
shards are skipped, so the job resumes after an interruption. Arrays of all replays are concatenated; replay i
owns rows `off[i]:off[i+1]` of each table (see il/data.py for the reader).

The replay converter is FirstLight's (../firstlight-cr; deal selection, side flips, forms, tower troops, ability
hints), imported on first use so this module (`encode`) also loads on the training node.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .engine import Engine

ENTITY_TYPES = ('troop', 'building', 'projectile', 'area')
# Per-entity columns: state row, native id, owner, object type id, x, y, hp, max hp,
# entity type, deploy ms, attack cooldown ms, target id. -1 = unknown/none.
ENTITY_COLUMNS = ('state', 'id', 'owner', 'object', 'x', 'y', 'hp', 'max_hp', 'type', 'deploy_ms',
                  'cooldown_ms', 'target')
# Human commands: execute tick, owner, card (-1 = ability), x, y, executed by native (0/1).
ACTION_COLUMNS = ('tick', 'owner', 'card', 'x', 'y', 'executed')


def _int(value):
    return -1 if value is None else value


def encode(result):
    """One rebuilt replay -> dict of numpy arrays (all int32/int64)."""
    states = result['states']
    T = len(states)
    tick = np.array([s.tick for s in states], np.int32)
    elixir = np.array([[p.elixir for p in s.players] for s in states], np.int32)
    crowns = np.array([s.crowns for s in states], np.int32)
    hand = np.full((T, 2, 4, 2), -1, np.int32)  # (card, cost) by hand slot
    cycle = np.full((T, 2, 4), -1, np.int32)
    entities, plays = [], []
    for t, s in enumerate(states):
        for o, p in enumerate(s.players):
            for c in p.hand:
                hand[t, o, c.slot] = (c.card, c.cost)
            cycle[t, o, :len(p.cycle)] = p.cycle[:4]
        for e in s.entities:
            entities.append((t, e.id, e.owner, e.card, e.x, e.y, _int(e.hp), _int(e.max_hp),
                             ENTITY_TYPES.index(e.entity_type), _int(e.deployment_remaining_ms),
                             _int(e.attack_cooldown_ms), _int(e.target_id)))
        plays.extend((t, p.tick, p.owner, p.card) for p in s.plays)
    actions = [(c.tick, c.owner, -1 if c.card is None else c.card, c.x, c.y, r.get('state') == 'succeeded')
               for c, r in zip(result['commands'], result['receipts'])]
    final = states[-1]
    return {
        'tick': tick, 'elixir': elixir, 'crowns': crowns, 'hand': hand, 'cycle': cycle,
        'entities': np.array(entities, np.int32).reshape(-1, len(ENTITY_COLUMNS)),
        'plays': np.array(plays, np.int32).reshape(-1, 4),
        'actions': np.array(actions, np.int32).reshape(-1, len(ACTION_COLUMNS)),
        # Per-replay scalars: native result (-1 draw/unknown), final crowns, fidelity flags.
        'meta': np.array([_int(final.result), *final.crowns, result['winner_matches'],
                          result['crowns_matches']], np.int32)[None],
    }


def write_shard(path, tags, replays):
    arrays = {'tags': np.array(tags)}
    for key in replays[0]:
        parts = [r[key] for r in replays]
        arrays[key] = np.concatenate(parts)
        arrays[key + '_off'] = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


SHARD_REPLAYS = 1000


def rebuild_with_recovery(engine, payload, recoveries=2):
    """An OSError means the engine is gone (emulator crash, socket timeout), not a bad replay:
    restart the emulator and engine, then retry. Raises if recovery keeps failing."""
    for attempt in range(recoveries + 1):
        try:
            return rebuild(engine, payload)
        except OSError as error:
            engine.close()
            if attempt == recoveries:
                raise
            print(f'engine lost ({type(error).__name__}: {error}); restarting it', flush=True)
            subprocess.run([sys.executable, '-m', 'native_engine.emulator', 'start'], check=True, capture_output=True)


def build_shard(engine, rows, out):
    tags, replays, failures = [], [], {}
    started = time.perf_counter()
    for row in rows:
        try:
            result = rebuild_with_recovery(engine, json.loads(row['payload_json']))
        except (OSError, subprocess.CalledProcessError):
            raise  # engine down: stop without writing a shard that silently misses replays
        except Exception as error:  # one bad replay must not stop the part; record why
            key = f'{type(error).__name__}: {str(error)[:80]}'
            failures[key] = failures.get(key, 0) + 1
            engine.close()  # drop a possibly half-read connection
            continue
        tags.append(result['tag'])
        replays.append(encode(result))
    if not replays:
        raise RuntimeError(f'no replay of {out.name} could be rebuilt: {failures}')
    write_shard(out, tags, replays)
    exact = sum(int(r['meta'][0, 3] and r['meta'][0, 4]) for r in replays)
    return {'shard': out.name, 'rows': len(rows), 'rebuilt': len(replays), 'exact': exact,
            'failures': failures, 'seconds': round(time.perf_counter() - started, 1)}


def firstlight():
    """FirstLight's converter. ponytail: borrowed (~2k verified lines), not ported."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'firstlight-cr'))
    from native_runner import contracts, match_factory, royaleapi_replay
    return contracts, match_factory, royaleapi_replay


DECISION_TICKS = 5


@dataclass(frozen=True)
class Command:
    tick: int  # First state tick that includes the command.
    owner: int
    card: int | None  # None for an ability activation.
    x: int = 0
    y: int = 0
    hints: tuple[str, ...] = ()


def prepare(payload):
    return firstlight()[2].prepare_collected_replay(payload)


def match_dict(replay, decks, seed):
    """Native match inputs; mirrors FirstLight's replay_viewer._match_config_from_replay.
    `decks` are per-owner card orders (the deal depends on the order)."""
    c, t = replay.episode_config, replay.episode_config.tags
    forms = [dict(zip(d, t[f'deck{o}_form_availability'])) for o, d in enumerate((c.deck0, c.deck1))]
    return firstlight()[1].MatchConfig(
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
    converter = firstlight()[2]
    replay = prepared.replay
    tags = replay.episode_config.tags
    plays = converter._replay_card_plays(replay)
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
                    want[o] = converter._special_deal_target(
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
    ActionKind = firstlight()[0].ActionKind
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
    prepared = prepare(payload)
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


def check(parquet, count, port):
    """Rebuild the first `count` replays of a part and report how faithfully they replay."""
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet, columns=['payload_json']).slice(0, count).to_pylist()
    engine = Engine(port=port)
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


def deck_pool(replays, out, per_part):
    """Deck pairs of real games as played: each card with its evolution flag ('el'), the level cap, king tower and
    experience levels. il.arena / il.rl read it (copy it to the training node)."""
    import pyarrow.parquet as pq
    pool = []
    for part in sorted(replays.glob('part-*.parquet')):
        for row in pq.read_table(part, columns=['payload_json']).slice(0, per_part).to_pylist():
            try:
                replay = prepare(json.loads(row['payload_json'])).replay
            except Exception:  # ponytail: a replay the converter rejects is simply skipped
                continue
            c = replay.episode_config
            b = match_dict(replay, (c.deck0, c.deck1), c.seed)['battle']
            pool.append({'decks': [b['deck0']['sp'], b['deck1']['sp']], 'lvlcap': b['lvlcap'],
                         'cardlvlmin': b['cardlvlmin'], 'kt': [b['hbd'][0]['kt'], b['hbd'][1]['kt']],
                         'exp': [b['avatar0']['expLevel'], b['avatar1']['expLevel']]})
        print(part.name, len(pool), flush=True)
    out.write_text(json.dumps(pool, separators=(',', ':')))
    print('wrote', len(pool), 'deck pairs to', out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replays', type=Path, help='directory of part-*.parquet (--check: one part)')
    parser.add_argument('out', type=Path)
    parser.add_argument('--parts', type=int, nargs='*', help='part numbers (default: all)')
    parser.add_argument('--limit', type=int, help='replays per part, for smoke tests')
    parser.add_argument('--port', type=int, default=26789)
    parser.add_argument('--deck-pool', type=int, metavar='PER_PART', help='write the deck pool to OUT instead')
    parser.add_argument('--check', type=int, metavar='COUNT', help='only report the fidelity of COUNT replays')
    args = parser.parse_args()
    if args.deck_pool:
        return deck_pool(args.replays, args.out, args.deck_pool)
    if args.check:
        return check(args.replays, args.check, args.port)
    args.out.mkdir(parents=True, exist_ok=True)
    parts = sorted(args.replays.glob('part-*.parquet'))
    if args.parts is not None:
        parts = [p for p in parts if int(p.stem.split('-')[1]) in args.parts]
    import pyarrow.parquet as pq
    engine = Engine(port=args.port)
    for parquet in parts:
        rows = None
        for k in range(-(-pq.ParquetFile(parquet).metadata.num_rows // SHARD_REPLAYS)):
            out = args.out / f'{parquet.stem}-{k}.npz'
            if out.exists():
                continue
            if rows is None:
                rows = pq.read_table(parquet, columns=['payload_json']).to_pylist()
            chunk = rows[k * SHARD_REPLAYS:(k + 1) * SHARD_REPLAYS][:args.limit]
            if not chunk:
                break
            report = build_shard(engine, chunk, out)
            print(json.dumps(report), flush=True)
            with open(args.out / 'build_log.jsonl', 'a') as log:
                log.write(json.dumps(report) + '\n')
            if args.limit:
                break


if __name__ == '__main__':
    main()
