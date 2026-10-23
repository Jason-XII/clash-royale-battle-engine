"""Collected 1v1 replays -> native engine playback -> one compressed archive per game (both players' inputs and labels).

    python convert.py --assignment ~/oracle/assignment.jsonl --replays ~/cr-engine/data/IL_Replay/replays \\
        --engine d86 ~/engine-86/engine ~/engine-86/data/assets --engine d29 ~/engine-29/engine ~/engine-29/data/assets \\
        --clapha-engine ~/clapha-engine --out ~/il-data/next --workers 23

Rerunning skips the games already written. manifest.json (what train.py reads) is written once every assigned game has
its archive: kept, or rejected with a reason (no exact replay, or the engine crashed on it twice). Needs Linux, the
clapha-engine native adapter and pyarrow; not torch.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from cr import observation as C

# ---- replay -> native match and commands --------------------------------------------------------------------------------
CARDS = json.loads((C.ASSETS / 'replay_cards.json').read_text())
FIRST_LEVEL = {'Common': 1, 'Rare': 3, 'Epic': 6, 'Legendary': 9, 'Champion': 11}
TOWER_FIRST_LEVEL = {159000000: 1, 159000001: 6, 159000002: 9, 159000004: 9}
DEAL = (((2, 0, 4, 7), (6, 1, 3, 5)), ((7, 1, 4, 3), (6, 0, 2, 5)))
SEED = 1784463263
ALIASES = defaultdict(set)
for _card, _info in CARDS['cards'].items():
    for _alias in _info['aliases']:
        ALIASES[_alias].add(int(_card))


def base_key(key):
    return re.sub(r'-(?:ev\d+|hero)$', '', str(key or '').strip(), flags=re.I).casefold()


def alias(value):
    return re.sub(r'[^a-z0-9]', '', str(value or '').casefold())


def card_id(key, name=None):
    key = base_key(key)
    if key in CARDS['public_ids']:
        return int(CARDS['public_ids'][key])
    probes = (alias(key), alias(name))
    for probe in probes:
        matches = ALIASES.get(probe, ())
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise ValueError(f'ambiguous native card: {key}')
    alternatives = {p + 's' for p in probes if p} | {p[:-1] for p in probes if p.endswith('s')}
    matches = set().union(*(ALIASES.get(p, ()) for p in alternatives))
    if len(matches) != 1:
        raise ValueError(f'unsupported replay card: {key}')
    return next(iter(matches))


def deck(player):
    if not isinstance(player.get('deck'), list) or len(player['deck']) != 8:
        raise ValueError('replay requires a complete eight-card deck')
    result = []
    for item in player['deck']:
        key = str(item.get('card_key') or '')
        if not key:
            raise ValueError('missing replay card key')
        cid = card_id(key, item.get('name'))
        form = int(bool(re.search(r'-ev\d+$', key, re.I)))
        form |= 2 if key.casefold().endswith('-hero') or str(cid) in CARDS['champions'] else 0
        level = item.get('level')
        level = int(level) if isinstance(level, (int, float)) and not isinstance(level, bool) else None
        result.append(dict(id=cid, key=key, form=form, level=level, omit=CARDS['cards'][str(cid)]['omit']))
    if len({c['id'] for c in result}) != 8:
        raise ValueError('duplicate native cards in replay deck')
    return result


def hand_slots(opening, queue, plays):
    if len(set(opening)) != 4 or len(queue) != 4 or len(set(queue)) != 4:
        return None
    hand, queue, slots = {card: i for i, card in enumerate(opening)}, list(queue), []
    for card in plays:
        if card not in hand:
            return None
        slot = hand.pop(card)
        slots.append(slot)
        hand[queue.pop(0)] = slot
        queue.append(card)
    return tuple(slots)


def select_deal(ids, plays, tag, owner, omitted=(), fixed_queue=None, salt=None):
    """Keep the recorded deterministic random search order, including omitted cards."""
    omitted, fixed = frozenset(omitted), dict(fixed_queue or {})
    if len(ids) != 8 or len(set(ids)) != 8 or not omitted.issubset(ids):
        raise ValueError('invalid deck for initial deal')
    if any(i not in range(4) for i in fixed) or len(set(fixed.values())) != len(fixed):
        raise ValueError('invalid fixed draw queue')
    if not set(fixed.values()).issubset(ids):
        raise ValueError('fixed draw queue contains cards outside deck')
    salt = salt or ('native-deal-v4-random-single-omit' if omitted else 'native-deal-v4-random-single')
    digest = hashlib.sha256(f'{tag}|owner={owner}|{salt}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], 'big'))
    eligible = tuple(card for card in ids if card not in omitted)
    first = plays[0] if plays else None
    if first is not None and first not in eligible:
        return None
    openings = (list(itertools.combinations(eligible, 4)) if first is None else
                [(first, *rest) for rest in itertools.combinations(tuple(c for c in eligible if c != first), 3)])
    rng.shuffle(openings)
    for raw in openings:
        opening = list(raw)
        rng.shuffle(opening)
        remaining = tuple(c for c in ids if c not in opening)
        if any(c not in remaining for c in fixed.values()):
            continue
        free_positions = tuple(i for i in range(4) if i not in fixed)
        orders = list(itertools.permutations(tuple(c for c in remaining if c not in fixed.values())))
        rng.shuffle(orders)
        for order in orders:
            queue = [fixed.get(i) for i in range(4)]
            for i, c in zip(free_positions, order, strict=True):
                queue[i] = c
            if hand_slots(opening, queue, plays) is not None:
                return tuple(opening), tuple(queue)
    return None


def ability_sources(event, cards):
    candidates = event.get('ability_source_candidates', ())
    keys = {base_key(k) for k in candidates if isinstance(k, str) and k.strip()} if isinstance(candidates, list) else set()
    explicit = [c for c in cards if base_key(c['key']) in keys]
    if explicit and event.get('ability_source_authoritative') is True:
        return explicit
    inferred = [c for c in cards if c['key'].casefold().endswith('-hero') or str(c['id']) in CARDS['champions']]
    return inferred or explicit


def prepare(payload):
    """Normalize exactly the fields needed by native playback and IL observations."""
    source = payload.get('source') or {}
    tag = str(source.get('replay_tag') or '')
    if not (re.fullmatch(r'[0289PYLQGRJCUV]+', tag, re.I) or
            (source.get('anonymized') is True and re.fullmatch(r'[0-9a-f]{32}', tag))):
        raise ValueError('invalid or missing replay tag')
    battle, events = payload['battle'], payload.get('events', [])
    if not isinstance(events, list):
        raise ValueError('replay events must be an array')
    flags = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('replay event must be an object')
        fields = event.get('source_fields') or {}
        if 'data_i' in fields:
            flag = fields['data_i']
            if isinstance(flag, bool) or not isinstance(flag, int) or flag not in (0, 1):
                raise ValueError('invalid true-side data_i marker')
            flags.add(flag)
    if len(flags) != 1:
        raise ValueError('missing or mixed true-side data_i markers')
    flag = flags.pop()
    side_owner = {'team': 1 - flag, 'opponent': flag}
    players, decks = {}, {}
    for side, owner in side_owner.items():
        rows = battle[side]['players']
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError('only one player per side is supported')
        players[owner] = rows[0]
        decks[owner] = deck(rows[0])
    parsed, plays = [], [[], []]
    for i, event in enumerate(events):
        owner = side_owner[event['side']]
        tick = int(event.get('replay_tick_20hz', -1))
        if tick < 1:
            raise ValueError('replay event tick must be positive')
        command = dict(tick=tick + 1, owner=owner)
        if event['kind'] == 'play_card':
            cid = card_id(event.get('card_key'))
            if cid not in {c['id'] for c in decks[owner]}:
                raise ValueError('replay action card is outside its deck')
            coordinates = event['coordinates']
            if not coordinates.get('inside_18x32_arena'):
                raise ValueError('replay action has invalid arena coordinates')
            grid, offset = coordinates['grid_cell_floor'], coordinates.get('subcell_offset_from_floor_center') or {}
            x, y = int(grid['x']), int(grid['y'])
            if not (0 <= x < 18 and 0 <= y < 32):
                raise ValueError('replay target is outside arena')
            dx, dy = float(offset.get('x', 0)), float(offset.get('y', 0))
            if flag:
                x, dx = 17 - x, -dx
            else:
                y, dy = 31 - y, -dy
            command.update(kind='card', card_id=cid, x=min(max(x * 1000 + 500 + int(round(dx * 1000)), 0), 17999),
                           y=min(max(y * 1000 + 500 + int(round(dy * 1000)), 0), 31999))
            plays[owner].append(cid)
        elif event['kind'] == 'activate_ability':
            sources = ability_sources(event, decks[owner])
            hints = []
            for card in sources:
                key = base_key(card['key'])
                for binding in (CARDS['heroes'].get(key) if card['form'] & 2 else None,
                                CARDS['champions'].get(str(card['id']))):
                    if binding and alias(binding[1]) not in hints:
                        hints.append(alias(binding[1]))
                for value in (CARDS['cards'][str(card['id'])]['name'], key):
                    if alias(value) and alias(value) not in hints:
                        hints.append(alias(value))
            for hint in tuple(hints):
                if hint.endswith('s') and len(hint) > 3 and hint[:-1] not in hints:
                    hints.append(hint[:-1])
            command.update(kind='ability', hints=tuple(hints), source_keys=tuple(c['key'] for c in sources))
        else:
            raise ValueError(f'unsupported replay event: {event["kind"]}')
        parsed.append((command['tick'], int(event.get('source_index', i)), i, command))
    openings, queues, configured, forms, omitted = [], [], [], [], []
    for owner in (0, 1):
        cards = decks[owner]
        ids = tuple(c['id'] for c in cards)
        omit = tuple(c['id'] for c in cards if c['omit'])
        chosen = select_deal(ids, plays[owner], tag, owner, omit)
        if chosen is None:
            raise ValueError(f'replay plays are incompatible with owner {owner} deck')
        opening, queue = chosen
        slots = iter(hand_slots(opening, queue, plays[owner]))
        for _, _, _, command in parsed:
            if command['owner'] == owner and command['kind'] == 'card':
                command['hand_slot'] = next(slots)
        arranged = [None] * 8
        for card, slot in zip((*opening, *queue), (*DEAL[owner][0], *DEAL[owner][1]), strict=True):
            arranged[slot] = card
        by_id = {c['id']: c for c in cards}
        openings.append(opening); queues.append(queue); configured.append(arranged)
        forms.append([by_id[c]['form'] for c in arranged]); omitted.append(omit)
    commands = [c for _, _, _, c in sorted(parsed, key=lambda row: row[:3])]
    last_source = max((c['tick'] - 1 for c in commands), default=0)
    duration = (payload.get('replay') or {}).get('duration') or {}
    times = [int(round(duration[k] * 20)) for k in ('timeline_seconds', 'display_seconds')
             if isinstance(duration.get(k), (int, float)) and not isinstance(duration[k], bool) and duration[k] >= 0]
    source_end = max([last_source, *times])
    end = max(source_end, max((c['tick'] for c in commands), default=0))
    levels = [c['level'] for cards in decks.values() for c in cards if c['level'] is not None]
    for p in players.values():
        level = (p.get('tower_card') or {}).get('level')
        if isinstance(level, (int, float)):
            levels.append(int(level))
    counts = Counter(levels)
    level = min(max(sorted(counts, key=lambda x: (-counts[x], -x))[0] if counts else 11, 1), 16)
    match = C.standard_match(SEED)
    native = match['battle']
    mode = alias(battle.get('game_mode'))
    native.update(gamemode=72000009 if mode in ('classicchallenge', 'grandchallenge') else 72000006,
                  arena=54000036 if mode in ('classicchallenge', 'grandchallenge') else 54000001,
                  location=15000013 if mode in ('classicchallenge', 'grandchallenge') else 15000199,
                  lvlcap=16, cardlvlmin=1)
    displayed_levels = {}
    for owner in (0, 1):
        p, cards = players[owner], {c['id']: c for c in decks[owner]}
        tower = p.get('tower_card') or {}
        tower_key = str(tower.get('card_key') or '').strip().casefold()
        tower_key = {'princess-tower': 'tower-princess', 'chef-tower': 'royal-chef'}.get(tower_key, tower_key)
        if tower_key and tower_key not in CARDS['towers']:
            raise ValueError(f'unsupported tower troop: {tower_key}')
        troop = int(CARDS['towers'].get(tower_key, 159000000))
        entries = []
        for cid, form in zip(configured[owner], forms[owner], strict=True):
            entry = {'d': cid}
            if form:
                entry['el'] = form
            first = FIRST_LEVEL.get(C.RARITY.get(str(cid)))
            if cards[cid]['level'] and first:
                entry['l'] = max(0, cards[cid]['level'] - first)
            entries.append(entry)
        support = native[f'deck{owner}']['sc'][0]
        support['d'] = troop
        if tower.get('level'):
            support['l'] = max(0, int(tower['level']) - TOWER_FIRST_LEVEL[troop])
        native[f'deck{owner}']['sp'] = entries
        avatar = native[f'avatar{owner}']
        avatar.update(name=str(p.get('name') or p.get('tag') or f'RoyaleAPI-{"Top" if owner == 0 else "Bottom"}').strip()[:48]
                      or f'RoyaleAPI-{"Top" if owner == 0 else "Bottom"}', expLevel=level)
        native['hbd'][owner]['kt'] = level
        displayed_levels[owner] = {cid: c['level'] for cid, c in cards.items() if c['level']}
        displayed_levels[f'tower{owner}'] = int(tower['level']) if tower.get('level') else None
    match.update(rndSeed=SEED, endTick=end)
    result = str(battle.get('result') or '').strip().casefold()
    winner_side = 'team' if result in ('win', 'victory') else 'opponent' if result in ('defeat', 'loss', 'lose') else None
    terminal = dict(winner=side_owner[winner_side] if winner_side else None, native_tick=end, source_end_tick=source_end,
                    last_source_event_tick=last_source, last_native_event_tick=max((c['tick'] for c in commands), default=0),
                    source_result=str(battle.get('result') or ''),
                    players=tuple({'owner': owner, 'crowns': int(battle[side]['crowns'])}
                                  for owner in (0, 1) for side in side_owner if side_owner[side] == owner))
    return dict(tag=tag, match=match, commands=commands, terminal=terminal, opening=openings, queue=queues,
                omitted=omitted, levels=displayed_levels, episode=dict(source_team_owner=side_owner['team'],
                deck0_form_availability=tuple(forms[0]), deck1_form_availability=tuple(forms[1])))


def calibrate(game, native):
    """Verify actual native 4+4 dealing; remap only when the observed trace requires it."""
    match = game['match']
    seen, tested = set(), [set(), set()]
    plays = [[c['card_id'] for c in game['commands'] if c['owner'] == owner and c['kind'] == 'card'] for owner in (0, 1)]
    for attempt in range(10):
        decks = [tuple(c['d'] for c in match['battle'][f'deck{o}']['sp']) for o in (0, 1)]
        signature = tuple(decks)
        if signature in seen:
            raise ValueError('native deal calibration repeated a deck configuration')
        seen.add(signature)
        observation = native.create_match(match)
        observed, slots = [], []
        for owner in (0, 1):
            p = next(p for p in observation['players'] if p['owner'] == owner)
            hand = sorted(p['hand'], key=lambda c: c['handIndex'])
            cycle = sorted(p['cycle'], key=lambda c: c['cycleIndex'])
            hs, qs = tuple(c['deckSlot'] for c in hand), tuple(c['deckSlot'] for c in cycle)
            if len(hand) != 4 or len(cycle) != 4 or set(hs + qs) != set(range(8)):
                raise ValueError('native opening hand and queue must cover eight distinct deck slots')
            opening, queue = tuple(c['cardId'] for c in hand), tuple(c['cardId'] for c in cycle)
            observed.append((hs, qs, opening, queue))
            slots.append(hand_slots(opening, queue, plays[owner]))
            if game['omitted'][owner]:
                tested[owner].add(tuple(decks[owner].index(c) for c in game['omitted'][owner]))
        if all(s is not None for s in slots):
            for owner in (0, 1):
                stream = iter(slots[owner])
                for command in game['commands']:
                    if command['owner'] == owner and command['kind'] == 'card':
                        command['hand_slot'] = next(stream)
                game['opening'][owner], game['queue'][owner] = observed[owner][2:]
            return game
        for owner in (0, 1):
            if slots[owner] is not None:
                continue
            entries = match['battle'][f'deck{owner}']['sp']
            by_id = {e['d']: e for e in entries}
            hs, qs, opening, queue = observed[owner]
            omitted = game['omitted'][owner]
            target = (game['opening'][owner], game['queue'][owner])
            if omitted:
                if set(omitted) & set(opening):
                    raise ValueError('native opening hand contains an omitted card')
                fixed = {i: c for i, c in enumerate(queue) if c in omitted}
                if set(fixed.values()) != set(omitted):
                    raise ValueError('native queue does not contain all omitted cards')
                constraint = ','.join(f'{i}:{c}' for i, c in sorted(fixed.items()))
                target = select_deal(decks[owner], plays[owner], game['tag'], owner, omitted, fixed,
                                     salt=f'native-deal-v4-random-single-omit-queue={constraint}')
                if target is None:
                    text = (f'{game["tag"]}|owner={owner}|native-deal-v3-multi-omit-slot-probe|'
                            f'deck={",".join(map(str, decks[owner]))}|omitted={",".join(map(str, omitted))}')
                    positions = list(itertools.permutations(range(8), len(omitted)))
                    random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')).shuffle(positions)
                    positions = next((p for p in positions if p not in tested[owner]), None)
                    if positions is None:
                        raise ValueError('no compatible native omitted-card layout')
                    fixed_slots = dict(zip(positions, omitted, strict=True))
                    others = iter(c for c in decks[owner] if c not in omitted)
                    arranged = [fixed_slots[i] if i in fixed_slots else next(others) for i in range(8)]
                else:
                    arranged = [None] * 8
            else:
                arranged = [None] * 8
            if target is not None:
                for cid, slot in zip((*target[0], *target[1]), (*hs, *qs), strict=True):
                    arranged[slot] = cid
                if any(arranged.index(c) != decks[owner].index(c) for c in omitted):
                    raise ValueError('calibration moved an omitted card')
            match['battle'][f'deck{owner}']['sp'] = [copy.copy(by_id[c]) for c in arranged]
            game['episode'][f'deck{owner}_form_availability'] = tuple(by_id[c].get('el', 0) for c in arranged)
    raise ValueError('native deal calibration failed after ten configurations')


def queue_ability(native, command):
    ids = set()
    for key in command['source_keys']:
        cid = card_id(key)
        ids.add(cid)
        if str(cid) in CARDS['hero_units']:
            ids.add(int(CARDS['hero_units'][str(cid)]))
    objects = [o for o in native.observe()['objects'] if o['owner'] == command['owner'] and
               (not ids or o.get('cardId') in ids) and o.get('nativeObjectId')]
    for obj in sorted(objects, key=lambda o: -o['nativeObjectId']):
        try:
            native.queue_ability_action_at(SimpleNamespace(owner=command['owner'], object_index=-2,
                                           secondary_index=obj['nativeObjectId']), execute_in_ticks=1)
            return True
        except RuntimeError:
            continue
    return False


def fidelity(final, terminal, payload, team_owner):
    alive = {}
    for obj in final.get('objects', []):
        owner = obj.get('owner')
        for name, x, y in (('king', 9000, 3000), ('left', 3500, 6500), ('right', 14500, 6500)):
            y = 32000 - y if owner == 1 else y
            if abs(obj['x'] - x) <= 600 and abs(obj['y'] - y) <= 600:
                alive[owner, name] = int(obj.get('hp') or 0)
    expected = {p['owner']: p['crowns'] for p in terminal['players']}
    crowns = {o: int(c) for o, c in enumerate(final.get('crownsRaw') or [])}
    towers, error = {}, 0
    for side, owner in (('team', team_owner), ('opponent', 1 - team_owner)):
        hp = payload['battle'][side]['players'][0].get('final_tower_hitpoints') or {}
        rec = [hp.get(k) for k in ('king', 'princess_left', 'princess_right')]
        sim = [alive.get((owner, k), 0) for k in ('king', 'left', 'right')]
        straight = sum(abs(a - b) for a, b in zip(sim, rec) if b is not None)
        swapped = abs(sim[0] - (rec[0] or 0)) + abs(sim[1] - (rec[2] or 0)) + abs(sim[2] - (rec[1] or 0))
        error += sim[0] if rec[0] == 0 else min(straight, swapped)
        towers[owner] = {'simulated': sim, 'recorded': rec}
    return dict(winner_match=final.get('winner') == terminal['winner'], crowns_match=crowns == expected,
                crowns=(crowns, expected), tower_hp_error=error, towers=towers)


class Recorder:
    """The native env, recording each decision's raw record (every 5 ticks) while the replay's commands drive it."""

    def __init__(self, env):
        self._env = env
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def run_lean(self, until, every=5):
        env, r = self._env, self._env.reader
        frames = []
        while env.tick() < until and not env.stalled:
            env._advance()
            tick = env.tick()
            _crowns, ended, _winner = env._result()
            if tick % every == 0 or ended:
                lean = r.frame(env.battle)
                frames.append(lean)
                # frame() already read the objects; share that snapshot instead of walking native memory twice.
                self.decisions.append({'tick': tick, 'lean': {k: v for k, v in lean.items() if k != 'objects'},
                                       'objects': lean['objects'], 'units': r.unit_states(env.battle)})
            if ended:
                return frames, tick, True
        if env.stalled:
            frames.append(r.frame(env.battle))
            return frames, env.tick(), True
        return frames, env.tick(), False


def play_replay(native, payload):
    """Play a replay's commands in the native engine (a Recorder) -> its game, held-play log and fidelity."""
    game = calibrate(prepare(payload), native)
    native.create_match(game['match'])
    tick, ended_at, failures = 0, None, []

    def advance(to):
        nonlocal tick, ended_at
        if to > tick and ended_at is None:
            _, tick, ended = native.run_lean(to, 5)
            if ended and tick < to:
                ended_at = tick

    for command in game['commands']:
        advance(command['tick'] - 1)
        if ended_at is not None:
            break
        try:
            if command['kind'] == 'card':
                native.queue_card_at(command['owner'], command['card_id'], command['x'], command['y'],
                                     execute_tick=command['tick'])
            elif not queue_ability(native, command):
                failures.append((command['tick'], 'ability', 'no ready source unit accepted it'))
        except Exception as error:
            failures.append((command['tick'], command['kind'], str(error)[:120]))
    advance(game['terminal']['native_tick'])
    return dict(game=game, holds=list(native.hold_log), ended_at=ended_at, failures=failures,
                fidelity=fidelity(native.observe(), game['terminal'], payload, game['episode']['source_team_owner']))


# ---- a played game -> inputs and labels ---------------------------------------------------------------------------------
def own_starts(deck_ids, plays):
    """Every starting (hand as a frozenset, queue as a tuple) of an 8-card deck that the play sequence fits. At most
    70 x 24 = 1,680 to try. Cards never played can sit anywhere among the slots the plays leave open."""
    deck_ids = tuple(deck_ids)
    if len(set(deck_ids)) != len(deck_ids) or len(deck_ids) != 8:
        raise ValueError('a deck is 8 different cards')
    unknown = [c for c in plays if c not in deck_ids]
    if unknown:
        raise ValueError(f'played cards not in the deck: {sorted(set(unknown))}')
    out = []
    for hand in itertools.combinations(deck_ids, 4):
        for queue in itertools.permutations([c for c in deck_ids if c not in hand]):
            if hand_slots(hand, queue, plays) is not None:
                out.append((frozenset(hand), tuple(queue)))
    return out


def game_features(game, owner, vocab, legal=None, compact=True):
    """All decisions of one played game for `owner` -> (inputs: dict of [T, ...] arrays, labels: dict of [T] arrays)."""
    header, decisions = game['header'], game['decisions']
    ticks = np.array([d['tick'] for d in decisions])
    landed = [(c.get('executed_tick') or c['tick'], c) for c in header['commands']
              if c['kind'] == 'card' and c.get('executed_tick', 0) is not None]
    own_plays = [cid for _, cid in sorted((t, c['card_id']) for t, c in landed if c['owner'] == owner)]
    first = [p for p in decisions[0]['lean']['state']['players'] if p['owner'] == owner][0]
    forms = (header.get('episode') or {}).get(f'deck{1 - owner}_form_availability')
    view = C.PlayerView(owner, vocab, C.opponent_heroes(decisions[0], forms, owner), legal,
                        own_starts([c['cardId'] for c in first['deck']], own_plays))
    if compact:
        view.state['object_slots'] = min(C.MAX_OBJECTS, max(1, max(len(d['objects']) for d in decisions)))
    rows = [view.step(d, landed) for d in decisions]
    inputs = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    labels = _labels(header, decisions, ticks, owner, vocab, inputs)
    if 'legal' in inputs:
        _snap_to_legal(inputs, labels)
    return inputs, labels


def _snap_to_legal(inputs, labels):
    """A play's tile where the mask says the card cannot go: the nearest legal tile within one tile (buildings are recorded
    by their centre and anchored half a tile off: 8 of 3,829 human plays, 2026-10-12); none -> label off."""
    for i in np.flatnonzero(labels['action_kind'] == 1):
        slots = np.flatnonzero(inputs['hand_card'][i] == labels['action_card'][i])
        if not len(slots):
            continue
        mask = inputs['legal'][i][slots[0]]
        r, c = divmod(int(labels['action_cell'][i]), C.W)
        if mask[r, c]:
            continue
        near = [(abs(dr) + abs(dc), r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if 0 <= r + dr < C.H and 0 <= c + dc < C.W and mask[r + dr, c + dc]]
        if near:
            _, rr, cc = min(near)
            labels['action_cell'][i] = rr * C.W + cc
        else:
            labels['label_ok'][i] = False


def _labels(header, decisions, ticks, owner, vocab, inputs):
    """The human's action at each decision: a play landing at tick c (the replay's tick) is labelled at the last
    decision before c - 22 (the board at the tap). Ability presses the same way, at their own tick."""
    T = len(decisions)
    out = {'action_kind': np.zeros(T, np.int64), 'action_card': np.zeros(T, np.int64),
           'action_cell': np.zeros(T, np.int64), 'action_controller': np.zeros(T, np.int64),
           'label_ok': np.ones(T, bool), 'value': np.zeros(T, np.float32)}
    winner = (header.get('terminal') or {}).get('winner')
    out['value'][:] = 0.0 if winner not in (0, 1) else (1.0 if winner == owner else -1.0)
    names = {}
    for a in [p for p in decisions[0]['lean']['players'] if p['owner'] == owner][0]['abilityRuntime']:
        names[a['controllerSlot'] - 1] = (a.get('actionDataName') or '').lower()
    for c in header['commands']:
        if c['owner'] != owner or (c['kind'] == 'card' and c.get('executed_tick', 0) is None):
            continue
        i = int(np.searchsorted(ticks, c['tick'] - C.DEPLOY_DELAY)) - 1 if c['kind'] == 'card' else \
            int(np.searchsorted(ticks, c['tick'])) - 1
        if i < 0 or i >= T or out['action_kind'][i]:
            continue
        if c['kind'] == 'card':
            card = vocab(c['card_id'])
            out['action_kind'][i], out['action_card'][i] = 1, card
            x, y = C.rotate(c['x'], c['y'], owner)
            out['action_cell'][i] = C.cell(x, y)
            if card not in inputs['hand_card'][i]:
                out['label_ok'][i] = False        # the card is not visible in the input (uncertain hand / refill)
        else:
            hints = [h.lower() for h in c.get('hints') or ()]
            slot = next((k for k, n in names.items() if any(h in n.replace('_', '') for h in hints)), 0)
            out['action_kind'][i], out['action_controller'][i] = 2, slot
    return out


def targets(x, y):
    """Labels as the model's heads use them: what (0..3), which (hand slot), tile (cell), the next play (how soon, which
    hand slot), and their masks."""
    T = len(y['action_kind'])
    what = np.full(T, C.WAIT, np.int64)
    which = np.zeros(T, np.int64)
    ok = y['label_ok'].copy()
    play = y['action_kind'] == 1
    for i in np.flatnonzero(play):
        slots = np.flatnonzero(x['hand_card'][i] == y['action_card'][i])
        if len(slots):
            what[i], which[i] = C.PLAY, slots[0]
        else:
            ok[i] = False
    ab = y['action_kind'] == 2
    what[ab] = C.ABILITY1 + y['action_controller'][ab]
    # which card: only where the whole hand is known. The replay's own hand is reconstructed from the game's plays, and
    # the card played next is always among the certain ones, so with a '?' in the hand "pick a card that is not '?'" would
    # be a hint from the future (live, the hand is always known). Found by the validation breakdown, 2026-10-12.
    hand_known = (x['hand_card'] != 1).all(-1)
    which_ok = ok & hand_known
    # the next play from each decision (a play is 0 decisions from its own decision); none left counts as far away
    plays = np.flatnonzero(play)
    k = np.searchsorted(plays, np.arange(T))
    nxt = plays[np.minimum(k, len(plays) - 1)] if len(plays) else np.zeros(T, np.int64)
    away = np.where(k < len(plays), nxt - np.arange(T), C.NEXT_EDGES[-1])
    match = x['hand_card'] == y['action_card'][nxt][:, None]
    return {'what': what, 'which': which, 'tile': y['action_cell'].astype(np.int64), 'ok': ok, 'which_ok': which_ok,
            'value': y['value'].astype(np.float32),
            'next_when': np.searchsorted(C.NEXT_EDGES, away, side='right').astype(np.int64),
            'next_which': match.argmax(1).astype(np.int64),
            'next_which_ok': (away < C.NEXT_EDGES[-1]) & match.any(1) & hand_known}


# ---- archives -----------------------------------------------------------------------------------------------------------
def held_out(tag, share=0.02):
    return int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def archive_path(root, tag):
    return Path(root) / (hashlib.sha256(tag.encode()).hexdigest() + '.npz')


def _pack(x, t):
    """One player's game -> arrays: objects only where present (with offsets), int32 ids, legal masks as bits."""
    present = x['obj_side'] > 0
    arrays = {'offsets': np.r_[0, present.sum(1).cumsum()].astype(np.int64)}
    for k, v in x.items():
        if k in C.OBJECT_KEYS:
            v = v[present]
        if k == 'legal':
            v = np.packbits(v.reshape(len(v), -1), axis=-1)
        elif v.dtype == np.int64:
            v = v.astype(np.int32)
        arrays['x_' + k] = v
    arrays.update({'t_' + k: v for k, v in t.items()})
    return arrays


def write_archive(path, game, vocab, legal, metadata):
    """Both player views (or none, for a rejected game) in one archive, renamed into place when complete."""
    metadata = dict(metadata, entries=[])
    tmp = path.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        def put(name, value):
            with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
        if metadata['kept']:
            for owner in (0, 1):
                x, y = game_features(game, owner, vocab, legal)
                t = targets(x, y)
                for name, value in _pack(x, t).items():
                    put(f'p{owner}_{name}', value)
                metadata['entries'].append(dict(file=path.name, tag=metadata['tag'], owner=owner,
                                                validation=held_out(metadata['tag']), decisions=len(t['what'])))
        put('_metadata', np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8))
    os.replace(tmp, path)


def read_metadata(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(z['_metadata'].tobytes())


def _jsonable(x):
    if isinstance(x, dict) or (hasattr(x, 'items') and hasattr(x, 'keys')):     # dicts and FrozenMapping
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


# ---- workers ------------------------------------------------------------------------------------------------------------
def worker(index, engine, jobs, out, clapha_engine):
    name, libdir, assets = engine
    sys.path.insert(0, str(Path(clapha_engine).expanduser() / 'crx'))
    log = open(out / f'log-{name}-{index}.txt', 'a')
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    import pyarrow.parquet as pq
    from env import CrxEnv
    time.sleep(index // 4 * 10)                 # start 4 engines at a time (data-table loading times out under load)
    for attempt in range(3):
        try:
            env = CrxEnv(libdir, out / 'native-work' / f'{name}-{index}-{attempt}', assets)
            break
        except RuntimeError as error:
            print(f'engine start failed ({error}), retrying', flush=True)
            time.sleep(15)
    else:
        raise SystemExit('the engine did not start')
    vocab, legal = C.Vocab.load(), C.cached_legal()
    index_file = open(out / f'index-{name}-{index}.jsonl', 'a')
    tables = {}
    for job in jobs:
        if job['path'] not in tables:
            tables = {job['path']: pq.read_table(job['path'], columns=['replay_tag', 'payload_json'])}
        payload = json.loads(tables[job['path']]['payload_json'][job['row']].as_py())
        if job.get('towers'):
            # a collected battle the oracle accepted with one tower-troop pair only (it did not keep them)
            for side, troop in zip(('team', 'opponent'), job['towers']):
                payload['battle'][side]['players'][0]['tower_card'] = {'card_key': troop, 'level': 16}
        entry = {'tag': job['tag'], 'part': job['part'], 'row': job['row'], 'engine': name,
                 **({'towers': job['towers']} if job.get('towers') else {})}
        started = time.perf_counter()
        (out / f'current-{name}-{index}.txt').write_text(job['tag'])  # a native crash kills the worker here
        game = None
        try:
            rec = Recorder(env)
            result = play_replay(rec, payload)
            fid = result['fidelity']
            entry.update(hp_error=int(fid['tower_hp_error']), winner=bool(fid['winner_match']),
                         crowns=bool(fid['crowns_match']), decisions=len(rec.decisions))
            if int(fid['tower_hp_error']) != 0 or not (fid['winner_match'] and fid['crowns_match']):
                entry.update(kept=False, why='did not reproduce on conversion')
            else:
                played = result['game']
                done = {(h[0], h[1], h[2]): h for h in result['holds']}
                for c in played['commands']:
                    if c['kind'] == 'card':
                        h = done.get((c['tick'], c['owner'], c['card_id']))
                        c['executed_tick'] = c['tick'] + h[3] if h and h[4] == 'played' else None
                dropped = [h for h in result['holds'] if h[4] == 'dropped']
                if dropped:
                    # the board after a wrong card is not the replay's: keep the decisions before it only
                    entry['cut_at'] = dropped[0][0]
                    rec.decisions = [d for d in rec.decisions if d['tick'] < dropped[0][0]]
                game = {'header': {'tag': job['tag'], 'commands': _jsonable(played['commands']),
                                   'terminal': _jsonable(played['terminal']), 'episode': _jsonable(played['episode'])},
                        'decisions': rec.decisions}
                entry['kept'] = bool(rec.decisions)
                if not rec.decisions:
                    entry['why'] = 'no decisions before the first dropped play'
            write_archive(archive_path(out, job['tag']), game, vocab, legal, entry)
        except Exception as error:  # noqa: BLE001
            entry.update(kept=False, why='{}: {}'.format(type(error).__name__, str(error)[:200]))
            write_archive(archive_path(out, job['tag']), None, vocab, legal, entry)
        entry['seconds'] = round(time.perf_counter() - started, 2)
        index_file.write(json.dumps(entry) + '\n')
        index_file.flush()
    os._exit(0)  # the engine's exit-time destructors crash; everything is flushed


def convert(args):
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    engines = {name: (name, str(Path(lib).expanduser()), str(Path(assets).expanduser())) for name, lib, assets in args.engine}
    parts = {}
    for directory in args.replays:
        for p in sorted(Path(directory).expanduser().glob('*.parquet')):
            parts.setdefault(p.name, str(p.resolve()))
    rows = [json.loads(line) for line in open(Path(args.assignment).expanduser())]
    selected = [dict(r, path=parts[r['part']]) for r in rows if r.get('engine') in engines and r['part'] in parts]
    if args.limit:
        selected = selected[:args.limit]
    if not selected or len({r['tag'] for r in selected}) != len(selected):
        raise SystemExit('no assigned replays found, or duplicate tags in the assignment')
    jobs = [r for r in selected if not archive_path(out, r['tag']).exists()]
    print(f'{len(selected)} games; {len(selected) - len(jobs)} already written; {len(jobs)} to convert', flush=True)

    ctx = mp.get_context('fork')
    running, crashes = {}, Counter()

    def start(name, index, share):
        (out / f'current-{name}-{index}.txt').unlink(missing_ok=True)
        process = ctx.Process(target=worker, args=(index, engines[name], share, out, args.clapha_engine))
        process.start()
        running[name, index] = (process, share)

    # one CPU budget over the engine builds, in proportion to their games
    by_engine = {name: sorted((r for r in jobs if r['engine'] == name), key=lambda r: (r['part'], r['row']))
                 for name in engines}
    by_engine = {k: v for k, v in by_engine.items() if v}
    for name, mine in by_engine.items():
        count = max(1, min(len(mine), args.workers * len(mine) // len(jobs)))
        for index in range(count):
            start(name, index, mine[index::count])
    try:
        while running:
            time.sleep(1)
            for (name, index), (process, share) in list(running.items()):
                if process.is_alive():
                    continue
                del running[name, index]
                if process.exitcode == 0:
                    continue
                marker = out / f'current-{name}-{index}.txt'
                if not marker.exists():
                    raise RuntimeError(f'{name} worker {index} failed before its first game; see its log')
                tag = marker.read_text().strip()
                crashes[tag] += 1
                if crashes[tag] >= 2:  # the native engine crashes on this replay: record it, do not drop it silently
                    job = next(r for r in share if r['tag'] == tag)
                    write_archive(archive_path(out, tag), None, None, None,
                                  dict(tag=tag, part=job['part'], row=job['row'], engine=name, kept=False,
                                       why='native engine crashed twice'))
                rest = [r for r in share if not archive_path(out, r['tag']).exists()]
                if rest:
                    start(name, index, rest)
    finally:
        for process, _ in running.values():
            process.terminate()
            process.join()

    entries, rejected, missing = [], [], []
    for r in selected:
        path = archive_path(out, r['tag'])
        if not path.exists():
            missing.append(r['tag'])
            continue
        meta = read_metadata(path)
        entries += meta['entries']
        if not meta['kept']:
            rejected.append(dict(tag=r['tag'], why=meta.get('why')))
    if missing:
        raise SystemExit(f'{len(missing)} games have no archive yet (first: {missing[:5]}); rerun to finish')
    manifest = dict(schema=dict(vocab=sorted(C.Vocab.load().index), legal=True), entries=entries, rejected=rejected)
    (out / 'manifest.tmp').write_text(json.dumps(manifest))
    (out / 'manifest.tmp').replace(out / 'manifest.json')
    print(f'{len(entries) // 2} games kept, {len(rejected)} rejected -> {out / "manifest.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assignment', required=True, help="the version oracle's assignment.jsonl (tag, part, row, engine)")
    parser.add_argument('--replays', nargs='+', required=True, help='directories of replay parquet shards')
    parser.add_argument('--engine', nargs=3, action='append', required=True, metavar=('NAME', 'LIBDIR', 'ASSETS'))
    parser.add_argument('--clapha-engine', required=True, help='the clapha-engine checkout (its crx/ adapter)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=23)
    parser.add_argument('--limit', type=int, default=0, help='convert only the first N assigned games')
    convert(parser.parse_args())


if __name__ == '__main__':
    main()
