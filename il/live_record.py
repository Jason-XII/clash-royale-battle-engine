"""Null's Royale battles read live, as clapha-il raw records: frames of Arron's live reader (clapha-console, its Nulls build:
~/clapha-console/nulls/README.md) -> the record ClaphaEngine.record() and clapha-il's convert.py give (lean frame, objects,
unit states), so a clapha-il model gets live what it got in imitation learning and self-play (il.policy).

    records = LiveRecords(tables)        # tables: clapha-engine crx/export_tables.py of the August build
    record = records(frame)              # {'tick', 'lean': {'state': {'players'}, 'players'}, 'objects', 'units'}
    plays = LivePlays(); landed = plays.update(frame)   # card plays as (tick, command), as il.policy.Commands gives

The unit states go through the console's own mapping (mac012/live_unit_states.py, CLAPHA_CONSOLE, default
~/clapha-console). Not read live yet (inputs stay empty): plays in flight before they land (the client's command queue).
"""
import os
from pathlib import Path
import sys

CONSOLE = Path(os.environ.get('CLAPHA_CONSOLE', Path.home() / 'clapha-console'))
# Plays an evolution needs, by card (evolutionRuntime's cycleRequired in the August build's converted replays: one value
# per card in every game). Static game data: the live reader reads only the progress.
CYCLE_REQUIRED = {
    26000000: 2, 26000001: 2, 26000004: 1, 26000007: 1, 26000008: 1, 26000010: 2, 26000011: 2, 26000012: 2, 26000013: 2,
    26000014: 2, 26000015: 2, 26000017: 1, 26000022: 1, 26000024: 1, 26000026: 2, 26000030: 2, 26000035: 2, 26000036: 2,
    26000037: 2, 26000040: 2, 26000043: 1, 26000044: 2, 26000045: 1, 26000047: 1, 26000049: 2, 26000050: 2, 26000055: 1,
    26000056: 2, 26000058: 2, 26000059: 2, 26000060: 1, 26000063: 1, 26000064: 2, 27000000: 2, 27000002: 2, 27000006: 2,
    27000010: 2, 27000012: 2, 27000013: 2, 28000004: 2, 28000008: 2, 28000017: 2}
EVOLUTION = 1  # deck_form_flags: 1 evolution, 2 hero


class LiveRecords:
    def __init__(self, tables):
        if str(CONSOLE / 'mac012') not in sys.path:
            sys.path.insert(0, str(CONSOLE / 'mac012'))
        from live_unit_states import LiveUnitStates
        self.unit_states = LiveUnitStates(Path(tables))

    def __call__(self, frame):
        entities = frame['entities']
        ids = {e['address']: (e['side'], e['category']) for e in entities}
        units = self.unit_states.rows(frame)
        players = sorted(frame['players'], key=lambda p: p['side'])
        return {'tick': frame['game_tick'],
                'lean': {'state': {'players': [_player(p) for p in players]}, 'players': [_lean(p) for p in players]},
                'objects': [_object(e, ids) for e in entities],
                'units': [units.get(e['address'], {'nativeObjectId': e['category'], 'kind': 'other'}) for e in entities]}


def _object(e, ids):
    """One objects() row (clapha-engine crx/observe.py): the attack's target, load and the deploy timer only with an attack
    component, as there."""
    attack = bool(e.get('has_attack'))
    target = ids.get(e.get('target'))
    return {'nativeObjectId': e['category'], 'owner': e['side'], 'cardId': e['card_id'],
            'dataGlobalId': (e.get('data_id') or 0) & 0xFFFFFFFF, 'x': e['x'], 'y': e['y'],
            'hp': e['hp'] if e['hp'] >= 0 else None, 'maxHp': e['max_hp'] if e['max_hp'] >= 0 else None,
            'targetEntityKey': [target[0], -2, target[1]] if attack and target else None,
            'phaseRuntime': {'attackValidated': attack, 'deployRemainingMs': e['deploy_remaining'] if attack else None,
                             'loadRemainingMs': e['atk_load'] if attack else None}}


def _player(p):
    """players_state()'s row: hand and cycle by deck slot, elixir, deck."""
    deck = p['deck_card_ids']
    return {'owner': p['side'], 'elixirRaw': p['elixir_raw'], 'elixir': p['elixir_raw'] / 10000.0,
            'hand': [{'handIndex': i, 'deckSlot': s, 'cardId': deck[s]} for i, s in enumerate(p['hand_deck_indices']) if s >= 0],
            'cycle': [{'cycleIndex': i, 'deckSlot': s, 'cardId': deck[s]} for i, s in enumerate(p['cycle_deck_indices']) if s >= 0],
            'deck': [{'deckSlot': i, 'cardId': c} for i, c in enumerate(deck)]}


def _lean(p):
    """players_lean()'s row: the hero / champion controllers and each deck slot's evolution progress."""
    abilities = [{'controllerSlot': a['controller_slot'], 'available': a['character_id'] != 0, 'buttonState': a['button'],
                  'remainingChargesRaw': a['charges'], 'remainingCooldownMs': a['cooldown_ms'],
                  'configuredCooldownMs': a['configured_ms'], 'championObjectIds': a['units'], 'actionDataName': ''}
                 for a in p['abilities']]
    evolutions = []
    for slot, (card, form, progress) in enumerate(zip(p['deck_card_ids'], p['deck_form_flags'], p['evo_progress'])):
        need = CYCLE_REQUIRED.get(card) if form == EVOLUTION else None
        evolutions.append({'deckSlot': slot, 'cardId': card, 'evolvable': need is not None, 'progress': progress,
                           'cycleRequired': need, 'cycleRemaining': max(0, need - progress) if need else None,
                           'ready': progress >= need if need else None})
    return {'owner': p['side'], 'abilityRuntime': abilities, 'evolutionRuntime': evolutions}


class LivePlays:
    """Card plays seen so far as (tick, {'owner', 'card_id'}): a card that left a hand between two frames was played (as
    ClaphaEngine detects them), at the tick of the frame that shows it gone."""

    def __init__(self):
        self.hands, self.landed = {}, []

    def update(self, frame):
        for p in frame['players']:
            deck = p['deck_card_ids']
            hand = {i: deck[s] for i, s in enumerate(p['hand_deck_indices']) if s >= 0}
            before = self.hands.get(p['side'])
            if before is not None:
                self.landed += [(frame['game_tick'], {'owner': p['side'], 'card_id': card})
                                for i, card in before.items() if hand.get(i) != card]
            self.hands[p['side']] = hand
        return list(self.landed)
