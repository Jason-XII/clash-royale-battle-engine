"""Replay conversion against frozen references, placement rules, archives and the observation's shared features."""
import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import convert
from conftest import ROOT
from cr import imitation, observation as C
from synthetic import arrays, dataset

FIXTURES = json.loads((C.ASSETS / 'test_replays.json').read_text())


class DealProbe:
    """Observed stock-deal permutations; omitted cards move to the draw queue."""

    def create_match(self, match):
        players = []
        for owner in (0, 1):
            ids = [c['d'] for c in match['battle'][f'deck{owner}']['sp']]
            opening, queue = map(list, convert.DEAL[owner])
            for i, slot in enumerate(opening):
                if convert.CARDS['cards'][str(ids[slot])]['omit']:
                    j = next(j for j, s in enumerate(queue) if not convert.CARDS['cards'][str(ids[s])]['omit'])
                    opening[i], queue[j] = queue[j], opening[i]
            players.append(dict(owner=owner,
                hand=[dict(handIndex=i, deckSlot=s, cardId=ids[s]) for i, s in enumerate(opening)],
                cycle=[dict(cycleIndex=i, deckSlot=s, cardId=ids[s]) for i, s in enumerate(queue)]))
        return {'players': players}


def test_replay_preparation_and_calibration_match_frozen_reference():
    for fixture in FIXTURES:
        game = convert.calibrate(convert.prepare(fixture['payload']), DealProbe())
        assert game['match'] == fixture['match']
        assert convert._jsonable(game['commands']) == fixture['commands']
        assert convert._jsonable(game['terminal']) == fixture['terminal']


def test_replay_rejects_ambiguous_sides_unknown_cards_and_impossible_deals():
    payload = FIXTURES[0]['payload']
    missing = copy.deepcopy(payload)
    for event in missing['events']:
        del event['source_fields']
    with pytest.raises(ValueError, match='data_i'):
        convert.prepare(missing)
    mixed = copy.deepcopy(payload)
    mixed['events'][0]['source_fields']['data_i'] = 1
    with pytest.raises(ValueError, match='data_i'):
        convert.prepare(mixed)
    unknown = copy.deepcopy(payload)
    unknown['battle']['team']['players'][0]['deck'][0]['card_key'] = 'future-card'
    with pytest.raises(ValueError, match='unsupported replay card'):
        convert.prepare(unknown)
    assert convert.select_deal(tuple(range(8)), [0, 0], '0289', 0) is None
    assert convert.select_deal(tuple(range(8)), [0], '0289', 0, omitted=(0,)) is None


def test_placement_matches_all_reference_states_including_mirror():
    digest = hashlib.sha256()
    for card in sorted(C.COSTS):
        for owner in (0, 1):
            for lanes in ((), ('left',), ('right',), ('left', 'right')):
                for last in ([None, *sorted(C.COSTS)] if card == C.MIRROR else [None]):
                    mask = C.legal_rows(card, owner, lanes, last_card=last)
                    assert mask.shape == (32, 18) and mask.dtype == bool
                    digest.update(np.packbits(mask).tobytes())
    # captured independently from the original placement provider over all 1,968 states
    assert digest.hexdigest() == '066d3b222ef48c1c5b5c82327099e12c6b7ed0b9ced564ab23ff1327abff684e'
    with pytest.raises(ValueError, match='unsupported placement card'):
        C.legal_rows(99999999, 0)


def test_recorder_shares_frame_objects_with_decisions():
    class Reader:
        reads = 0

        def frame(self, battle):
            Reader.reads += 1
            return dict(state={'players': []}, players=[], objects=[{'nativeObjectId': 7, 'hp': 100}])

        def unit_states(self, battle):
            return [{'nativeObjectId': 7, 'hidden': False}]
    env = SimpleNamespace(reader=Reader(), battle=object(), stalled=False, now=0)
    env.tick = lambda: env.now
    env._advance = lambda: setattr(env, 'now', env.now + 1)
    env._result = lambda: (None, False, None)
    recorder = convert.Recorder(env)
    frames, tick, ended = recorder.run_lean(10, every=5)
    assert (tick, ended, Reader.reads) == (10, False, 2)
    assert [d['tick'] for d in recorder.decisions] == [5, 10]
    assert recorder.decisions[0]['objects'] is frames[0]['objects']
    assert recorder.decisions[0]['units'] == [{'nativeObjectId': 7, 'hidden': False}]


def test_converter_does_not_import_torch():
    subprocess.run([sys.executable, '-c', "import sys, convert; assert 'torch' not in sys.modules"], check=True, cwd=ROOT)


def test_archive_round_trip(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    entry = json.loads((root / 'manifest.json').read_text())['entries'][0]
    assert entry['decisions'] == 7
    x, y = arrays(7, 0)
    expected = convert._pack(x, convert.targets(x, y))
    actual = imitation.read(root / entry['file'], 0)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    chunk_x, chunk_t = next(imitation.chunks(actual, 100, True, None))
    np.testing.assert_array_equal(chunk_x['legal'], x['legal'])
    np.testing.assert_array_equal(chunk_x['obj_card'][:, :1], x['obj_card'][:, :1])
    rejected = convert.archive_path(tmp_path, 'rejected')
    convert.write_archive(rejected, None, None, None, dict(tag='rejected', kept=False, why='test'))
    assert convert.read_metadata(rejected)['entries'] == []


def test_shared_board_features_preserve_both_player_views():
    card = 26000000
    obj = dict(nativeObjectId=123, cardId=card, dataGlobalId=card, owner=0, x=4500, y=12000,
               hp=400, maxHp=1000, targetEntityKey=None)
    unit = dict(nativeObjectId=123, kind='unit', hidden=True, airborne=False, surfaceAt=(9000, 20000),
                attack={'cycleMs': 500, 'hitSpeedMs': 1000}, buffs=[{'remainingMs': 700, 'stopsMovement': True}])
    record = dict(objects=[obj], units=[unit])
    vocab = C.Vocab([card])
    shared = {}
    for owner in (0, 1):
        state = dict(dropped=0, effects_dropped=0)
        expected = C.decision_inputs(record, owner, vocab, {}, dict(state))
        actual = C.decision_inputs(record, owner, vocab, {}, state, shared)
        for k in expected:
            np.testing.assert_array_equal(actual[k], expected[k])
    assert shared['values'][123][16:18] == (0.0, 0.0), 'relative aim must not mutate shared features'


@pytest.mark.skipif(C._native_features is None, reason='optional native observation helpers are not built')
def test_native_feature_helpers_match_python():
    import ctypes
    fast = C._native_features
    for value in (-2147483648, -1, 0, 2147483647):
        scalar = ctypes.c_int32(value)
        assert fast.i32(ctypes.addressof(scalar)) == value
    for value in (0, 1, 2 ** 63, 2 ** 64 - 1):
        scalar = ctypes.c_uint64(value)
        assert fast.u64(ctypes.addressof(scalar)) == value
    rng = np.random.default_rng(17)
    for _ in range(1000):
        n = lambda: int(rng.integers(-1000, 30000))
        u = dict(kind=rng.choice(['unit', 'area', 'timed', 'projectile']), state=n(), charge=n(), z=n(),
                 hidden=bool(rng.integers(2)), immune=None, invisible=False, airborne=True, pushed=True,
                 remainingMs=n(), fuseMs=n(), spawnTimerMs=n(), pushRemainingMs=n(), jumpRemainingMs=n(),
                 invisibleAgainMs=n(), shield=n(), shieldMax=1000,
                 buffs=[dict(stopsMovement=True, stopsAttacks=False, remainingMs=n(), speedPercent=n(), hitSpeedPercent=n())],
                 attack=dict(cycleMs=n(), hitSpeedMs=1000, loadMs=n(), stage=n(), dashCooldownMs=n(),
                             specialCooldownMs=n(), dashHop=n(), returnShot=dict(elapsedMs=n(), totalMs=1500)),
                 ability=dict(phase=rng.choice(['active', 'casting', '']), used=True, sinceUseMs=n(), uses=n(), maxCharges=3))
        for unit in ({}, u, dict(u, buffs=None, attack=None, ability=None, shieldMax=None)):
            assert fast.unit_state_values(unit) == C.unit_state_values(unit)
