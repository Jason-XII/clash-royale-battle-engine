"""The pipeline's checks: replay preparation against frozen references, placement rules, archives, a tiny training run
with resume, live play bookkeeping and the PPO update. No engine, emulator or GPU needed.

    OMP_NUM_THREADS=2 python -m pytest test_pipeline.py
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import common as C
import convert
import play
import ppo
import train

FIXTURES = json.loads((C.ASSETS / 'test_replays.json').read_text())


# ---- conversion ---------------------------------------------------------------------------------------------------------
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
    subprocess.run([sys.executable, '-c', "import sys, convert; assert 'torch' not in sys.modules"], check=True)


# ---- archives and training ----------------------------------------------------------------------------------------------
def arrays(size, marker):
    zeros = lambda *shape: np.zeros((size, *shape), np.float32)  # noqa: E731
    ids = lambda *shape: np.zeros((size, *shape), np.int64)      # noqa: E731
    x = dict(obj_card=ids(2), obj_data=ids(2), obj_side=ids(2), obj_kind=ids(2), obj_float=zeros(2, C.OBJ_FLOATS),
             eff_id=ids(2, 1), eff_left=zeros(2, 1), hand_card=np.full((size, 4), 2, np.int64), mirror_card=ids(4),
             hand_float=zeros(4, C.HAND_FLOATS), cycle_card=ids(4), ability=zeros(2, 4), own_flight_card=ids(),
             own_flight_float=zeros(4), opp_flight_card=ids(2), opp_flight_float=zeros(2, 4), opp_seen_card=ids(8),
             opp_seen_float=zeros(8, 3), opp_hero_card=ids(2), scalars=zeros(C.SCALARS),
             legal=np.ones((size, 4, C.H, C.W), bool))
    x['obj_side'][:, 0], x['obj_card'][:, 0] = 1, 2
    x['obj_float'][:, 0, 0] = np.arange(size) / size
    x['scalars'][:, 0], x['scalars'][:, 2] = marker / 10, 1
    y = dict(action_kind=ids(), action_card=ids(), action_cell=ids(), action_controller=ids(),
             label_ok=np.ones(size, bool), value=np.full(size, 1 if marker % 2 else -1, np.float32))
    y['action_kind'][::3], y['action_card'][::3] = 1, 2
    return x, y


def dataset(root, monkeypatch, sizes=(7, 11, 5)):
    """A converted dataset of synthetic games, written by convert.write_archive and sealed like convert.convert."""
    root.mkdir()
    entries = []
    for index, size in enumerate(sizes):
        tag = next(f'g{index}-{i}' for i in range(10000) if convert.held_out(f'g{index}-{i}') == (index == 2))
        monkeypatch.setattr(convert, 'game_features', lambda game, owner, vocab, legal: arrays(size, index + owner))
        path = convert.archive_path(root, tag)
        convert.write_archive(path, {}, C.Vocab([]), None, dict(tag=tag, kept=True))
        entries += convert.read_metadata(path)['entries']
    (root / 'manifest.json').write_text(json.dumps(dict(schema=dict(vocab=[26000000]), entries=entries)))
    return root


def test_archive_round_trip(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    entry = json.loads((root / 'manifest.json').read_text())['entries'][0]
    assert entry['decisions'] == 7
    x, y = arrays(7, 0)
    expected = convert._pack(x, convert.targets(x, y))
    actual = train.read(root / entry['file'], 0)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    chunk_x, chunk_t = next(train.chunks(actual, 100, True, None))
    np.testing.assert_array_equal(chunk_x['legal'], x['legal'])
    np.testing.assert_array_equal(chunk_x['obj_card'][:, :1], x['obj_card'][:, :1])
    rejected = convert.archive_path(tmp_path, 'rejected')
    convert.write_archive(rejected, None, None, None, dict(tag='rejected', kept=False, why='test'))
    assert convert.read_metadata(rejected)['entries'] == []


def test_training_runs_resumes_and_loads_for_play(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    settings = dict(train.DEFAULTS, epochs=2, batch=2, chunk=3, workers=0, card_dim=4, object_dim=4, conv_channels=4,
                    conv_layers=1, core=8, attention_layers=0, log_every=1, checkpoint_every=2, device='cpu')
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
    model, vocab, _ = train.load(out / 'latest.pt')
    x, _ = arrays(1, 0)
    action = train.Agent(model, 0).act({k: v[0] for k, v in x.items()})
    assert action[0] in ('wait', 'card', 'ability')


def test_packed_inputs_and_ppo_update(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    vocab = C.Vocab([26000000])
    torch.manual_seed(0)
    model = train.Policy(train.Config(vocab_size=len(vocab), card_dim=4, object_dim=4, conv_channels=4, conv_layers=1,
                                      core=8, attention_layers=0))
    trajectories = []
    for entry in json.loads((root / 'manifest.json').read_text())['entries']:
        x, _ = next(train.chunks(train.read(root / entry['file'], entry['owner']), 300, True, None))
        T = len(x['scalars'])
        rows = [train.pack({k: v[t] for k, v in x.items()}) for t in range(T)]
        inputs = {k: train.pad_stack([r[k] for r in rows]) for k in rows[0]}
        packed = train.tensors({k: v[None] for k, v in inputs.items()}, 'cpu')
        with torch.no_grad():
            a, b = model({k: torch.from_numpy(v[None]) for k, v in x.items()}), model(packed)
            for k in ('what', 'which', 'value'):
                assert torch.allclose(a[k], b[k], atol=1e-4, equal_nan=True), k
            action, logp = train.sample(model, b, packed, train.ABILITY_EXPLORE)
            again = ppo.evaluate(model, packed, action[None], torch.ones(1, T, dtype=torch.bool))
        assert torch.allclose(again['logp'], logp, atol=1e-4)
        reward = np.zeros(T, np.float32)
        reward[-1] = 1.0
        trajectories.append(dict(inputs=inputs, opp={k: inputs[k][:, ::-1].copy() for k in ppo.PRIVATE},
                                 action=action.numpy(), logp=logp.numpy(), reward=reward, outcome=1.0,
                                 value=b['value'][0].numpy()))
    trajectories[-1]['reward'][-1] = trajectories[-1]['outcome'] = -1.0
    reference = copy.deepcopy(model).requires_grad_(False)
    critic = ppo.Critic(copy.deepcopy(model))
    args = argparse.Namespace(gamma=0.999, lam=0.95, upgo=0.5, minibatch=2, epochs=1, critic_epochs=2, clip=0.2, entropy=0.002,
                              kl_coef=0.05)
    device = torch.device('cpu')
    batches = ppo.minibatches(trajectories, args, device)
    stats = ppo.critic_targets(batches, critic, args, device)
    assert all(len(t['adv']) == len(t['action']) for t in trajectories)
    stats.update(ppo.update(model.train(), reference, torch.optim.Adam(model.parameters(), lr=1e-5), batches,
                            args, device))
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    first = ppo.train_critic(critic, optimizer, batches, args, device)
    for _ in range(5):
        last = ppo.train_critic(critic, optimizer, batches, args, device)
    assert all(np.isfinite(v) for v in {**stats, **first, **last}.values())
    assert stats['kl_ref'] < 1e-3 and stats['approx_kl'] < 1e-3, 'the first update starts from the reference'
    assert last['win_loss'] < first['win_loss'], 'the critic learns who won'
    advantage, returns = ppo.advantages(np.array([1., 0., -1.], np.float32), np.zeros(3, np.float32), gamma=1., lam=1.)
    np.testing.assert_array_equal(advantage, [0., -1., -1.])
    value = np.array([0., .5, .9], np.float32)  # UPGO follows the path while it beats the critic, else bootstraps
    np.testing.assert_allclose(ppo.upgo(np.array([0., 0., 1.], np.float32), value, 1.), [1., .5, .1], atol=1e-6)
    np.testing.assert_allclose(ppo.upgo(np.array([0., 0., -1.], np.float32), value, 1.), [.9, .4, -1.9], atol=1e-6)


# ---- live play ----------------------------------------------------------------------------------------------------------
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


def test_actor_learner_link_and_shaping(tmp_path):
    inbox = ppo.Inbox(b'k' * 16, (3, {'w': np.ones(2, np.float32)}, {'il': ('il.pt', 0.3)}))
    (tmp_path / 'endpoint.json').write_text(json.dumps(dict(host='127.0.0.1', ports=[inbox.port], authkey=(b'k' * 16).hex())))
    link = ppo.Link(tmp_path)
    weights, opponents = link.exchange([], [], {})
    assert link.version == 3 and weights['w'].sum() == 2 and opponents['il'][1] == 0.3
    weights, _ = link.exchange([dict(version=1), dict(version=3)], [dict(opponent='il', outcome=1.0)], {'steps': 1})
    assert weights is None, 'weights are sent once per version'
    games, results, timing, stale = inbox.take(2)
    assert [g['version'] for g in games] == [3] and stale == 1 and len(results) == 1 and timing == [{'steps': 1}]
    tower = lambda i, owner, hp, x=3500: SimpleNamespace(id=i, owner=owner, hp=hp, x=x, card=-1, entity_type='building')  # noqa: E731
    full = SimpleNamespace(crowns=(0, 0), entities=(tower(1, 0, 100), tower(2, 1, 100), tower(3, 1, 200, 9000)))
    start = ppo.tower_hp(full)
    hit = SimpleNamespace(crowns=(1, 0), entities=(tower(1, 0, 50), tower(3, 1, 100, 9000)))  # theirs gone, king half
    assert ppo.potential(full, start, 0) == 0
    assert ppo.potential(hit, start, 0) == pytest.approx(1 + 1 + 1.5 - 0.5)
    assert ppo.potential(hit, start, 1) == pytest.approx(-3.0)


def test_wire_rows_stage_like_padded_rows():
    x, _ = arrays(3, 1)
    rows = [train.pack({k: v[t] for k, v in x.items()}) for t in range(3)]
    rows[1] = {k: v[:1] if k in C.OBJECT_KEYS else v for k, v in rows[1].items()}  # fewer objects
    rows[2] = {k: v[:0] if k in C.OBJECT_KEYS else v for k, v in rows[2].items()}  # none
    shape = ppo.layout(rows[0])
    staged = ppo.stage([ppo.wire(r, shape) for r in rows], shape, 4, 16)
    expected = train.tensors({k: train.pad_stack([r[k] for r in rows]) for k in rows[0]}, 'cpu')
    for k, v in expected.items():
        got = torch.from_numpy(staged[k][:3, 0])
        assert got.dtype == v.dtype, k
        assert torch.equal(got[(slice(None), *map(slice, v.shape[1:]))], v), k
        assert not staged[k][3].any() and (k not in C.OBJECT_KEYS or not got[:, v.shape[1]:].any()), k


def test_search_targets_and_distillation(tmp_path, monkeypatch):
    """Search scores -> targets; trajectories carry them; the PPO update pulls the tile head to the search's choice."""
    assert ppo.search_target([0.5, 0.5, 0.5], 0.02) is None, 'scores that do not differ give no label'
    target = ppo.search_target([0.0, 0.04, 0.02], 0.02)
    assert target.argmax() == 1 and abs(target.sum() - 1) < 1e-6
    root = dataset(tmp_path / 'data', monkeypatch)
    torch.manual_seed(0)
    model = train.Policy(train.Config(vocab_size=len(C.Vocab([26000000])), card_dim=4, object_dim=4, conv_channels=4,
                                      conv_layers=1, core=8, attention_layers=0))
    entry = json.loads((root / 'manifest.json').read_text())['entries'][0]
    x, _ = next(train.chunks(train.read(root / entry['file'], entry['owner']), 300, True, None))
    T = len(x['scalars'])
    rows = [train.pack({k: v[t] for k, v in x.items()}) for t in range(T)]
    seat = ppo.Seat()
    for t, r in enumerate(rows):
        seat.rows.append(r)
        seat.opp.append(r)
        seat.action.append(0)
        seat.logp.append(0.0)
        seat.value.append(0.0)
        seat.reward.append(0.0)
        legal = x['legal'][t].reshape(4, -1)
        slot = int(legal.any(1).argmax())
        cells = [int(c) for c in np.flatnonzero(legal[slot])[:3]]
        if t % 3 == 0 and len(cells) == 3:
            seat.search.append((t, slot, cells, np.array([0.1, 0.8, 0.1], np.float32)))
    trajectory = seat.trajectory(version=0, opponent='il', outcome=1.0)
    S = len(seat.search)
    assert S and trajectory['search_cells'].shape == (S, ppo.SEARCH_K + 1)
    assert (trajectory['search_cells'][:, 3:] == -1).all() and trajectory['search_t'].tolist() == [e[0] for e in seat.search]
    trajectory['adv'] = np.zeros(T, np.float32)
    reference = copy.deepcopy(model).requires_grad_(False)
    args = argparse.Namespace(minibatch=1, epochs=1, clip=0.2, entropy=0.0, kl_coef=0.0, search_coef=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    device = torch.device('cpu')
    batches = ppo.minibatches([trajectory], args, device)
    first = ppo.update(model.train(), reference, optimizer, batches, args, device)
    for _ in range(15):
        last = ppo.update(model, reference, optimizer, batches, args, device)
    assert np.isfinite(first['search_loss']) and last['search_loss'] < first['search_loss']
    assert last['search_agree'] == 1.0, 'the tile head now prefers the cell the search chose'
    xt = train.tensors({k: np.stack([r[k] for r in rows])[None] for k in rows[0]}, 'cpu')
    with torch.no_grad(), torch.autocast('cpu', torch.bfloat16):
        f, fx = train.flat(model(xt), xt)
        assert model.tiles(f, fx, torch.zeros(T, dtype=torch.long)).dtype == torch.float32, 'tile scores stay fp32'


def test_decision_server_returns_the_sampled_cards_legal_top_tiles(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    torch.manual_seed(0)
    model = train.Policy(train.Config(vocab_size=len(C.Vocab([26000000])), card_dim=4, object_dim=4, conv_channels=4,
                                      conv_layers=1, core=8, attention_layers=0)).eval()
    entry = json.loads((root / 'manifest.json').read_text())['entries'][0]
    x, _ = next(train.chunks(train.read(root / entry['file'], entry['owner']), 300, True, None))
    rows = [train.pack({k: v[t] for k, v in x.items()}) for t in range(len(x['scalars']))]
    decider = ppo.Decider({'current': model}, torch.device('cpu'), 8)
    decider.shape = ppo.layout(rows[0])
    decider.reset(0)
    played = 0
    for t, row in enumerate(rows):
        (a, lp, v, top), = decider.act([((0, 0), 'current', None, ppo.wire(row, decider.shape))])
        assert len(top) == ppo.SEARCH_K
        if a >= train.CARD:
            slot, cell = divmod(a - train.CARD, train.CELLS)
            legal = x['legal'][t].reshape(4, -1)[slot]
            assert all(legal[c] for c in top if c >= 0) and any(c >= 0 for c in top)
            played += 1
    assert played, 'the untrained policy played at least one card'


def test_unaffordable_cards_are_not_offered_and_held_taps_are_in_flight():
    legal = torch.ones(1, 4, C.H, C.W, dtype=torch.bool)
    hf = torch.zeros(1, 4, C.HAND_FLOATS)
    hf[0, :, 0] = torch.tensor([3, 4, 5, 9]) / 10          # costs 3, 4, 5, 9
    sc = torch.zeros(1, C.SCALARS)
    sc[0, 0], sc[0, 2] = 0.40, 1.0                           # 4.0 elixir, 1x rate: the hold adds ~0.41
    x = {'legal': legal, 'hand_float': hf, 'scalars': sc, 'own_flight_card': torch.zeros(1, dtype=torch.long)}
    assert train.playable(x).tolist() == [[True, True, False, False]]
    which = train.which_logp(torch.zeros(1, 4), x)
    assert torch.isinf(which[0, 2:]).all() and torch.isfinite(which[0, :2]).all()
    sc[0, 0] = 0.0                                           # nothing affordable: playing is not an option
    what = train.what_logp(torch.zeros(1, 4), x)
    assert torch.isinf(what[0, train.PLAY]) and torch.isfinite(what[0, train.WAIT])
    # PlayerView's own-flight rule: a held tap (tap known) is in flight from the tap, not only from landing - 22
    landed = [(200, {'owner': 0, 'card_id': 26000021, 'x': 3500, 'y': 10500, 'tap': 150})]
    assert [tt for tt, c in landed if c.get('tap', tt - C.DEPLOY_DELAY) <= 160 < tt] == [200]
    assert not [tt for tt, c in landed if tt - C.DEPLOY_DELAY <= 160 < tt]
