"""PPO: the update, the critic, the actor-learner link, decision servers, routing, ranks in step, the run's opponents."""
import argparse
import copy
import json
import math
import random
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import ROOT
from cr import actors, engine, imitation, matchmaking, observation as C, policy
import ppo
from synthetic import arrays, dataset


@pytest.mark.parametrize('option', [dict(conv_channels=4, conv_layers=1),
                                    dict(conv_channels=8, conv_layers=2, res_blocks=1, tile_blocks=1)])  # 21 Oct network
def test_packed_inputs_and_ppo_update(tmp_path, monkeypatch, option):
    root = dataset(tmp_path / 'data', monkeypatch)
    vocab = C.Vocab([26000000])
    torch.manual_seed(0)
    model = policy.Policy(policy.Config(vocab_size=len(vocab), card_dim=4, object_dim=4, core=8, attention_layers=0, **option))
    trajectories = []
    for entry in json.loads((root / 'manifest.json').read_text())['entries']:
        x, _ = next(imitation.chunks(imitation.read(root / entry['file'], entry['owner']), 300, True, None))
        T = len(x['scalars'])
        rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(T)]
        inputs = {k: policy.pad_stack([r[k] for r in rows]) for k in rows[0]}
        packed = policy.tensors({k: v[None] for k, v in inputs.items()}, 'cpu')
        with torch.no_grad():
            a, b = model({k: torch.from_numpy(v[None]) for k, v in x.items()}), model(packed)
            for k in ('what', 'which', 'value'):
                assert torch.allclose(a[k], b[k], atol=1e-4, equal_nan=True), k
            action, logp = policy.sample(model, b, packed, policy.ABILITY_EXPLORE)
            again = ppo.evaluate(model, packed, action[None], torch.ones(1, T, dtype=torch.bool))
            model.temperature, model.mix = 0.1, 0.1
            sharp, sharp_logp = policy.sample(model, b, packed, policy.ABILITY_EXPLORE)
            sharp_again = ppo.evaluate(model, packed, sharp[None], torch.ones(1, T, dtype=torch.bool))
            del model.temperature, model.mix
        assert torch.allclose(again['logp'], logp, atol=1e-4)
        assert torch.allclose(sharp_again['logp'], sharp_logp, atol=1e-4), 'the update scores what was sampled'
        reward = np.zeros(T, np.float32)
        reward[-1] = 1.0
        trajectories.append(dict(inputs=inputs, opp={k: inputs[k][:, ::-1].copy() for k in ppo.PRIVATE},
                                 action=action.numpy(), logp=logp.numpy(), reward=reward, outcome=1.0,
                                 value=b['value'][0].numpy()))
    trajectories[-1]['reward'][-1] = trajectories[-1]['outcome'] = -1.0
    reference = copy.deepcopy(model).requires_grad_(False)
    critic = ppo.Critic(copy.deepcopy(model))
    args = argparse.Namespace(gamma=0.999, lam=0.95, minibatch=2, epochs=1, critic_epochs=2, clip=0.2, entropy=0.002,
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


def test_actor_learner_link_and_shaping(tmp_path):
    inbox = ppo.Inbox(b'k' * 16, (3, {'w': np.ones(2, np.float32)}, {'il': ('il.pt', 0.3)}))
    (tmp_path / 'endpoint.json').write_text(json.dumps(dict(host='127.0.0.1', ports=[inbox.port], authkey=(b'k' * 16).hex())))
    link = actors.Link(tmp_path)
    weights, opponents = link.exchange([], [], {})
    assert link.version == 3 and weights['w'].sum() == 2 and opponents['il'][1] == 0.3
    weights, _ = link.exchange([dict(version=1), dict(version=3)], [dict(opponent='il', outcome=1.0)], {'steps': 1})
    assert weights is None, 'weights are sent once per version'
    games, results, timing, stale = inbox.take(2)
    assert [g['version'] for g in games] == [3] and stale == 1 and len(results) == 1 and timing == [{'steps': 1}]
    tower = lambda i, owner, hp, x=3500: SimpleNamespace(id=i, owner=owner, hp=hp, x=x, card=-1, entity_type='building')  # noqa: E731
    full = SimpleNamespace(crowns=(0, 0), entities=(tower(1, 0, 100), tower(2, 1, 100), tower(3, 1, 200, 9000)))
    start = engine.tower_hp(full)
    hit = SimpleNamespace(crowns=(1, 0), entities=(tower(1, 0, 50), tower(3, 1, 100, 9000)))  # theirs gone, king half
    assert engine.potential(full, start, 0) == 0
    assert engine.potential(hit, start, 0) == pytest.approx(1 + 1 + engine.KING_WEIGHT * 0.5 - 0.5)
    assert engine.potential(hit, start, 1) == pytest.approx(-(1 + 1 + engine.KING_WEIGHT * 0.5 - 0.5))
    assert engine.margin(hit, start, 0) == pytest.approx(0.5 - 0.0) and engine.margin(hit, start, 1) == pytest.approx(-0.5)
    assert engine.margin(full, start, 0) == 0


def test_wire_rows_stage_like_padded_rows():
    x, _ = arrays(3, 1)
    rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(3)]
    rows[1] = {k: v[:1] if k in C.OBJECT_KEYS else v for k, v in rows[1].items()}  # fewer objects
    rows[2] = {k: v[:0] if k in C.OBJECT_KEYS else v for k, v in rows[2].items()}  # none
    shape = actors.layout(rows[0])
    staged = actors.stage([actors.wire(r, shape) for r in rows], shape, 4, 16)
    expected = policy.tensors({k: policy.pad_stack([r[k] for r in rows]) for k in rows[0]}, 'cpu')
    for k, v in expected.items():
        got = torch.from_numpy(staged[k][:3, 0])
        assert got.dtype == v.dtype, k
        assert torch.equal(got[(slice(None), *map(slice, v.shape[1:]))], v), k
        assert not staged[k][3].any() and (k not in C.OBJECT_KEYS or not got[:, v.shape[1]:].any()), k
    fresh = actors.stage([actors.wire(rows[2], shape)], shape, 4, 16)
    actors.stage([actors.wire(rows[2], shape)], shape, 4, 16, out=staged)
    for k in fresh:
        np.testing.assert_array_equal(staged[k], fresh[k], err_msg=k)


def test_opponent_routing_groups_games_without_changing_their_opponents():
    rng = random.Random(17)
    opponents = {f'v{i}': ('unused.pt', i + 1) for i in range(7)}
    original = dict(opponents)
    counts, pairs = [0] * 4, set()
    for _ in range(20000):
        name = rng.choices(list(opponents), [p[1] for p in opponents.values()])[0]
        sid = matchmaking.route_collector(name, opponents, rng, 4)
        counts[sid] += 1
        pairs.add((name, sid))
    assert opponents == original
    assert max(abs(n - 5000) for n in counts) < 250, 'routing balances traffic despite skewed opponent weights'
    assert len(pairs) <= len(opponents) + 4 - 1
    assert 0 <= matchmaking.route_collector('current', opponents, rng, 4) < 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph serving requires a GPU')
def test_cuda_serving_preserves_recurrence_reload_and_reused_buffers(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = policy.Config(vocab_size=4, card_dim=4, object_dim=8, conv_channels=8, conv_layers=2,
                          core=16, attention_layers=1)
    models = {name: policy.Policy(config).cuda().eval() for name in ('current', 'opponent')}
    baseline = actors.Decider(models, torch.device('cuda'), config.core, capacity=128, fast=False)
    fast = actors.Decider({k: copy.deepcopy(v) for k, v in models.items()}, torch.device('cuda'), config.core,
                       capacity=128, fast=True)
    x, _ = arrays(40, 1)
    rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(40)]
    baseline.shape = fast.shape = actors.layout(rows[0])
    # Exercise repeated same-shaped chunks without allocating 512 full-size rows in the test.
    monkeypatch.setattr(actors, 'ROW_BUCKETS', (8, 16))
    requests = [((i, 0), 'opponent' if i % 5 == 0 else 'current', None, actors.wire(row, fast.shape))
                for i, row in enumerate(rows)]
    for d in (baseline, fast):
        d.h.zero_()
        d.c.zero_()
    for step in range(6):
        if step == 1:
            # Introduce a new object bucket while other model groups already have queued results.
            larger = [{k: np.repeat(v[:1], 20, axis=0) if k in C.OBJECT_KEYS else v
                       for k, v in row.items()} for row in rows]
            requests = [((i, 0), 'opponent' if i % 5 == 0 else 'current', None, actors.wire(row, fast.shape))
                        for i, row in enumerate(larger)]
        if step == 2:
            for d in (baseline, fast):
                d.reset(3)
        if step == 3:
            state = {k: v.clone() for k, v in models['current'].state_dict().items()}
            state['what.bias'] += 0.1
            for d in (baseline, fast):
                d.models['current'].load_state_dict(state)
        torch.cuda.manual_seed(100 + step)
        expected = baseline.act(requests)
        torch.cuda.manual_seed(100 + step)
        actual = fast.act(requests)
        assert actual == expected
        torch.testing.assert_close(fast.h[:, :len(fast.rows)], baseline.h[:, :len(baseline.rows)], rtol=0, atol=0)
        torch.testing.assert_close(fast.c[:, :len(fast.rows)], baseline.c[:, :len(baseline.rows)], rtol=0, atol=0)
    fast.drop('opponent')
    assert all(key[0] != 'opponent' for key in fast.graphs)


def test_decision_server_plays_only_legal_tiles_and_tile_scores_stay_fp32(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    torch.manual_seed(0)
    model = policy.Policy(policy.Config(vocab_size=len(C.Vocab([26000000])), card_dim=4, object_dim=4, conv_channels=4,
                                      conv_layers=1, core=8, attention_layers=0)).eval()
    entry = json.loads((root / 'manifest.json').read_text())['entries'][0]
    x, _ = next(imitation.chunks(imitation.read(root / entry['file'], entry['owner']), 300, True, None))
    rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(len(x['scalars']))]
    decider = actors.Decider({'current': model}, torch.device('cpu'), 8)
    decider.shape = actors.layout(rows[0])
    decider.reset(0)
    played = 0
    for t, row in enumerate(rows):
        (a, lp, v), = decider.act([((0, 0), 'current', None, actors.wire(row, decider.shape))])
        if a >= policy.CARD:
            slot, cell = divmod(a - policy.CARD, policy.CELLS)
            assert x['legal'][t].reshape(4, -1)[slot][cell]
            played += 1
    assert played, 'the untrained policy played at least one card'
    xt = policy.tensors({k: np.stack(rows_k)[None] for k, rows_k in ((k, [r[k] for r in rows]) for k in rows[0])}, 'cpu')
    with torch.no_grad(), torch.autocast('cpu', torch.bfloat16):
        f, fx = policy.flat(model(xt), xt)
        assert model.tiles(f, fx, torch.zeros(len(rows), dtype=torch.long)).dtype == torch.float32, 'tile scores stay fp32'


def test_a_non_finite_gradient_skips_the_step():
    w = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([w], lr=1.0)
    stats = {'skipped': []}
    (w * float('nan')).sum().backward()
    assert not ppo.finite_step([w], optimizer, 1.0, stats, 'skipped') and torch.equal(w.data, torch.ones(2))
    w.grad = None
    w.sum().backward()
    assert ppo.finite_step([w], optimizer, 1.0, stats, 'skipped') and stats['skipped'] == [1.0, 0.0]


def test_every_rank_gets_collectors_and_an_empty_rank_stops_the_update_loudly(monkeypatch):
    ports = []
    monkeypatch.setattr(actors, 'endpoint', lambda run: dict(host='h', ports=[1, 2, 3, 4], authkey='00'))
    monkeypatch.setattr(actors, 'Client', lambda address, authkey: ports.append(address[1]) or SimpleNamespace(send=lambda message: None))
    monkeypatch.setattr(actors, 'receive', lambda conn: (0, None, {}))
    for i in range(8):
        actors.Link('run', weights=False, index=i).exchange([], [], {})
    assert sorted(ports) == [1, 1, 2, 2, 3, 3, 4, 4]
    monkeypatch.setattr(ppo.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(ppo.dist, 'get_world_size', lambda: 2)
    monkeypatch.setattr(ppo.dist, 'all_gather_object', lambda out, n: out.__setitem__(slice(None), [n, 0]))
    with pytest.raises(RuntimeError):
        ppo.schedule(list(range(64)), 2)  # learn() waits for every rank; were one still empty, nobody would train


def test_frozen_models_play_without_the_press_floor_and_are_told_apart_by_checkpoint(tmp_path):
    def model(bias):  # [wait, play, ability 1, ability 2]
        m = policy.Policy(policy.Config(vocab_size=3, card_dim=4, object_dim=4, attention_layers=0, conv_channels=4,
                                      conv_layers=1, core=8)).eval()
        with torch.no_grad():
            m.what.weight.zero_()
            m.what.bias[:] = torch.tensor(bias)
        return m
    never, always = [0., -1000., -1000., -1000.], [-1000., -1000., 0., -1000.]
    paths = []
    for run, bias in (('a', never), ('b', always)):  # two runs' v000100
        m, path = model(bias), tmp_path / run / 'v000100.pt'
        path.parent.mkdir()
        torch.save(dict(config=m.config.__dict__, model=m.state_dict(), vocab=[26000000]), path)
        paths.append(str(path))
    x, _ = arrays(1, 10)
    x['ability'][0, 0, :2] = 1  # controller 1 ready
    row = policy.pack({k: np.asarray(v[0]) for k, v in x.items()})
    decider = actors.Decider({'current': model(never)}, torch.device('cpu'), 8)
    decider.shape = actors.layout(row)
    decider.reset(0)
    w = actors.wire(row, decider.shape)
    (a0, lp0, _), (a1, lp1, _), (a2, _, _) = decider.act(
        [((0, 0), 'current', None, w), ((0, 1), 'v000100', paths[0], w), ((1, 0), 'v000100', paths[1], w)])
    assert a0 == a1 == C.WAIT and abs(lp0 - math.log(1 - policy.ABILITY_EXPLORE)) < 1e-4, 'the learner explores'
    assert abs(lp1) < 1e-4, 'a frozen model plays as deployed: no press floor'
    assert a2 == C.ABILITY1, "the second run's v000100 is its own model"


def test_a_specialist_deck_goes_to_a_learning_seat_or_the_judged_candidate():
    rng = random.Random(0)
    assert matchmaking.deck_seat([('v000160', '/a'), ('v000140', '/b')], dict(a=['v000140', '/b'], b=['v000160', '/a']), rng) == 1
    assert matchmaking.deck_seat([('il', '/il'), ('current', None)], None, rng) == 1


def test_ranks_step_together_so_adam_stays_in_sync(tmp_path):
    """Two gloo ranks with 3 and 5 minibatches (PPO) and 3 and 5 batches (IL): the same number of optimizer steps,
    the same weights and Adam moments after (with DDP's Join they differed). Then a step past DDP, as the imitation
    step takes it (the bare module, gradients averaged by hand): the same weights again."""
    script = tmp_path / 'ranks.py'
    script.write_text(textwrap.dedent(f'''
        import sys, torch, torch.distributed as dist
        sys.path.insert(0, {str(ROOT)!r})
        import ppo, train
        rank = int(sys.argv[1])
        dist.init_process_group('gloo', init_method={('file://' + str(tmp_path / 'store'))!r}, rank=rank, world_size=2)
        torch.manual_seed(0)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(3, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        torch.manual_seed(1 + rank)
        data = [torch.randn(4, 3) for _ in range(3 + 2 * rank)]
        def step(b):
            optimizer.zero_grad()
            model(b).pow(2).mean().backward()
            optimizer.step()
        for b in ppo.schedule(data, 2):
            step(b)
        n = sum(1 for b in train.lockstep(iter(data), torch.device('cpu'), 2) if step(b) or True)
        optimizer.zero_grad()
        model.module(data[0] * (1 + rank)).pow(2).mean().backward()
        ppo.average_gradients(model.parameters())
        optimizer.step()
        state = [p.detach().clone() for p in model.parameters()] + [s[k] for s in optimizer.state.values() for k in ('exp_avg', 'exp_avg_sq')]
        out = [None, None]
        dist.all_gather_object(out, ([t.tolist() for t in state], n))
        assert out[0] == out[1], out
        print('same', n)
    '''))
    procs = [subprocess.Popen([sys.executable, str(script), str(r)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for r in (0, 1)]
    results = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), results
    assert all(out.strip() == 'same 3' for out, _ in results), results


def test_an_exploiter_plays_only_the_main_runs_champion(tmp_path):
    main, run = tmp_path / 'main', tmp_path / 'exploiter'
    for d in (main, run / 'pool', run / 'league'):
        d.mkdir(parents=True)
    (run / 'pool' / 'v000140.pt').touch()
    (main / 'champion.json').write_text(json.dumps(dict(name='v000120', path='/main/league/v000120.pt')))
    s = argparse.Namespace(**ppo.DEFAULTS)
    assert matchmaking.opponent_paths(run, 'il.pt', s) == {'il': 'il.pt', 'v000140': str(run / 'pool' / 'v000140.pt')}
    s.against = str(main)
    assert matchmaking.opponent_paths(run, 'il.pt', s) == {'v000120': '/main/league/v000120.pt'}
    (main / 'champion.json').write_text(json.dumps(dict(name='v000140', path='/main/league/v000140.pt')))
    assert matchmaking.opponent_paths(run, 'il.pt', s) == {'v000140': '/main/league/v000140.pt'}, 'it follows promotions'


def test_a_server_sees_when_a_restarted_learner_changed_its_sampling_settings(tmp_path):
    write = lambda **kw: (tmp_path / 'endpoint.json').write_text(json.dumps({'settings': kw}))  # noqa: E731
    write(learner_temperature=0.5, lr=1e-5)
    before = actors.served(tmp_path)
    write(learner_temperature=0.5, lr=2e-5)
    assert actors.served(tmp_path) == before, 'other settings do not restart servers'
    write(learner_temperature=0.1, lr=2e-5)
    assert actors.served(tmp_path) != before


def test_a_decision_server_holds_models_of_different_sizes(tmp_path):
    """A large run's servers also play its medium benchmarks: each LSTM size keeps its own states."""
    torch.manual_seed(0)
    config = lambda core: policy.Config(vocab_size=4, card_dim=4, object_dim=8, conv_channels=8, conv_layers=1,  # noqa: E731
                                       core=core, attention_layers=0)
    small = policy.Policy(config(8)).eval()
    torch.save(dict(config=config(8).__dict__, model=small.state_dict(), vocab=[26000000]), tmp_path / 'small.pt')
    decider = actors.Decider({'current': policy.Policy(config(16)).eval()}, torch.device('cpu'), 16, frozen_temperature=0.1)
    x, _ = arrays(6, 1)
    rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(6)]
    decider.shape = actors.layout(rows[0])
    decider.reset(0)
    for row in rows:
        decider.act([((0, 0), 'current', None, actors.wire(row, decider.shape)),
                     ((0, 1), 'small', tmp_path / 'small.pt', actors.wire(row, decider.shape))])
    assert set(decider.states) == {8, 16}
    seat = decider.rows[(0, 1)]
    assert decider.states[8][0][0, seat].abs().sum() > 0 and decider.states[16][0][0, seat].abs().sum() == 0
    decider.reset(0)
    assert all(h[0, seat].abs().sum() == 0 for h, _ in decider.states.values())
    assert decider.models[tmp_path / 'small.pt'].temperature == 0.1 and getattr(decider.models['current'], 'temperature', 1.0) == 1.0, \
        'frozen models play sharpened, the learner as trained'


def test_fixed_opponent_shares_leave_the_rest_to_pfsp():
    paths = {'il': '/il.pt', 'g-v240': '/v240.pt', 'v000020': '/a.pt', 'v000040': '/b.pt'}
    results = {n: [] for n in paths} | {'v000020': [1.0, 1.0], 'v000040': [-1.0]}
    plain = matchmaking.weights(paths, results)
    assert plain['v000020'][1] == pytest.approx(0.05) and plain['il'][1] == pytest.approx(0.3), 'PFSP as before'
    w = matchmaking.weights(paths, results, {'il': ['/il.pt', 0.4], 'g-v240': ['/v240.pt', 0.4]})
    assert w['il'] == ('/il.pt', 0.4) and w['g-v240'][1] == 0.4
    assert w['v000020'][1] + w['v000040'][1] == pytest.approx(0.2) and w['v000040'][1] > w['v000020'][1]
    assert matchmaking.weights({'il': '/il.pt'}, {'il': []}, {'il': ['/il.pt', 0.4], 'x': ['/x.pt', 0.4]})['x'][1] == 0.4, \
        'no PFSP opponent yet: the shares alone (the collectors draw by weight)'
    with pytest.raises(ValueError):
        matchmaking.weights(paths, results, {'il': ['/il.pt', 0.7], 'g-v240': ['/v240.pt', 0.4]})


def test_a_specialist_meets_focus_hard_and_pool_decks(tmp_path):
    deck = lambda *ids: [dict(d=i) for i in ids]  # noqa: E731
    pool = [dict(decks=[deck(1, 2), deck(3, 4)], lvlcap=16), dict(decks=[deck(5, 6), deck(1, 2)], lvlcap=14),
            dict(decks=[deck(7, 8), deck(9, 10)], lvlcap=15)]
    decks = matchmaking.Decks(pool)
    assert decks.focus[0] == '1-2' and len(decks.seats['1-2']) == 2
    rng = random.Random(0)
    drawn = [decks.pick(pool[2], 1, {'focus': 0.3, 'hard': 0.3}, {'5-6': 1.0}, rng) for _ in range(3000)]
    theirs = [matchmaking.deck_key(p['decks'][1]) for p in drawn]
    assert 0.25 < theirs.count('5-6') / 3000 < 0.35 + 0.3 / 5, 'the hard share (and focus, uniform over 5 sets)'
    assert 0.35 < theirs.count('9-10') / 3000 < 0.45 + 0.3 / 5, "the pair's own"
    hard_5_6 = next(p for p in drawn if matchmaking.deck_key(p['decks'][1]) == '5-6')
    assert hard_5_6['lvlcap'] == 14, "a drawn deck comes with its own pair's levels"
    assert all(decks.pick(pool[2], 0, {'hard': 1.0}, {}, rng) is pool[2] for _ in range(20)), 'no hard decks yet: the pool'
    scores = {'easy': [1.0] * 30, 'hard': [0.0] * 30, 'new': [0.0], 'even': [0.5] * 30}
    hard = matchmaking.hard_decks(scores)
    assert set(hard) == {'hard', 'new', 'even'} and hard['hard'] > hard['even'] > hard['new'], \
        'below 0.55 smoothed; many losses weigh more than one'
    record = matchmaking.DeckScores(tmp_path)
    record.add([dict(deck='a', deck_score=0.0)] * 40 + [dict(opponent='il')])
    record.save()
    assert len(matchmaking.DeckScores(tmp_path).scores['a']) == matchmaking.RECENT, 'recent games only, kept on restart'
    assert matchmaking.read_hard(tmp_path) == matchmaking.hard_decks({'a': [0.0] * matchmaking.RECENT})


def test_the_imitation_step_learns_the_human_games(tmp_path, monkeypatch):
    root = dataset(tmp_path / 'data', monkeypatch)
    entries = json.loads((root / 'manifest.json').read_text())['entries']
    index = tmp_path / 'index.json'
    index.write_text(json.dumps(dict(data=str(root), entries=[dict(e, outcome=1.0 if i % 2 else -1.0)
                                                              for i, e in enumerate(entries)])))
    experts = imitation.Experts(index, 0)
    assert all(e['outcome'] > 0 and not e['validation'] for e in experts.entries) and experts.entries
    torch.manual_seed(0)
    model = policy.Policy(policy.Config(vocab_size=len(C.Vocab([26000000])), card_dim=4, object_dim=4, conv_channels=4,
                                        conv_layers=1, core=8, attention_layers=0)).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = experts.batch(2)
    first = ppo.imitation_step(model, optimizer, batch, 0.2, torch.device('cpu'))
    for _ in range(20):
        last = ppo.imitation_step(model, optimizer, batch, 0.2, torch.device('cpu'))
    assert all(math.isfinite(v) for v in {**first, **last}.values()) and last['bc_loss'] < first['bc_loss']
    assert model.next_when.weight.grad is None, 'heads the step does not train keep no gradient'
