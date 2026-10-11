"""Offline checks for the IL stack (no engine). Needs a small cache:

    python -m native_engine.build_cache data/IL_Replay/replays data/cache-smoke --parts 0 --limit 100
    python -m unittest il.test_il -v
"""
from pathlib import Path
import socket
import unittest

import numpy as np
import torch

from . import data as D
from .model import INPUT_KEYS, Policy

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data' / 'cache-smoke' / 'part-000000-0.npz'
PARQUET = ROOT / 'data' / 'IL_Replay' / 'replays' / 'part-000000.parquet'


def engine_running(port=26789):
    try:
        socket.create_connection(('127.0.0.1', port), timeout=1).close()
        return True
    except OSError:
        return False


@unittest.skipUnless(CACHE.exists(), 'build data/cache-smoke first')
class IL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.vocab = D.build_vocab([CACHE])
        cls.shard = D.Shard(CACHE)

    def test_cells_round_trip_for_both_owners(self):
        cells = np.arange(D.CELLS)
        for owner in (0, 1):
            x, y = D.cell_center(cells, owner)
            np.testing.assert_array_equal(D.cell_of(*D.rotate(x, y, owner)), cells)

    def test_both_players_defend_the_bottom(self):
        tower = self.vocab[-1]
        for owner in (0, 1):
            s = D.sequence(self.shard[0], owner, self.vocab)
            towers = s['ent_obj'][0] == tower
            own_rows = s['ent_cell'][0][towers & (s['ent_side'][0] == 1)] // D.W
            enemy_rows = s['ent_cell'][0][towers & (s['ent_side'][0] == 2)] // D.W
            self.assertTrue((own_rows < 16).all() and (enemy_rows >= 16).all())

    def test_labels_are_legal_and_mostly_kept(self):
        acts = kept = 0
        for i in range(len(self.shard)):
            replay = self.shard[i]
            for owner in (0, 1):
                s = D.sequence(replay, owner, self.vocab)
                card = (s['action'] > D.ABILITY) & s['label_mask']
                slot = (s['action'][card] - 2) // D.CELLS
                self.assertTrue(s['slot_ok'][np.flatnonzero(card), slot].all())
                acts += (replay['actions'][:, 1] == owner).sum()
                kept += ((s['action'] != D.WAIT) & s['label_mask']).sum()
                # Each label sits on the last board at least DEPLOY_DELAY ticks before execution.
                first = replay['actions'][(replay['actions'][:, 1] == owner) & (replay['actions'][:, 5] == 1)]
                labeled = np.flatnonzero(s['action'] != D.WAIT)
                if len(first) and len(labeled):
                    tick = replay['tick'][labeled[0]]
                    self.assertTrue(tick < first[0, 0] - D.DEPLOY_DELAY <= tick + 5)
        self.assertGreater(kept / acts, 0.95)

    def test_stepwise_equals_sequence(self):
        model = Policy(max(self.vocab.values()) + 1).eval()
        s = D.sequence(self.shard[0], 1, self.vocab)
        x = {k: torch.as_tensor(s[k][:6])[None] for k in INPUT_KEYS}
        with torch.no_grad():
            full, _, _ = model(x)
            state, steps = None, []
            for t in range(6):
                logits, _, state = model({k: v[:, t:t + 1] for k, v in x.items()}, state)
                steps.append(logits)
        self.assertTrue(torch.allclose(torch.cat(steps, 1), full, atol=1e-5))


class PPODistribution(unittest.TestCase):
    """il.rl's factorised policy: wait with p(wait), else a card x cell at a temperature."""

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(3, D.ACTIONS) * 2
        self.logits[0, 2 + D.CELLS:] = float('-inf')  # only hand slot 0 playable
        self.logits[1, 2:] = float('-inf')            # nothing playable: must wait

    def test_normalized_legal_and_consistent(self):
        from .rl import distribution, entropy, kl, log_prob
        dist = distribution(self.logits)
        every = torch.arange(D.ACTIONS).expand(3, -1)
        p = log_prob(tuple(t.unsqueeze(1).expand(-1, D.ACTIONS, *t.shape[1:]) if t.dim() > 1 else
                           t.unsqueeze(1).expand(-1, D.ACTIONS) for t in dist), every).exp()
        p[:, D.ABILITY] = 0  # ability is never chosen
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(3), atol=1e-4))
        self.assertEqual(p[0, 2 + D.CELLS:].sum().item(), 0)
        self.assertAlmostEqual(p[1, D.WAIT].item(), 1, places=5)
        direct = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(-1)
        self.assertTrue(torch.allclose(entropy(dist), direct, atol=1e-3))
        self.assertTrue(torch.allclose(kl(dist, dist), torch.zeros(3), atol=1e-5))

    def test_play_in_flight_forces_wait(self):
        from .rl import distribution, hold_while_in_flight, log_prob, sample
        rows = [{'slot_ok': np.ones(4, bool)} for _ in (0, 1)]
        rows = hold_while_in_flight(rows, [{'owner': 0}])
        self.assertFalse(rows[0]['slot_ok'].any())
        self.assertTrue(rows[1]['slot_ok'].all())
        logits = torch.randn(1, D.ACTIONS)
        logits[0, 2:] = float('-inf')                 # what the model's mask makes of slot_ok all False
        dist = distribution(logits.expand(1000, -1))
        self.assertTrue((sample(dist) == D.WAIT).all())
        self.assertTrue(torch.allclose(log_prob(dist, torch.zeros(1000, dtype=torch.long)), torch.zeros(1000)))

    def test_sampling_matches_probabilities(self):
        from .rl import distribution, log_prob, sample
        dist = distribution(self.logits[2:].expand(20000, -1))
        actions = sample(dist)
        p_wait = log_prob(dist, torch.zeros_like(actions))[0].exp().item()
        self.assertAlmostEqual((actions == D.WAIT).float().mean().item(), p_wait, delta=0.015)
        self.assertFalse((actions == D.ABILITY).any())


@unittest.skipUnless(CACHE.exists(), 'build data/cache-smoke first')
class PPOServer(unittest.TestCase):
    """il.rl.Server keeps every seat's LSTM state apart: interleaved seats see what one full-sequence pass sees."""

    def test_interleaved_seats_match_full_sequences(self):
        from .rl import Server, pack
        torch.manual_seed(0)
        shard = D.Shard(CACHE)
        vocab = D.build_vocab([CACHE])
        model = Policy(max(vocab.values()) + 1).eval()
        seqs = [D.sequence(shard[g], g % 2, vocab) for g in range(3)]
        T = 12
        with torch.no_grad():
            full = [model({k: torch.as_tensor(s[k][:T])[None] for k in INPUT_KEYS})[1][0] for s in seqs]
        server = Server({'current': model}, torch.device('cpu'), model.config['core'], capacity=1)  # grows
        got = [[] for _ in seqs]
        for t in range(T):
            order = [2, 0, 1] if t % 2 else [0, 1, 2]  # requests arrive in varying order
            reqs = [((g, 0), 'current', pack({k: seqs[g][k][t] for k in INPUT_KEYS})) for g in order]
            for g, (_, _, value) in zip(order, server.act(reqs)):
                got[g].append(value)
        for g in range(3):
            self.assertTrue(torch.allclose(torch.tensor(got[g]), full[g], atol=1e-4))
        server.reset([(0, 0)])  # a new game for seat 0 starts from zero state again
        _, _, value = server.act([((0, 0), 'current', pack({k: seqs[0][k][0] for k in INPUT_KEYS}))])[0]
        self.assertAlmostEqual(value, full[0][0].item(), places=4)


    def test_mixed_models_in_one_round(self):
        """Requests for two models in one round: each model gets exactly its own rows."""
        from .rl import Server, pack
        torch.manual_seed(1)
        shard = D.Shard(CACHE)
        vocab = D.build_vocab([CACHE])
        models = {'a': Policy(max(vocab.values()) + 1).eval(), 'b': Policy(max(vocab.values()) + 1).eval()}
        seqs = [D.sequence(shard[g], g % 2, vocab) for g in range(4)]
        keys = ['a', 'b', 'b', 'a']
        T = 8
        with torch.no_grad():
            full = [models[keys[g]]({k: torch.as_tensor(s[k][:T])[None] for k in INPUT_KEYS})[1][0] for g, s in enumerate(seqs)]
        server = Server(models, torch.device('cpu'), models['a'].config['core'])
        got = [[] for _ in seqs]
        for t in range(T):
            reqs = [((g, 0), keys[g], pack({k: seqs[g][k][t] for k in INPUT_KEYS})) for g in (3, 0, 2, 1)]
            for g, (_, _, value) in zip((3, 0, 2, 1), server.act(reqs)):
                got[g].append(value)
        for g in range(4):
            self.assertTrue(torch.allclose(torch.tensor(got[g]), full[g], atol=1e-4))

    def test_silent_peer_times_out(self):
        from . import rl
        old, rl.REPLY_TIMEOUT = rl.REPLY_TIMEOUT, 0.2
        try:
            a, b = __import__('multiprocessing').Pipe()
            with self.assertRaises(EOFError):
                rl.receive(a)
        finally:
            rl.REPLY_TIMEOUT = old

def _reference_rows(states, owner, vocab):
    """The rows Agent.observe built before il.data.Observer: a [previous, state] window through build_cache.encode and
    data.features, with every play so far."""
    from native_engine.build_cache import encode
    previous, plays, out = None, [], []
    for state in states:
        raw = encode({'states': [previous or state, state], 'commands': [], 'receipts': [], 'winner_matches': 0,
                      'crowns_matches': 0})
        plays += [(1, p.tick, p.owner, p.card) for p in state.plays]
        raw['plays'] = np.array(plays, np.int32).reshape(-1, 4)
        previous = state
        feats = D.features(raw, owner, vocab)
        out.append({k: feats[k][1] for k in INPUT_KEYS})
    return out


def _random_states(seed, steps=300):
    """States with what games have: units appearing, moving and dying, ids reused within a state, unknown values
    (None), more than MAX_ENTITIES objects, both players' plays, short cycles, empty hand slots."""
    import random
    from native_engine.engine import CardPlay, Entity, HandCard, Player, State
    rng = random.Random(seed)
    cards = [26000000 + i for i in range(30)] + [28000000 + i for i in range(5)]
    alive, next_id, seq, states = {}, 1, 0, []
    for t in range(steps):
        tick = 90 + 5 * t
        for _ in range(rng.randrange(0, 4)):
            alive[next_id] = [rng.randrange(2), rng.choice(cards + [-1]), rng.randrange(18000), rng.randrange(32000)]
            next_id += 1
        for k in list(alive):
            if rng.random() < 0.03:
                del alive[k]
            else:
                alive[k][2] = min(17999, max(0, alive[k][2] + rng.randrange(-900, 900)))
                alive[k][3] = min(31999, max(0, alive[k][3] + rng.randrange(-900, 900)))
        ents = []
        for k, (o, c, x, y) in list(alive.items())[:90]:
            hp = None if rng.random() < 0.1 else rng.randrange(0, 3000)
            ents.append(Entity(k, o, c, x, y, hp, None if hp is None else rng.randrange(1, 4000),
                               rng.choice([None, rng.randrange(1, next_id)]), rng.choice([None, 0, rng.randrange(6000)]),
                               rng.choice(['troop', 'building', 'projectile', 'area']),
                               rng.choice([None, 0, rng.randrange(6000)])))
        if ents and rng.random() < 0.2:
            ents.append(ents[0])                      # the same id twice in one state
        plays = []
        for o in (0, 1):
            if rng.random() < 0.15:
                seq += 1
                plays.append(CardPlay(seq, tick, o, rng.choice(cards), rng.randrange(1, 8)))
        players = tuple(Player(o, rng.randrange(0, 100001),
                               tuple(HandCard(i, rng.choice(cards), rng.choice([-1, 1, 3, 9]), 0, 0)
                                     for i in range(4) if rng.random() < 0.9),
                               tuple(rng.choice(cards) for _ in range(rng.randrange(0, 6)))) for o in (0, 1))
        states.append(State(1, tick, False, None, (rng.randrange(4), rng.randrange(4)), players, tuple(ents),
                            tuple(plays), seq + 1))
    return states


class FastObserve(unittest.TestCase):
    """il.data.Observer gives the rows the window-and-encode path gave, byte for byte."""

    def test_same_rows_as_the_reference(self):
        from .data import Observer
        cards = [26000000 + i for i in range(30)] + [28000000 + i for i in range(5)]
        vocab = {c: i + 2 for i, c in enumerate(cards[::2])}       # half the ids unknown (-> 1)
        for seed in range(4):
            states = _random_states(seed)
            observer = Observer(vocab)
            fast = [observer.observe(s) for s in states]
            for owner in (0, 1):
                for t, ref in enumerate(_reference_rows(states, owner, vocab)):
                    for key in INPUT_KEYS:
                        got = fast[t][owner][key]
                        self.assertEqual(got.dtype, ref[key].dtype, f'{key} dtype')
                        self.assertEqual(got.shape, ref[key].shape, f'{key} shape')
                        self.assertEqual(got.tobytes(), ref[key].tobytes(), f'seed {seed} owner {owner} t={t} {key}')


@unittest.skipUnless(engine_running() and PARQUET.exists(), 'needs the emulator engine and IL_Replay part 0')
class EngineFeatures(unittest.TestCase):
    def test_agent_sees_what_training_sees(self):
        """Online features, built state by state while a replay runs, equal the cached ones."""
        import json
        import pyarrow.parquet as pq
        from native_engine import Engine
        from native_engine.build_cache import encode, rebuild
        from .model import Agent
        row = pq.read_table(PARQUET, columns=['payload_json']).slice(0, 1).to_pylist()[0]
        result = rebuild(Engine(), json.loads(row['payload_json']))
        raw = encode(result)
        vocab = D.build_vocab([CACHE]) if CACHE.exists() else {}
        for owner in (0, 1):
            cached = D.features(raw, owner, vocab)
            agent = Agent(None, vocab, owner)
            for t, state in enumerate(result['states'][:400]):
                online = agent.observe(state)
                for key in INPUT_KEYS:
                    np.testing.assert_allclose(online[key][0, 0].numpy(), cached[key][t], err_msg=f'{key} t={t}')


if __name__ == '__main__':
    unittest.main()
