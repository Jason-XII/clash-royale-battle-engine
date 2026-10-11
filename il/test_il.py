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
        from .rl import Server
        torch.manual_seed(0)
        shard = D.Shard(CACHE)
        vocab = D.build_vocab([CACHE])
        model = Policy(max(vocab.values()) + 1).eval()
        seqs = [D.sequence(shard[g], g % 2, vocab) for g in range(3)]
        T = 12
        with torch.no_grad():
            full = [model({k: torch.as_tensor(s[k][:T])[None] for k in INPUT_KEYS})[1][0] for s in seqs]
        server = Server({'current': model}, torch.device('cpu'), model.config['core'])
        got = [[] for _ in seqs]
        for t in range(T):
            order = [2, 0, 1] if t % 2 else [0, 1, 2]  # requests arrive in varying order
            reqs = [((g, 0), 'current', {k: seqs[g][k][t] for k in INPUT_KEYS}) for g in order]
            for g, (_, _, value) in zip(order, server.act(reqs)):
                got[g].append(value)
        for g in range(3):
            self.assertTrue(torch.allclose(torch.tensor(got[g]), full[g], atol=1e-4))
        server.reset([(0, 0)])  # a new game for seat 0 starts from zero state again
        _, _, value = server.act([((0, 0), 'current', {k: seqs[0][k][0] for k in INPUT_KEYS})])[0]
        self.assertAlmostEqual(value, full[0][0].item(), places=4)


    def test_mixed_models_in_one_round(self):
        """Requests for two models in one round: each model gets exactly its own rows."""
        from .rl import Server
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
            reqs = [((g, 0), keys[g], {k: seqs[g][k][t] for k in INPUT_KEYS}) for g in (3, 0, 2, 1)]
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
