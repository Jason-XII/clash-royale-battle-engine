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
from .train import collate

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

    def test_model_can_memorize_a_short_game(self):
        torch.manual_seed(0)
        model = Policy(max(self.vocab.values()) + 1)
        s = D.sequence(self.shard[0], 0, self.vocab)
        T = int(np.flatnonzero(s['action'] > D.ABILITY)[4]) + 1  # first five card plays
        batch = collate([{k: v[:T] for k, v in s.items()}])
        x = {k: batch[k] for k in INPUT_KEYS}
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        for step in range(80):
            logits, _, _ = model(x)
            labeled = batch['label_mask']
            loss = torch.nn.functional.cross_entropy(logits[labeled], batch['action'][labeled])
            if step == 0:
                first = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first * 0.1)


@unittest.skipUnless(engine_running() and PARQUET.exists(), 'needs the emulator engine and IL_Replay part 0')
class EngineFeatures(unittest.TestCase):
    def test_agent_sees_what_training_sees(self):
        """Online features, built state by state while a replay runs, equal the cached ones."""
        import json
        import pyarrow.parquet as pq
        from native_engine import Engine
        from native_engine.build_cache import encode
        from native_engine.replay import rebuild
        from .agent import Agent
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
