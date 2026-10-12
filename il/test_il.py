"""Offline checks for the IL stack (no engine). Needs a small cache:

    python -m native_engine.build_cache data/IL_Replay/replays data/cache-smoke --parts 0 --limit 100
    python -m unittest il.test_il -v
"""
from pathlib import Path
import unittest

import numpy as np
import torch

from . import data as D
from .model import INPUT_KEYS, Policy

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data' / 'cache-smoke' / 'part-000000-0.npz'
PARQUET = ROOT / 'data' / 'IL_Replay' / 'replays' / 'part-000000.parquet'


def engine_running(port=26789):
    """The emulator engine in headless mode (never a live game's probe)."""
    from native_engine.emulator import read_status
    try:
        return read_status(port).get('mode') == 'headless'
    except (OSError, ValueError, RuntimeError):
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


class AddedLayers(unittest.TestCase):
    def test_zero_initialised_and_old_checkpoints_load(self):
        from .model import load_weights
        torch.manual_seed(0)
        model = Policy(40).eval()
        old = {k: v for k, v in model.state_dict().items() if not k.startswith(('extras.', 'matchup.'))}
        fresh = Policy(40).eval()
        load_weights(fresh, old)
        x = {'ent_obj': torch.randint(0, 40, (2, 3, D.MAX_ENTITIES)), 'ent_side': torch.randint(0, 3, (2, 3, D.MAX_ENTITIES)),
             'ent_type': torch.randint(0, 5, (2, 3, D.MAX_ENTITIES)), 'ent_cell': torch.randint(0, D.CELLS, (2, 3, D.MAX_ENTITIES)),
             'ent_float': torch.randn(2, 3, D.MAX_ENTITIES, D.ENTITY_FLOATS), 'rev_card': torch.randint(0, 40, (2, 3, D.REVEALED)),
             'rev_age': torch.rand(2, 3, D.REVEALED), 'scalars': torch.rand(2, 3, D.SCALARS),
             'hand_card': torch.randint(1, 40, (2, 3, 4)), 'hand_cost': torch.rand(2, 3, 4),
             'next_card': torch.randint(0, 40, (2, 3)), 'slot_ok': torch.ones(2, 3, 4, dtype=torch.bool),
             'extras': torch.rand(2, 3, D.EXTRAS), 'cycle_card': torch.randint(0, 40, (2, 3, 3)),
             'pending_card': torch.randint(0, 40, (2, 3)), 'ability_ok': torch.ones(2, 3, dtype=torch.bool),
             'opp_deck': torch.randint(0, 40, (2, 3, 8))}
        blank = dict(x, extras=torch.zeros(2, 3, D.EXTRAS), cycle_card=torch.zeros(2, 3, 3, dtype=torch.long),
                     pending_card=torch.zeros(2, 3, dtype=torch.long), opp_deck=torch.zeros(2, 3, 8, dtype=torch.long))
        with torch.no_grad():
            got, want = fresh(x), fresh(blank)
        self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]))
        with torch.no_grad():
            masked = fresh(dict(x, ability_ok=torch.zeros(2, 3, dtype=torch.bool)))[0]
        self.assertTrue(torch.isinf(masked[..., D.ABILITY]).all())


class PPODistribution(unittest.TestCase):
    """il.rl's factorised policy: wait with p(wait), else a card x cell at a temperature."""

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(3, D.ACTIONS) * 2
        self.logits[:, D.ABILITY] = float('-inf')     # the model masks the ability unless it is ready
        self.logits[0, 2 + D.CELLS:] = float('-inf')  # only hand slot 0 playable
        self.logits[1, 2:] = float('-inf')            # nothing playable: must wait

    def test_normalized_legal_and_consistent(self):
        from .rl import distribution, entropy, kl, log_prob
        dist = distribution(self.logits)
        every = torch.arange(D.ACTIONS).expand(3, -1)
        p = log_prob(tuple(t.unsqueeze(1).expand(-1, D.ACTIONS, *t.shape[1:]) if t.dim() > 1 else
                           t.unsqueeze(1).expand(-1, D.ACTIONS) for t in dist), every).exp()
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(3), atol=1e-4))
        self.assertEqual(p[0, 2 + D.CELLS:].sum().item(), 0)
        self.assertAlmostEqual(p[1, D.WAIT].item(), 1, places=5)
        direct = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(-1)
        self.assertTrue(torch.allclose(entropy(dist), direct, atol=1e-3))
        self.assertTrue(torch.allclose(kl(dist, dist), torch.zeros(3), atol=1e-5))

    def test_ready_ability_is_one_of_the_play_options(self):
        from .rl import distribution, log_prob, sample
        logits = self.logits[2:].clone()
        logits[0, D.ABILITY] = logits[0, 2:].max() + 1  # temperature 0.5: about e^2 times the best card
        dist = distribution(logits.expand(40000, -1))
        actions = sample(dist)
        p_ability = log_prob(dist, torch.full_like(actions, D.ABILITY))[0].exp().item()
        self.assertGreater(p_ability, 0.01)
        self.assertAlmostEqual((actions == D.ABILITY).float().mean().item(), p_ability, delta=0.01)

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
    """The rows imitation learning trains on: data.features on the whole game through build_cache.encode."""
    from native_engine.build_cache import encode
    raw = encode({'states': list(states), 'commands': [], 'receipts': [], 'winner_matches': 0, 'crowns_matches': 0})
    feats = D.features(raw, owner, vocab)
    return [{k: feats[k][t] for k in INPUT_KEYS} for t in range(len(states))]


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


class ObserverExtras(unittest.TestCase):
    """What the inputs added after IL say: a play in flight, the hero ability, evolution, the opponent's elixir."""

    def state(self, abilities=(), evolution=(), plays=(), tick=200):
        from native_engine.engine import HandCard, Player, State
        hand = tuple(HandCard(i, 26000000 + i, 3, 0, 26000000 + i) for i in range(4))
        players = tuple(Player(o, 60000, hand, (26000004, 26000005, 26000006, 26000007),
                               abilities if o == 0 else (), evolution if o == 0 else ()) for o in (0, 1))
        return State(1, tick, False, None, (0, 0), players, (), tuple(plays), 1)

    def test_play_in_flight_masks_everything(self):
        from native_engine.engine import Ability
        from .data import InFlight, Observer
        from .rl import distribution, log_prob, sample
        ready = (Ability(2, 0, 10000, 1, 1, (77,)),)
        row, other = Observer({}).observe(self.state(ready), [InFlight(0, 26000001, 4, 222)])
        self.assertFalse(row['slot_ok'].any() or row['ability_ok'])
        self.assertTrue(other['slot_ok'].all())
        np.testing.assert_allclose(row['extras'][:4], [1, 0.4, 1, (60000 - 40000) / 100000], rtol=1e-6)
        logits = torch.randn(1000, D.ACTIONS)
        logits[:, 1:] = float('-inf')  # what the model makes of it
        dist = distribution(logits)
        self.assertTrue((sample(dist) == D.WAIT).all())
        self.assertTrue(torch.allclose(log_prob(dist, torch.zeros(1000, dtype=torch.long)), torch.zeros(1000)))

    def test_ability_and_evolution(self):
        from native_engine.engine import Ability, Evolution
        from .data import Observer
        row = Observer({}).observe(self.state((Ability(2, 0, 10000, 2, 3, (77,)),),
                                              (Evolution(1, 26000001, True, 2, 0, True),
                                               Evolution(2, 26000002, True, 2, 1, False))))[0]
        self.assertTrue(row['ability_ok'])
        np.testing.assert_allclose(row['extras'][4:17], [1, 1, 0, 2 / 3, 0] + [0, 1, 0, 0] + [0, 1, 0.5, 0], rtol=1e-6)
        cooling = Observer({}).observe(self.state((Ability(8, 2500, 10000, 1, 1, (77,)),)))[0]
        self.assertFalse(cooling['ability_ok'])
        np.testing.assert_allclose(cooling['extras'][4:9], [1, 0, 0, 1 / 3, 0.25], rtol=1e-6)

    def test_opponent_elixir(self):
        from native_engine.engine import CardPlay
        from .data import Observer
        observer = Observer({})
        self.assertAlmostEqual(observer.observe(self.state(tick=90))[0]['extras'][-1], 0.6)
        # 56 ticks regenerate one elixir; the opponent's Musketeer (4) is spent
        row = observer.observe(self.state(plays=(CardPlay(1, 146, 1, 26000014, 4),), tick=146))[0]
        self.assertAlmostEqual(row['extras'][-1], 0.3, places=5)


class FastObserve(unittest.TestCase):
    """il.data.Observer gives the rows imitation learning trained on, byte for byte (ability_ok aside: the cache does
    not know whether the ability is ready, so it leaves it unmasked)."""

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
                    for key in set(INPUT_KEYS) - {'ability_ok'}:
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
