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


@unittest.skipUnless(CACHE.exists(), 'build data/cache-smoke first')
class ValidStepsOnly(unittest.TestCase):
    """Policy.forward(valid=mask), as the PPO update runs it, gives the valid steps' outputs and gradients unchanged."""

    def test_same_outputs_and_gradients(self):
        from .train import collate
        torch.manual_seed(0)
        shard = D.Shard(CACHE)
        vocab = D.build_vocab([CACHE])
        model = Policy(max(vocab.values()) + 1)
        for layer in (model.extras, model.matchup[-1]):  # not zero, so the added inputs are exercised too
            torch.nn.init.normal_(layer.weight, std=0.1)
        seqs = [D.sequence(shard[g], g % 2, vocab) for g in range(3)]
        batch = collate([{k: s[k][:n] for k in INPUT_KEYS + ('action',)} for s, n in zip(seqs, (40, 25, 33))])
        x, m = {k: batch[k] for k in INPUT_KEYS}, batch['time_mask']
        outs, grads = [], []
        for valid in (None, m):
            model.zero_grad()
            logits, value, _ = model(x, valid=valid)
            finite = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
            (finite[m].sum() + value[m].sum()).backward()
            outs.append((logits[m], value[m]))
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        self.assertTrue(torch.allclose(outs[0][0], outs[1][0], atol=1e-5, equal_nan=True))
        self.assertTrue(torch.allclose(outs[0][1], outs[1][1], atol=1e-5))
        self.assertTrue(torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-5))


class PPODistribution(unittest.TestCase):
    """il.rl's factorised policy: wait with p(wait), else a card x cell at a temperature."""

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(3, D.ACTIONS) * 2
        self.logits[:, D.ABILITY] = float('-inf')     # the model masks the ability unless it is ready
        self.logits[0, 2 + D.CELLS:] = float('-inf')  # only hand slot 0 playable
        self.logits[1, 2:] = float('-inf')            # nothing playable: must wait

    def test_normalized_legal_and_consistent(self):
        from .model import distribution, entropy, kl, log_prob
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
        from .model import distribution, log_prob, sample
        logits = self.logits[2:].clone()
        logits[0, D.ABILITY] = logits[0, 2:].max() + 1  # temperature 0.5: about e^2 times the best card
        dist = distribution(logits.expand(40000, -1))
        actions = sample(dist)
        p_ability = log_prob(dist, torch.full_like(actions, D.ABILITY))[0].exp().item()
        self.assertGreater(p_ability, 0.01)
        self.assertAlmostEqual((actions == D.ABILITY).float().mean().item(), p_ability, delta=0.01)

    def test_ability_exploration(self):
        """explore mixes "press the ability" into the whole distribution where it is ready, and only there."""
        from .model import distribution, entropy, log_prob, sample
        logits = self.logits.clone()
        logits[2, D.ABILITY] = -30.0  # ready, but the model would almost never press it
        dist = distribution(logits, explore=0.05)
        every = torch.arange(D.ACTIONS).expand(3, -1)
        p = log_prob(tuple(t.unsqueeze(1).expand(-1, D.ACTIONS, *t.shape[1:]) if t.dim() > 1 else
                           t.unsqueeze(1).expand(-1, D.ACTIONS) for t in dist), every).exp()
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(3), atol=1e-4))
        self.assertGreaterEqual(p[2, D.ABILITY].item(), 0.05 - 1e-6)
        plain = distribution(logits)
        for a, b in zip(dist, plain):  # rows 0 and 1: the ability is masked, nothing changes
            self.assertTrue(torch.allclose(a[:2], b[:2]))
        actions = sample(distribution(logits[2:].expand(20000, -1), explore=0.05))
        self.assertAlmostEqual((actions == D.ABILITY).float().mean().item(), p[2, D.ABILITY].item(), delta=0.01)
        self.assertTrue(torch.isfinite(entropy(dist)).all())

    def test_sampling_matches_probabilities(self):
        from .model import distribution, log_prob, sample
        dist = distribution(self.logits[2:].expand(20000, -1))
        actions = sample(dist)
        p_wait = log_prob(dist, torch.zeros_like(actions))[0].exp().item()
        self.assertAlmostEqual((actions == D.WAIT).float().mean().item(), p_wait, delta=0.015)
        self.assertFalse((actions == D.ABILITY).any())


@unittest.skipUnless(CACHE.exists(), 'build data/cache-smoke first')
class PPOServer(unittest.TestCase):
    """il.actors.Server keeps every seat's LSTM state apart: interleaved seats see what one full-sequence pass sees."""

    def test_interleaved_seats_match_full_sequences(self):
        from .actors import Server, pack
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
        from .actors import Server, pack
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
        from . import actors as rl
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
    feats = D.features(raw, owner, vocab, online=True)
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
        from .model import distribution, log_prob, sample
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
            cached = D.features(raw, owner, vocab, online=True)
            agent = Agent(None, vocab, owner)
            for t, state in enumerate(result['states'][:400]):
                online = agent.observe(state)
                for key in set(INPUT_KEYS) - {'ability_ok', 'extras'}:  # ability_ok: unknown in the cache, unmasked
                    np.testing.assert_allclose(online[key][0, 0].numpy(), cached[key][t], err_msg=f'{key} t={t}')
                # extras: the cache has no ability or evolution state (the probe reports them now); compare the rest
                known = [0, 1, 2, 3, D.EXTRAS - 1]
                np.testing.assert_allclose(online['extras'][0, 0, known].numpy(), cached['extras'][t, known],
                                           err_msg=f'extras t={t}')


class Shaping(unittest.TestCase):
    """FirstLight's tower potential: 0 at the start, damage counts for the dealer and against the owner, a destroyed
    tower counts its whole hp plus the crown, the king x3; and il.defense finds an unanswered push and its cost."""

    @staticmethod
    def state(hp, crowns=(0, 0)):
        from native_engine.engine import Entity, State
        towers = [(1, 0, 9000, 3000, 7728), (2, 0, 3500, 6500, 4858), (3, 1, 9000, 29000, 7728), (4, 1, 3500, 25500, 4858)]
        entities = tuple(Entity(i, o, -1, x, y, hp.get(i, full), full, None, None, 'building', None)
                         for i, o, x, y, full in towers if hp.get(i, full) is not None)
        return State(0, 0, False, None, crowns, (), entities, (), 0)

    def test_potential(self):
        from .actors import potential, tower_hp
        start = tower_hp(self.state({}))
        self.assertEqual(potential(self.state({}), start, 0), 0.0)
        half = self.state({4: 4858 // 2})  # owner 1's princess at half hp
        self.assertAlmostEqual(potential(half, start, 0), 0.5, 3)
        self.assertAlmostEqual(potential(half, start, 1), -0.5, 3)
        gone = self.state({4: None}, crowns=(1, 0))  # destroyed: its hp and the crown
        self.assertAlmostEqual(potential(gone, start, 0), 2.0)
        king = self.state({3: 7728 // 2})
        self.assertAlmostEqual(potential(king, start, 0), 1.5, 3)

    def test_unanswered_push(self):
        from . import defense
        tower = lambda hp: [0, -1, 3500, 6500, hp, 4858, 1]
        troop = lambda y: [1, 26000000, 3500, y, 1500, 1500, 0]  # an enemy troop in the left lane, our half below 15000
        frames = [(t, np.array([tower(4858 - max(0, t - 20) * 10)] + ([troop(10000)] if 10 <= t <= 50 else []), np.int64))
                  for t in range(0, 100, 5)]
        ours = defense.game(frames, [])[0]
        self.assertEqual(len(ours['pushes']), 1)
        push = ours['pushes'][0]
        self.assertFalse(push['answered'])
        self.assertEqual(push['damage'], 10 * (90 - 20))  # damage until the push ended + GAP
        answered = defense.game(frames, [(30, 0, 4000, 8000)])[0]['pushes'][0]  # a play landing in the left lane
        self.assertTrue(answered['answered'])
        self.assertEqual(answered['latency'], 20)
        wrong_lane = defense.game(frames, [(30, 0, 14000, 8000)])[0]['pushes'][0]
        self.assertFalse(wrong_lane['answered'])


class Matchmaking(unittest.TestCase):
    """The specialist's game specs: our deck in the learning seat, a frozen opponent on its home server, and a deck we
    keep losing to becomes a hard deck (until its results decay away)."""

    def test_specialist_specs(self):
        import argparse
        from .rl import Matchmaker, deck_key
        ours = [{'d': 26000043, 'el': 1}] + [{'d': 26000000 + i} for i in range(7)]
        ladder = [[{'d': 27000000 + 10 * j + i} for i in range(8)] for j in range(30)]
        pairs = [{'decks': [ladder[j], ladder[(j + 1) % 30]]} for j in range(30)]
        args = argparse.Namespace(self_play=0.0, il_share=0.4, pool_size=6, servers=4, deck_mix='0,0,1',
                                  shaping=0.1, gamma=0.999)
        m = Matchmaker(args, pairs, ours, ['il', 'update-00150', 'update-00300'])
        for _ in range(20):
            spec = m.spec()
            seat = spec['train'][0]
            self.assertEqual(spec['decks']['decks'][seat], ours)
            opponent = spec['models'][1 - seat]
            self.assertIn(opponent, m.frozen)
            self.assertEqual(spec['server'], m.frozen.index(opponent) % 4)
            self.assertEqual(spec['tag'], deck_key(spec['decks']['decks'][1 - seat]))
        lost_to = deck_key(ladder[3])
        self.assertEqual(m.hard_decks(), [])
        for _ in range(3):
            m.record({'opponent': 'il', 'outcome': -1.0, 'tag': lost_to})
        self.assertEqual(m.hard_decks(), [lost_to])
        self.assertTrue(all(m.spec()['tag'] == lost_to for _ in range(10)))  # deck mix: hard decks only
        for _ in range(5):
            m.decay()
        self.assertEqual(m.hard_decks(), [])  # 3 * 0.8^5 < 1 game


if __name__ == '__main__':
    unittest.main()
