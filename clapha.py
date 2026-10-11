"""The battle engine on x86_64 Linux through clapha-engine (the game's own libg.so loaded in-process: no
emulator). Same interface and State as Engine, so il/ agents and features run unchanged.

    engine = ClaphaEngine('~/clapha-engine')        # after its prepare.py; boots in ~8 s
    state = engine.reset(seed=1)                     # or reset(match=replay dict); tick 90, like Engine
    transition = engine.step([Play(owner, slot, x, y)], 5)

Differences from the probe, each filled to match it:
  - card costs: card_costs.json (built from the cache); variable-cost cards ask the engine's card selection
  - card plays: the probe logs them as consumed; here a card that left a hand between snapshots was played
  - projectile vs area (objects without hitpoints): by the class of the object's data global id
"""
import ctypes
import json
from pathlib import Path
import resource
import sys
import tempfile
from types import SimpleNamespace

from .engine import Transition
from .protocol import CardPlay, Entity, HandCard, Player, State

COSTS = {int(k): v for k, v in json.loads(Path(__file__).with_name('card_costs.json').read_text()).items()}
ENGINE_CPUS = 2  # what the engine is told about the machine: it sized its thread pool from all 128 CPUs (~70 threads)
WAIT_LIMIT = 60  # ticks a delayed play waits for elixir before it is dropped (ponytail: the game's own limit is unknown)
PROJECTILE_CLASS = 10  # data global id // 1e6 of LogicProjectileData (the probe checks type 0x0a); measured


def _ms(value):
    """The probe's rounding: up to a whole 50 ms tick; out of range -> unknown."""
    return None if value is None or value < 0 or value > 3600000 else (value + 49) // 50 * 50


def raise_thread_limit():
    """Each engine starts ~70 threads, and `ulimit -u` (1024 by default on the cluster) counts every thread of
    the user. Call this in the parent before starting engine processes: a process whose own soft limit stays
    low (the PPO learner) then fails to create threads ('Resource temporarily unavailable'). The soft limit
    may be raised to the hard one without privileges."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or hard > soft):
        resource.setrlimit(resource.RLIMIT_NPROC, (hard if hard == resource.RLIM_INFINITY else min(hard, 1 << 20), hard))


def _report_cpus(loader):
    """Answer the engine's 'how many CPUs' (sysconf, get_nprocs) with ENGINE_CPUS: clapha passes it through to the node,
    which has 128; each engine only ever uses one core. Registered before boot, when clapha links the engine."""
    android_nprocessors = (0x60, 0x61)  # _SC_NPROCESSORS_CONF / _ONLN in bionic's numbering
    passthrough = loader._sysconf
    loader.shim('sysconf', ctypes.c_long, ctypes.c_int)(
        lambda name: ENGINE_CPUS if name in android_nprocessors else passthrough(name))
    loader.shim('get_nprocs', ctypes.c_int)(lambda: ENGINE_CPUS)
    loader.shim('get_nprocs_conf', ctypes.c_int)(lambda: ENGINE_CPUS)


class ClaphaEngine:
    def __init__(self, clapha_dir, workdir=None):
        raise_thread_limit()  # also here, for scripts that start a single engine
        clapha = Path(clapha_dir).expanduser().resolve()
        sys.path.insert(0, str(clapha / 'crx'))
        import loader  # clapha's crx modules import each other as top-level modules
        _report_cpus(loader)
        from env import CrxEnv
        self.env = CrxEnv(clapha / 'engine', Path(workdir or tempfile.mkdtemp(prefix='clapha-')),
                          str(clapha / 'data' / 'assets'))
        self.state, self.generation, self.sequence, self.pending = None, 0, 0, []

    def reset(self, seed=1, match=None):
        if match is None:
            match = json.loads(Path(__file__).with_name('standard_match.json').read_text())
            match['rndSeed'] = seed
        self.env.create_match(match)
        self.generation += 1
        self.hands, self.pending = None, []
        r = self.env.reader
        logic = r.logic(self.env.battle)  # a deck never changes within a battle: read its card ids once
        self.deck = [[r.card_id(r.card_data(slot)) for slot in r.deck_slots(logic, owner)] for owner in (0, 1)]
        self.env.step(90 - self.env.tick())
        self.state = self._capture()
        return self.state

    def step(self, actions=(), ticks=5, delay=0):
        """Plays decided on the current state land `delay` ticks later (live: il.data.DEPLOY_DELAY), or on the
        next tick with delay=0. As in the live game, a play that is not affordable when due waits for elixir
        (up to WAIT_LIMIT ticks), and, as il.live does, a player with a play in flight cannot start another.
        `executed[i]` says whether play i was accepted; landed plays show up in later states' `plays`."""
        before = self.state
        executed = []
        for a in actions:
            card = next((h for h in before.players[a.owner].hand if h.slot == a.slot), None)
            ok = card is not None and all(p['owner'] != a.owner for p in self.pending)
            if ok:
                due = before.tick + max(delay, 1)
                self.pending.append(dict(due=due, until=due + WAIT_LIMIT, owner=a.owner, slot=a.slot,
                                         card=card.card, cost=card.cost, x=a.x, y=a.y))
            executed.append(ok)
        for _ in range(ticks):
            if self.pending:
                self._release()
            self.env.step(1)
        after = self._capture()
        self.state = after
        return Transition(after, after.tick - before.tick, tuple(executed))

    def _players(self):
        """Per owner: ({hand index: (deck slot, card id)}, cycle card ids, elixir raw), with the battle's cached deck
        (clapha's reader re-reads both decks on every call)."""
        r = self.env.reader
        f = r.f
        logic = r.logic(self.env.battle)
        out = []
        for owner in (0, 1):
            ids = self.deck[owner]
            player = r.player(logic, owner)
            data, count = r.vector(player, f['player.hand'], 8)
            hand = {}
            for i in range(count):
                slot = ctypes.c_int32.from_address(data + 4 * i).value
                if slot >= 0:
                    hand[i] = (slot, ids[slot])
            data, count = r.vector(player, f['player.cycle'], 8)
            cycle = tuple(ids[ctypes.c_int32.from_address(data + 4 * i).value] for i in range(count))
            out.append((hand, cycle, ctypes.c_int32.from_address(player + f['player.elixir_raw']).value))
        return out

    def _release(self):
        """Hand the plays due on the next tick to the engine, if their card is still there and affordable."""
        now = self.env.tick()
        due = [p for p in self.pending if p['due'] <= now + 1]
        if not due:
            return
        players = self._players()
        for p in due:
            hand, _, elixir = players[p['owner']]
            in_hand = hand.get(p['slot'], (None, None))[1] == p['card']
            if in_hand and elixir >= p['cost'] * 10000:
                self.env.queue_hand_action_at(SimpleNamespace(owner=p['owner'], hand_index=p['slot'], x=p['x'],
                                                              y=p['y']), execute_tick=now + 1)
            elif in_hand and now + 1 < p['until']:
                continue  # wait for elixir
            self.pending.remove(p)

    def run(self, ticks, count):
        states = []
        for _ in range(count):
            states.append(self.step((), ticks).state)
            if states[-1].finalized:
                break
        return states

    # ---- snapshot -> State --------------------------------------------------------------------------------------------
    def _capture(self):
        env = self.env
        objects = env.reader.objects(env.battle)
        crowns, ended, winner = env._result(objects)
        tick = env.tick()
        players, plays, hands = [], [], {}
        for owner, (hand, cycle, elixir) in enumerate(self._players()):
            hands[owner] = {i: card for i, (_, card) in hand.items()}
            players.append(Player(owner, elixir, tuple(HandCard(i, card, self._cost(owner, slot, card), 0, card)
                                                       for i, (slot, card) in sorted(hand.items())), cycle))
            if self.hands is not None:  # a card that left its slot since the last snapshot was played
                for i, card in self.hands[owner].items():
                    if hands[owner].get(i) != card:
                        self.sequence += 1
                        plays.append(CardPlay(self.sequence, tick, owner, card, COSTS.get(card, -1)))
        self.hands = hands
        entities = tuple(self._entity(o) for o in objects)
        return State(self.generation, tick, bool(ended), winner, (crowns[0], crowns[1]),
                     tuple(players), entities, tuple(plays), self.sequence + 1)

    def _cost(self, owner, deck_slot, card):
        cost = COSTS.get(card)
        if cost is None:  # variable cost (Mirror, ...) or unseen card: the engine's own selection
            chosen = self.env._selection(owner, deck_slot)
            cost = chosen[1] >> 28 if chosen else -1
        return cost

    @staticmethod
    def _entity(o):
        phase = o['phaseRuntime']
        if o['hp'] is None:
            kind = 'projectile' if o['dataGlobalId'] // 1000000 == PROJECTILE_CLASS else 'area'
            return Entity(o['nativeObjectId'], o['owner'], o['cardId'], o['x'], o['y'], None, None, None, None, kind,
                          None)
        attack = phase['attackValidated']
        target = o['targetEntityKey'][2] if attack and o['targetEntityKey'] else None
        return Entity(o['nativeObjectId'], o['owner'], o['cardId'], o['x'], o['y'], o['hp'], o['maxHp'], target,
                      _ms(phase['loadRemainingMs']) if attack else None,
                      'building' if o['cardId'] == -1 else 'troop', _ms(phase['deployRemainingMs']))
