"""The battle engine on x86_64 Linux through clapha-engine (the game's own libg.so loaded in-process: no
emulator). Same interface and State as Engine, so il/ agents and features run unchanged.

    engine = ClaphaEngine('~/clapha-engine')        # after its prepare.py; boots in ~8 s
    state = engine.reset(seed=1)                     # or reset(match=replay dict); tick 90, like Engine
    transition = engine.step([Play(owner, slot, x, y)], 5)

Differences from the probe, each filled to match it:
  - card costs: COSTS (from the cache); variable-cost cards ask the engine's card selection
  - card plays: the probe logs them as consumed; here a card that left a hand between snapshots was played
  - projectile vs area (objects without hitpoints): by the class of the object's data global id
"""
import ctypes
from pathlib import Path
import resource
import sys
import tempfile
from types import SimpleNamespace

from .engine import ABILITY, Transition, standard_match
from .engine import Ability, CardPlay, Entity, Evolution, HandCard, Player, State

# Elixir cost of every fixed-cost card in the IL cache (Mirror 28000006 and 28000025 vary: the engine is asked).
COSTS = {
    26000000: 3, 26000001: 3, 26000002: 2, 26000003: 5, 26000004: 7, 26000005: 3, 26000006: 5, 26000007: 5,
    26000008: 5, 26000009: 8, 26000010: 1, 26000011: 4, 26000012: 3, 26000013: 2, 26000014: 4, 26000015: 4,
    26000016: 5, 26000017: 5, 26000018: 4, 26000019: 2, 26000020: 6, 26000021: 4, 26000022: 5, 26000023: 3,
    26000024: 6, 26000025: 3, 26000026: 3, 26000027: 4, 26000028: 9, 26000029: 7, 26000030: 1, 26000031: 1,
    26000032: 3, 26000033: 6, 26000034: 5, 26000035: 4, 26000036: 4, 26000037: 4, 26000038: 2, 26000039: 3,
    26000040: 3, 26000041: 3, 26000042: 4, 26000043: 6, 26000044: 4, 26000045: 5, 26000046: 3, 26000047: 7,
    26000048: 4, 26000049: 2, 26000050: 3, 26000051: 5, 26000052: 4, 26000053: 5, 26000054: 5, 26000055: 7,
    26000056: 3, 26000057: 4, 26000058: 2, 26000059: 5, 26000060: 6, 26000061: 3, 26000062: 4, 26000063: 5,
    26000064: 3, 26000065: 4, 26000067: 3, 26000068: 4, 26000069: 4, 26000072: 5, 26000074: 4, 26000077: 5,
    26000080: 4, 26000083: 4, 26000084: 1, 26000085: 7, 26000087: 4, 26000093: 3, 26000095: 4, 26000096: 5,
    26000097: 2, 26000099: 5, 26000101: 4, 26000102: 2, 26000103: 6, 26000106: 5, 27000000: 3, 27000001: 4,
    27000002: 4, 27000003: 5, 27000004: 4, 27000005: 6, 27000006: 4, 27000007: 6, 27000008: 6, 27000009: 3,
    27000010: 4, 27000012: 4, 27000013: 4, 28000000: 4, 28000001: 3, 28000002: 2, 28000003: 6, 28000004: 3,
    28000005: 4, 28000007: 6, 28000008: 2, 28000009: 4, 28000010: 5, 28000011: 2, 28000012: 3, 28000013: 3,
    28000014: 3, 28000015: 2, 28000016: 1, 28000017: 2, 28000018: 3, 28000023: 5, 28000024: 2, 28000026: 3,
}
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


_EMPTY = {}


def _ms_s(value, cap):
    """Milliseconds -> seconds, clipped to [0, cap]; None -> 0."""
    return 0.0 if value is None else min(max(float(value), 0.0), cap * 1000.0) / 1000.0


def unit_state_tuple(u):
    """One object's Reader.unit_states() dict -> il.data.UNIT_STATE_NAMES values, then the aim's world x, y (None, None
    when it has no aim). Kinds: 'unit' (troops, buildings, towers), 'area' (spell areas), 'projectile', 'timed' (fuses)."""
    kind = u.get('kind')
    buffs = u.get('buffs')
    stops_move = stops_attack = False
    stopped = speed = hit = 0
    if buffs:  # most units have none
        for b in buffs:
            m, a = b.get('stopsMovement'), b.get('stopsAttacks')
            stops_move = stops_move or bool(m)
            stops_attack = stops_attack or bool(a)
            if m or a:
                stopped = max(stopped, b.get('remainingMs') or 0)
            speed += b.get('speedPercent') or 0
            hit += b.get('hitSpeedPercent') or 0
    attack = u.get('attack') or _EMPTY
    ability = u.get('ability') or _EMPTY
    shot = attack.get('returnShot')
    hit_speed = attack.get('hitSpeedMs') or 0
    timer = (u.get('remainingMs') if kind == 'area' else u.get('fuseMs') if kind == 'timed' else u.get('spawnTimerMs'))
    values = (
        float(bool(u.get('hidden'))), float(bool(u.get('immune'))), float(bool(u.get('invisible'))),
        float(bool(u.get('airborne'))),
        float(stops_move), float(stops_attack),
        _ms_s(stopped, 5),
        min(max(speed / 100.0, -1.0), 2.0),
        min(max(hit / 100.0, -1.0), 2.0),
        _ms_s(u.get('pushRemainingMs'), 5) if u.get('pushed') else 0.0,
        min(max((attack.get('cycleMs') or 0) / hit_speed, 0.0), 1.0) if hit_speed > 0 else 0.0,
        _ms_s(attack.get('loadMs'), 5),
        min(max(float(u.get('charge') or 0) / 10000.0, 0.0), 1.0),
        _ms_s(attack.get('dashCooldownMs'), 5),
        float(ability.get('phase') is not None),
        _ms_s(timer, 10),
        0.0, 0.0,  # aim, filled per seat by il.data.Observer
        float(bool(ability.get('used'))),
        min(max((u.get('shield') or 0) / u['shieldMax'], 0.0), 1.0) if u.get('shieldMax') else 0.0,
        min(max(attack.get('stage') or 0, 0), 2) / 2.0,  # damage ramp (Inferno Tower / Dragon, Mighty Miner): 0, 0.5, 1
        float(u.get('state') == 3),  # dashing: Bandit / Boss Bandit dash, Mega Knight jump, Golden Knight's chain
        float(u.get('state') == 5),  # jumping the river (Hog Rider, Royal Hogs)
        _ms_s(ability.get('sinceUseMs'), 10),  # time since this unit's last ability use (0 before any: see ability_used)
        min(max((ability.get('uses') or 0) / ability['maxCharges'], 0.0), 1.0) if ability.get('maxCharges') else 0.0,
        min(max(float(u.get('z') or 0) / 3000.0, 0.0), 2.0),  # height: Mega Knight's jump (3000), a unit knocked up
        _ms_s(u.get('jumpRemainingMs'), 2),
        min(max(shot['elapsedMs'] / shot['totalMs'], 0.0), 1.0) if shot else 0.0,  # Executioner's axe / Fisherman's hook out
        _ms_s(attack.get('specialCooldownMs'), 5),  # the Fisherman's countdown before the hook
        _ms_s(u.get('invisibleAgainMs'), 5),  # Royal Ghost: time until invisible again
        min(float(attack.get('dashHop') or 0) / 3.0, 1.0),  # Golden Knight's hop in its dash chain
    )
    # where it is heading: a projectile's aim, or the point an underground Miner / Goblin Drill / Mighty Miner comes up
    if kind == 'projectile' and u.get('aimX') is not None:
        aim = (u['aimX'], u['aimY'])
    elif u.get('surfaceAt'):
        aim = tuple(u['surfaceAt'])
    else:
        aim = (None, None)
    # which effects are on it (data ids, longest time left first; il.data.UNIT_EFFECTS of them, 0 = none)
    ids = [b.get('globalId') or 0 for b in sorted(u.get('buffs') or (), key=lambda b: -(b.get('remainingMs') or 0))][:3]
    return values + aim + tuple(ids + [0] * (3 - len(ids)))


class ClaphaEngine:
    def __init__(self, clapha_dir, workdir=None, unit_states=False):
        raise_thread_limit()  # also here, for scripts that start a single engine
        clapha = Path(clapha_dir).expanduser().resolve()
        sys.path.insert(0, str(clapha / 'crx'))
        import loader  # clapha's crx modules import each other as top-level modules
        _report_cpus(loader)
        from env import CrxEnv
        self.env = CrxEnv(clapha / 'engine', Path(workdir or tempfile.mkdtemp(prefix='clapha-')),
                          str(clapha / 'data' / 'assets'))
        self.state, self.generation, self.sequence, self.pending = None, 0, 0, []
        self.unit_states = unit_states  # read clapha-engine's unit states into Entity.states (~+0.2 ms a step)
        self.ability_clock = None

    def reset(self, seed=1, match=None):
        if match is None:
            match = standard_match(seed)
        self.env.create_match(match)
        self.generation += 1
        self.hands, self.pending = None, []
        # time since each hero / champion unit used its ability (the engine keeps no such timer); placed at the first
        # snapshot that shows the use (up to one decision late) unless the ability has a cooldown (then exact)
        observe = sys.modules.get(type(self.env.reader).__module__)
        self.ability_clock = observe.AbilityClock() if hasattr(observe, 'AbilityClock') else None
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
            free = all(p['owner'] != a.owner for p in self.pending)
            due = before.tick + max(delay, 1)
            if a.slot == ABILITY:  # a tap on the ability button: lands like a card, if a carrier is still ready then
                ok = free and any(x.ready for x in before.players[a.owner].abilities)
                if ok:
                    self.pending.append(dict(due=due, until=due, owner=a.owner, slot=ABILITY, card=None, cost=0))
                executed.append(ok)
                continue
            card = next((h for h in before.players[a.owner].hand if h.slot == a.slot), None)
            ok = card is not None and free
            if ok:
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
            if p['slot'] == ABILITY:
                self.pending.remove(p)
                self._activate(p['owner'])
                continue
            hand, _, elixir = players[p['owner']]
            in_hand = hand.get(p['slot'], (None, None))[1] == p['card']
            if in_hand and elixir >= p['cost'] * 10000:
                self.env.queue_hand_action_at(SimpleNamespace(owner=p['owner'], hand_index=p['slot'], x=p['x'],
                                                              y=p['y']), execute_tick=now + 1)
            elif in_hand and now + 1 < p['until']:
                continue  # wait for elixir
            self.pending.remove(p)

    def _activate(self, owner):
        """Activate the first ready hero / champion ability of `owner`, if any (a press on a button no longer ready
        does nothing, as in the game)."""
        r = self.env.reader
        for row in r.abilities(r.player(r.logic(self.env.battle), owner)):
            if row['buttonState'] in (2, 4) and row['remainingCooldownMs'] == 0 and row['remainingChargesRaw'] != 0:
                for unit in row['championObjectIds']:
                    try:
                        self.env.queue_ability_action_at(SimpleNamespace(owner=owner, secondary_index=unit))
                        return True
                    except RuntimeError:  # that carrier died or is not the one ready
                        continue
        return False

    def _runtime(self, owner):
        """Hero / champion controllers (every capture) and evolution progress (cached until a card is played)."""
        r = self.env.reader
        player = r.player(r.logic(self.env.battle), owner)
        abilities = tuple(Ability(a['buttonState'], a['remainingCooldownMs'], a['configuredCooldownMs'],
                                  a['remainingChargesRaw'], a['maxCharges'] or 0, tuple(a['championObjectIds']))
                          for a in r.abilities(player))
        return abilities

    def _evolution(self):
        return [tuple(Evolution(e['deckSlot'], e['cardId'], e['evolvable'], e['cycleRequired'] or 0,
                                e['cycleRemaining'] or 0, bool(e['ready'])) for e in p['evolutionRuntime'])
                for p in self.env.reader.players_lean(self.env.battle)]

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
        if hasattr(env.reader, 'entities'):  # clapha's lean reader: the same fields as objects(), as tuples
            rows = env.reader.entities(env.battle)
            crowns, ended, winner = env._result([{'nativeObjectId': r[0], 'hp': r[5]} for r in rows])
            if self.unit_states and hasattr(env.reader, 'unit_states'):  # clapha-engine >= a9dceac: each unit's states
                units = env.reader.unit_states(env.battle)
                if self.ability_clock is not None:
                    self.ability_clock.update(env.tick(), units)
                states = {u['nativeObjectId']: unit_state_tuple(u) for u in units}
                entities = tuple(self._entity_row(r, states.get(r[0])) for r in rows)
            else:
                entities = tuple(self._entity_row(r) for r in rows)
        else:
            objects = env.reader.objects(env.battle)
            crowns, ended, winner = env._result(objects)
            entities = tuple(self._entity(o) for o in objects)
        tick = env.tick()
        players, plays, hands = [], [], {}
        lean = self._players()
        for owner, (hand, _, _) in enumerate(lean):
            hands[owner] = {i: card for i, (_, card) in hand.items()}
            if self.hands is not None:  # a card that left its slot since the last snapshot was played
                for i, card in self.hands[owner].items():
                    if hands[owner].get(i) != card:
                        self.sequence += 1
                        plays.append(CardPlay(self.sequence, tick, owner, card, COSTS.get(card, -1)))
        if self.hands is None or plays:
            self.evolution = self._evolution()
        self.hands = hands
        for owner, (hand, cycle, elixir) in enumerate(lean):
            players.append(Player(owner, elixir, tuple(HandCard(i, card, self._cost(owner, slot, card), 0, card)
                                                       for i, (slot, card) in sorted(hand.items())), cycle,
                                  self._runtime(owner), self.evolution[owner]))
        return State(self.generation, tick, bool(ended), winner, (crowns[0], crowns[1]),
                     tuple(players), entities, tuple(plays), self.sequence + 1)

    def _cost(self, owner, deck_slot, card):
        cost = COSTS.get(card)
        if cost is None:  # variable cost (Mirror, ...) or unseen card: the engine's own selection
            chosen = self.env._selection(owner, deck_slot)
            cost = chosen[1] >> 28 if chosen else -1
        return cost

    @staticmethod
    def _entity_row(r, states=None):
        """_entity for a row of clapha's Reader.entities (the same values); states: its unit_state_tuple, if read."""
        oid, owner, card, x, y, hp, max_hp, target, load, deploy, global_id = r
        if hp is None:
            kind = 'projectile' if global_id // 1000000 == PROJECTILE_CLASS else 'area'
            return Entity(oid, owner, card, x, y, None, None, None, None, kind, None, states)
        attack = load is not None  # load and deploy are read exactly when there is an attack component
        return Entity(oid, owner, card, x, y, hp, max_hp, target if attack else None, _ms(load) if attack else None,
                      'building' if card == -1 else 'troop', _ms(deploy), states)

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
