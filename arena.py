"""Paired-seat native battles between two trained models (Linux, inside a Slurm allocation).

    python arena.py A.pt B.pt --engine ~/clapha-engine --pool data/deck_pool.json --games 20 --out runs/arena.jsonl
"""
import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import resource
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

import common as C
import train

# decision -> landing ticks, one drawn per play: as live (136 v350/v400 games, taps it could pay for, 2026-10-18:
# 21 16%, 22 58%, 23 16%, 24 5%, 25-26 5%); was uniform 23-27, as replays are labelled, 1-3 ticks late
PLAY_DELAY = (21,) * 16 + (22,) * 58 + (23,) * 16 + (24,) * 5 + (25,) * 3 + (26,) * 2
# the engine is asked the cost of variable-cost cards (Mirror, ...) and cards outside the original table
FIXED_COSTS = {k: v for k, v in C.COSTS.items() if k not in (28000006, 28000025, 26000107)}


@dataclass(frozen=True)
class HandCard:
    slot: int
    card: int
    cost: int
    parameter: int  # Native command encoding; never a policy feature.
    form_card: int  # Card id of the form that would be played (evolution/hero), else == card.


@dataclass(frozen=True)
class Ability:
    """A hero / champion controller (the probe's abilityRuntime)."""
    button: int  # 1 no unit, 2 ready, 3 unit deploying, 4 limited availability, 6 no charges, 8 cooldown, 10 casting, ...
    cooldown_ms: int
    cooldown_total_ms: int
    charges: int  # left; -1 unlimited
    max_charges: int
    units: tuple[int, ...]  # native object ids of the units carrying it

    @property
    def ready(self):
        return self.button in (2, 4) and self.cooldown_ms == 0 and self.charges != 0 and bool(self.units)


@dataclass(frozen=True)
class Evolution:
    """Evolution progress of one deck slot (the probe's evolutionRuntime)."""
    deck_slot: int
    card: int
    evolvable: bool
    required: int  # cycles
    remaining: int
    ready: bool  # its next play is evolved


@dataclass(frozen=True)
class Player:
    owner: int
    elixir: int  # Fixed-point native units: 10,000 = one elixir.
    hand: tuple[HandCard, ...]
    cycle: tuple[int, ...]
    abilities: tuple[Ability, ...] = ()
    evolution: tuple[Evolution, ...] = ()


@dataclass(frozen=True)
class Entity:
    id: int
    owner: int
    card: int  # -1 for towers.
    x: int
    y: int
    hp: int | None
    max_hp: int | None
    target_id: int | None  # Stable entity ID held by the native attack component.
    attack_cooldown_ms: int | None  # Zero means ready; rounded up to a 50 ms tick.
    entity_type: str  # troop, building, projectile or area
    deployment_remaining_ms: int | None


@dataclass(frozen=True)
class CardPlay:
    sequence: int
    tick: int
    owner: int
    card: int
    cost: int


@dataclass(frozen=True)
class State:
    generation: int
    tick: int
    finalized: bool
    result: int | None
    crowns: tuple[int, int]
    players: tuple[Player, ...]
    entities: tuple[Entity, ...]
    plays: tuple[CardPlay, ...]
    next_sequence: int



@dataclass(frozen=True)
class Play:
    owner: int
    slot: int  # ABILITY: activate the hero / champion ability of controller x (0 or 1; y unused)
    x: int
    y: int


ABILITY = -1


@dataclass(frozen=True)
class Transition:
    state: State
    advanced: int
    executed: tuple[bool, ...]  # One result per submitted Play, in submission order.



ENGINE_CPUS = 2  # what the engine is told about the machine: it sized its thread pool from all 128 CPUs (~70 threads)
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


class Engine:
    def __init__(self, clapha_dir, workdir=None):
        raise_thread_limit()  # also here, for scripts that start a single engine
        clapha = Path(clapha_dir).expanduser().resolve()
        sys.path.insert(0, str(clapha / 'crx'))
        import loader  # clapha's crx modules import each other as top-level modules
        if C._native_features is not None:
            import observe
            observe._i32, observe._u64 = C._native_features.i32, C._native_features.u64
        _report_cpus(loader)
        from env import CrxEnv
        self.env = CrxEnv(clapha / 'engine', Path(workdir or tempfile.mkdtemp(prefix='clapha-')),
                          str(clapha / 'data' / 'assets'))
        self.state, self.generation, self.sequence, self.pending = None, 0, 0, []
        self.expired = 0  # accepted plays that never landed (held too long, or the card gone): should stay ~0
        # per owner (22 Oct): may tap another card while one is in flight, as humans do (5.7% of their plays come
        # <= 20 ticks after the previous one); its cost is reserved from the elixir the pending plays haven't paid yet
        self.overlap = [False, False]

    def reset(self, seed=1, match=None):
        if match is None:
            match = C.standard_match(seed)
        self.env.create_match(match)
        self.generation += 1
        self.hands, self.pending, self.played = None, [], []
        r = self.env.reader
        logic = r.logic(self.env.battle)  # a deck never changes within a battle: read its card ids once
        self.deck = [[r.card_id(r.card_data(slot)) for slot in r.deck_slots(logic, owner)] for owner in (0, 1)]
        self.env.step(90 - self.env.tick())
        self.state = self._capture()
        return self.state

    def step(self, actions=(), ticks=5, delay=0):
        """Plays decided on the current state land `delay` ticks later (live: common.DEPLOY_DELAY), or on the
        next tick with delay=0; `delay` = a tuple of choices: each play draws its own (PLAY_DELAY, as live). As the live client does (C.HOLD_TICKS), a play the elixir cannot pay for yet is held
        until it can, for at most C.HOLD_TICKS, then issued (and lands `delay` ticks later); a play that could not be paid
        for within that is refused. As live does, a player with a card in flight cannot start another (unless
        self.overlap[owner]: 22 Oct), and a pending ability press blocks only another press, not a card.
        `executed[i]` says whether play i was accepted; landed plays show up in later states' `plays`."""
        before = self.state
        self.played = []  # confirmed card consumption this step, with exact tick and commanded position
        executed = []
        for a in actions:
            # a card waits only for this player's cards, a press only for its presses: the live client takes a card tap
            # while a press is on its way (22 Oct: blocking it refused ~0.4 card taps a game the model could not see)
            mine = [p for p in self.pending if p['owner'] == a.owner and (p['slot'] == ABILITY) == (a.slot == ABILITY)]
            free = not mine or (self.overlap[a.owner] and a.slot != ABILITY and
                                all(p['slot'] != a.slot and not p['held'] for p in mine))
            reserved = sum(p['cost'] for p in mine) * 10000
            lag = max(delay if isinstance(delay, int) else random.choice(delay), 1)
            due = before.tick + lag
            if a.slot == ABILITY:  # a tap on the ability button: lands like a card, if a carrier is still ready then
                ok = free and any(x.ready for x in before.players[a.owner].abilities)
                if ok:
                    self.pending.append(dict(due=due, held=False, owner=a.owner, slot=ABILITY, card=None, cost=0,
                                             controller=a.x))
                executed.append(ok)
                continue
            me = before.players[a.owner]
            card = next((h for h in me.hand if h.slot == a.slot), None)
            ok = card is not None and free and card.cost * 10000 <= me.elixir - reserved + C.hold_slack(before.tick)
            if ok:
                self.pending.append(dict(due=due, tap=before.tick, delay=lag, held=card.cost * 10000 > me.elixir - reserved,
                                         hold_until=before.tick + C.HOLD_TICKS, owner=a.owner, slot=a.slot,
                                         card=card.card, cost=card.cost, x=a.x, y=a.y))
            executed.append(ok)
        for _ in range(ticks):
            released = self._release() if self.pending else []
            self.env.step(1)
            if released:
                players = self._players()
                for p in released:
                    if players[p['owner']][0].get(p['slot'], (None, None))[1] != p['card']:
                        self.played.append(dict(p, tick=self.env.tick()))
                    else:
                        self.expired += 1
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
        """Issue held plays once the elixir pays for them (or drop them after C.HOLD_TICKS), and hand the plays due on the
        next tick to the engine if their card is still there."""
        now = self.env.tick()
        if not any(p['held'] or p['due'] <= now + 1 for p in self.pending):
            return []
        players = self._players()
        released = []
        for p in list(self.pending):
            if p['slot'] != ABILITY and p['held']:
                if players[p['owner']][2] >= p['cost'] * 10000:
                    p['held'], p['due'] = False, now + p['delay']  # issued now
                elif now >= p['hold_until']:
                    self.pending.remove(p)
                    self.expired += 1
                else:
                    p['due'] = now + 1 + p['delay']  # not issued yet: lands no earlier than this
                continue
            if p['due'] > now + 1:
                continue
            self.pending.remove(p)
            if p['slot'] == ABILITY:
                self._activate(p['owner'], p['controller'])
                continue
            hand, _, elixir = players[p['owner']]
            if hand.get(p['slot'], (None, None))[1] == p['card'] and elixir >= p['cost'] * 10000:
                self.env.queue_hand_action_at(SimpleNamespace(owner=p['owner'], hand_index=p['slot'], x=p['x'],
                                                              y=p['y']), execute_tick=now + 1)
                released.append(p)
            else:
                self.expired += 1
        return released

    def _activate(self, owner, controller):
        """Activate `owner`'s hero / champion ability of `controller` (0 or 1, the model's ABILITY1 / ABILITY2), if
        ready (a press on a button no longer ready does nothing, as in the game)."""
        r = self.env.reader
        for row in r.abilities(r.player(r.logic(self.env.battle), owner)):
            if row['controllerSlot'] - 1 == controller and row['buttonState'] in (2, 4) and row['remainingCooldownMs'] == 0 and row['remainingChargesRaw'] != 0:
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

    def record(self):
        """The current decision's raw record as convert.py records it (the lean frame without its objects,
        objects(), unit_states()): the input of common.PlayerView, so games played here and converted
        replays are read by the same code."""
        r, battle = self.env.reader, self.env.battle
        lean = r.frame(battle)
        objects = lean.pop('objects')  # frame() has read them already
        return {'tick': self.env.tick(), 'lean': lean, 'objects': objects, 'units': r.unit_states(battle)}

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
                        plays.append(CardPlay(self.sequence, tick, owner, card, FIXED_COSTS.get(card, -1)))
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
        cost = FIXED_COSTS.get(card)
        if cost is None:  # variable cost (Mirror, ...) or unseen card: the engine's own selection
            chosen = self.env._selection(owner, deck_slot)
            cost = chosen[1] >> 28 if chosen else -1
        return cost

    @staticmethod
    def _entity_row(r):
        """_entity for a row of clapha's Reader.entities (the same values)."""
        oid, owner, card, x, y, hp, max_hp, target, load, deploy, global_id = r
        if hp is None:
            kind = 'projectile' if global_id // 1000000 == PROJECTILE_CLASS else 'area'
            return Entity(oid, owner, card, x, y, None, None, None, None, kind, None)
        attack = load is not None  # load and deploy are read exactly when there is an attack component
        return Entity(oid, owner, card, x, y, hp, max_hp, target if attack else None, _ms(load) if attack else None,
                      'building' if card == -1 else 'troop', _ms(deploy))

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


class Commands:
    """Card plays issued so far as (execution tick, command), as a converted game's header holds them: landed plays from
    confirmed per-tick hand consumption, plays in flight from the engine queue (their predicted landing tick)."""

    def __init__(self):
        self.landed, self.last_tick = [], None

    def update(self, state, engine):
        if state.tick != self.last_tick:
            self.landed += [(p['tick'], {'owner': p['owner'], 'card_id': p['card'], 'x': p['x'], 'y': p['y']})
                            for p in engine.played]
            self.last_tick = state.tick
        flying = [(p['due'], {'owner': p['owner'], 'card_id': p['card'], 'x': p['x'], 'y': p['y'], 'tap': p['tap']})
                  for p in engine.pending if p.get('card') is not None]
        return self.landed + flying


class Inputs:
    """The inputs of the players in `vocabs` ({owner: vocabulary}), decision by decision, in one ClaphaEngine game: the
    raw record once per decision, a PlayerView per player."""

    def __init__(self, engine, match, vocabs, legal):
        record = engine.record()
        self.engine, self.commands = engine, Commands()
        self.views = {}
        for owner, vocab in vocabs.items():
            other = 1 - owner
            forms = [c.get('el', 0) for c in match['battle'][f'deck{other}']['sp']]  # 1 evolution, 2 hero (as replays)
            heroes = C.opponent_heroes(record, forms, owner)
            self.views[owner] = C.PlayerView(owner, vocab, heroes, legal)

    def __call__(self, state):
        record, issued = self.engine.record(), self.commands.update(state, self.engine)
        shared = {}  # owner-independent board features, valid only for this decision
        return {owner: view.step(record, issued, shared) for owner, view in self.views.items()}


def make_match(pair, seed):
    match = C.standard_match(seed)
    battle = match['battle']
    battle['lvlcap'], battle['cardlvlmin'] = pair['lvlcap'], pair['cardlvlmin']
    for owner in (0, 1):
        battle[f'deck{owner}']['sp'] = pair['decks'][owner]
        battle[f'avatar{owner}']['expLevel'] = pair['exp'][owner]
        battle['hbd'][owner]['kt'] = pair['kt'][owner]
    return match


def evaluate(engine, models, pair, seed, a_owner, device='cpu', a_overlap=False):
    match = make_match(pair, seed)
    state = engine.reset(match=match)
    engine.overlap = [a_overlap and owner == a_owner for owner in (0, 1)]
    seats = [0 if owner == a_owner else 1 for owner in (0, 1)]
    agents = [train.Agent(models[index][0], owner, device) for owner, index in enumerate(seats)]
    inputs = Inputs(engine, match, {owner: models[index][1] for owner, index in enumerate(seats)}, C.cached_legal())
    plays, refused = [0, 0], [0, 0]
    while not state.finalized and state.tick < 7200:
        rows = inputs(state)
        choices = [agent.act(rows[agent.owner]) for agent in agents]
        actions = [Play(agent.owner, ABILITY, choice[1], 0) if choice[0] == 'ability' else Play(agent.owner, *choice[1:])
                   for agent, choice in zip(agents, choices) if choice[0] != 'wait']
        transition = engine.step(actions, 5, delay=PLAY_DELAY)
        state = transition.state
        for action, ok in zip(actions, transition.executed):
            refused[action.owner] += not ok
        for play in engine.played:
            plays[play['owner']] += 1
    if not state.finalized:
        raise RuntimeError('native battle did not finalize before tick 7200')
    winner = None if state.result not in (0, 1) else 'A' if state.result == a_owner else 'B'
    return dict(winner=winner, a_owner=a_owner, crowns_a=state.crowns[a_owner], crowns_b=state.crowns[1-a_owner],
                plays_a=plays[a_owner], plays_b=plays[1-a_owner], refused_a=refused[a_owner], refused_b=refused[1-a_owner],
                tick=state.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True, help='deck pairs with actual levels and evolution flags')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--unmasked_b', action='store_true', help="B samples as before PR 11 (no elixir mask)")
    parser.add_argument('--a-temperature', type=float, default=1.0, help="A's card and tile temperature")
    parser.add_argument('--b-temperature', type=float, default=1.0,
                        help="B's card and tile temperature (< 1: sharper; when to play stays sampled as trained)")
    parser.add_argument('--deck', type=Path, help="A always plays this deck (JSON, as in the pool), B the pair's second deck")
    parser.add_argument('--device', default='cpu', help='where both models run (cuda: big models; processes can share a GPU)')
    parser.add_argument('--a-overlap', action='store_true', help='A may tap a card while another is still in flight')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error('games must be positive and even for paired seats')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pool = json.loads(args.pool.read_text())
    if not pool:
        parser.error('deck pool is empty')
    deck = args.deck and json.loads(args.deck.read_text())
    models = [train.load(path, args.device)[:2] for path in (args.a, args.b)]
    models[1][0].elixir_mask = not args.unmasked_b
    models[0][0].temperature, models[1][0].temperature = args.a_temperature, args.b_temperature
    models[0][0].overlap = args.a_overlap  # the sampler too: it would wait while A's card is in flight
    engine = Engine(args.engine)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.out.open('x') as stream:
        for pair_index, index in enumerate(rng.integers(len(pool), size=args.games // 2)):
            for seat in (0, 1):
                pair = pool[index]
                if deck:
                    pair = dict(pair, decks=[deck, pair['decks'][1]][::1 if seat == 0 else -1])
                result = dict(game=len(results), **evaluate(engine, models, pair, args.seed + pair_index, seat, args.device,
                                                               args.a_overlap))
                results.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(json.dumps(result), flush=True)
    wins = sum(r['winner'] == 'A' for r in results)
    draws = sum(r['winner'] is None for r in results)
    print(json.dumps(dict(games=len(results), wins=wins, losses=len(results)-wins-draws, draws=draws,
                          score=(wins + 0.5 * draws) / len(results),
                          refused_per_game_a=sum(r['refused_a'] for r in results) / len(results),
                          refused_per_game_b=sum(r['refused_b'] for r in results) / len(results))), flush=True)
    sys.stdout.flush()
    os._exit(0)  # Native exit destructors are unsafe; all result files have been flushed.


if __name__ == '__main__':
    main()
