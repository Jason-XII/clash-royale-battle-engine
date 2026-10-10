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
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from .engine import Transition
from .protocol import CardPlay, Entity, HandCard, Player, State

COSTS = {int(k): v for k, v in json.loads(Path(__file__).with_name('card_costs.json').read_text()).items()}
PROJECTILE_CLASS = 10  # data global id // 1e6 of LogicProjectileData (the probe checks type 0x0a); measured


def _ms(value):
    """The probe's rounding: up to a whole 50 ms tick; out of range -> unknown."""
    return None if value is None or value < 0 or value > 3600000 else (value + 49) // 50 * 50


class ClaphaEngine:
    def __init__(self, clapha_dir, workdir=None):
        clapha = Path(clapha_dir).expanduser().resolve()
        sys.path.insert(0, str(clapha / 'crx'))
        from env import CrxEnv  # clapha's crx modules import each other as top-level modules
        self.env = CrxEnv(clapha / 'engine', Path(workdir or tempfile.mkdtemp(prefix='clapha-')),
                          str(clapha / 'data' / 'assets'))
        self.state, self.generation, self.sequence = None, 0, 0

    def reset(self, seed=1, match=None):
        if match is None:
            match = json.loads(Path(__file__).with_name('standard_match.json').read_text())
            match['rndSeed'] = seed
        self.env.create_match(match)
        self.generation += 1
        self.hands = None
        self.env.step(90 - self.env.tick())
        self.state = self._capture()
        return self.state

    def step(self, actions=(), ticks=5):
        before = self.state
        actions = tuple(actions)
        for a in actions:
            self.env.queue_hand_action_at(SimpleNamespace(owner=a.owner, hand_index=a.slot, x=a.x, y=a.y),
                                          execute_tick=before.tick + 1)
        self.env.step(ticks)
        after = self._capture()
        played = {p.owner for p in after.plays}
        executed = tuple(a.owner in played for a in actions)  # ponytail: one play per owner per step, as Engine
        self.state = after
        return Transition(after, after.tick - before.tick, executed)

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
        snap = env.observe()
        players, plays, hands = [], [], {}
        for p in sorted(snap['players'], key=lambda p: p['owner']):
            hand = {h['handIndex']: h for h in p['hand']}
            hands[p['owner']] = {i: h['cardId'] for i, h in hand.items()}
            players.append(Player(p['owner'], p['elixirRaw'],
                                  tuple(HandCard(i, h['cardId'], self._cost(p['owner'], h), 0, h['cardId'])
                                        for i, h in sorted(hand.items())),
                                  tuple(c['cardId'] for c in p['cycle'])))
            if self.hands is not None:  # a card that left its slot since the last snapshot was played
                for i, card in self.hands[p['owner']].items():
                    if hands[p['owner']].get(i) != card:
                        self.sequence += 1
                        plays.append(CardPlay(self.sequence, snap['tick'], p['owner'], card,
                                              COSTS.get(card, -1)))
        self.hands = hands
        entities = tuple(self._entity(o) for o in snap['objects'])
        return State(self.generation, snap['tick'], bool(snap['ended']), snap['winner'], tuple(snap['crownsRaw']),
                     tuple(players), entities, tuple(plays), self.sequence + 1)

    def _cost(self, owner, hand_card):
        cost = COSTS.get(hand_card['cardId'])
        if cost is None:  # variable cost (Mirror, ...) or unseen card: the engine's own selection
            chosen = self.env._selection(owner, hand_card['deckSlot'])
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
