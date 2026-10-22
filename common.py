"""What conversion, training and live play share: the model's observation of one decision (PlayerView), legal
placement, the game clock and elixir timing. NumPy only (the converter runs without torch)."""
import json
import math
import os
from pathlib import Path

try:
    import _native_features
except ImportError:
    _native_features = None
if os.environ.get('CR_NATIVE_FEATURES', '1') == '0':
    _native_features = None

import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'
W, H = 18, 32
CELLS = W * H
MAX_OBJECTS, MAX_EFFECTS = 128, 6
DEPLOY_DELAY = 22            # a play lands ~22 ticks after the tap (measured live 21-25): its label is the board at the tap
OPP_VISIBLE = 13             # the opponent's command reaches our queue a median 13 ticks before it lands (2-21)
UNIT_STATES = 31
OBJ_FLOATS = 12 + UNIT_STATES + 2 + 2
HAND_FLOATS = 7
SCALARS = 2 + 6 + 2 + 4     # elixir both, clock_features, crowns both, spare
KINDS = {'unit': 1, 'tower': 2, 'area': 3, 'projectile': 4, 'timed': 5}
# names of every float input, in order (keep in step with the code below)
UNIT_STATE_NAMES = ('hidden', 'immune', 'invisible', 'airborne', 'stops_movement', 'stops_attacks', 'stopped_s',
                    'speed_pct', 'hit_speed_pct', 'pushed_s', 'attack_cycle', 'load_s', 'charge', 'dash_cooldown_s',
                    'ability_phase', 'timer_s', 'aim_dx', 'aim_dy', 'ability_used', 'shield', 'attack_stage',
                    'state_3', 'state_5', 'since_ability_s', 'ability_uses', 'height_z', 'jump_s', 'return_shot',
                    'special_cooldown_s', 'invisible_again_s', 'dash_hop')
OBJ_FLOAT_NAMES = ('x', 'y', 'dx', 'dy', 'hp_frac', 'log_max_hp', 'deploy_s', 'has_target', 'target_dx', 'target_dy',
                   'own', 'is_tower') + UNIT_STATE_NAMES + ('king_awake', 'daggers', 'spare_1', 'spare_2')
HAND_FLOAT_NAMES = ('cost', 'affordable', 'evo_ready', 'evo_progress', 'unknown', 'spare_1', 'spare_2')
SCALAR_NAMES = ('elixir_own', 'elixir_opp', 'rate_1x', 'rate_2x', 'rate_3x', 'overtime', 'period_left_min',
                'next_rate_min', 'crowns_own', 'crowns_opp', 'chef_cooking_own', 'chef_cooking_opp', 'spare_3', 'spare_4')
assert len(UNIT_STATE_NAMES) == UNIT_STATES and len(OBJ_FLOAT_NAMES) == OBJ_FLOATS
assert len(HAND_FLOAT_NAMES) == HAND_FLOATS and len(SCALAR_NAMES) == SCALARS
OBJECT_KEYS = ('obj_card', 'obj_data', 'obj_side', 'obj_kind', 'obj_float', 'eff_id', 'eff_left')
TOWER_IDS = range(5000000, 5000006)
COSTS = {int(k): v for k, v in json.loads((ASSETS / 'card_costs.json').read_text()).items()}
RARITY = json.loads((ASSETS / 'card_rarity.json').read_text())
_EFF = json.loads((ASSETS / 'effect_vectors.json').read_text())
EFFECT_VECTOR = {name: np.array(v, np.float32) for name, v in _EFF['effects'].items()}
EFFECT_FIELDS = len(_EFF['fields'])
_EFFECT_NAMES = {int(i): n for i, n in json.loads((ASSETS / 'effect_ids.json').read_text())['effects']}
MIRROR = 28000006
UNKNOWN = -1                 # a hand card the replay reconstruction is not sure of
WAIT, PLAY, ABILITY1, ABILITY2 = 0, 1, 2, 3
NEXT_EDGES = (1, 2, 3, 5, 9, 17)
# Royal Chef (clapha-engine docs/ASPECTS.md): its cooking progress is not found in memory; the game data's rule
# (chef_tower.toml ChefTower_CookingAction): cooking starts 7 s into the battle, contributions add up to 23,000 (600 a
# second, +200 per idle tower, none while attacking: ~23 s), then a pancake (this effect on one of the owner's troops).
# The input is the time since cooking last started; the towers' attacking / idle state is in the objects.
CHEF_PANCAKE = 3181413909
CHEF_START_TICKS = 140

# ---- clock and elixir -------------------------------------------------------------------------------------------------
TICKS_PER_SECOND = 20
DOUBLE, OVERTIME, TRIPLE, END = 2400, 3600, 4800, 6000   # engine, live and replays alike
REGEN = 10000 / 56  # raw elixir per tick (1 per 2.8 s); x2 after tick 2400, x3 after 4800 (measured on the IL cache)
# A tap the elixir bar cannot pay for yet is held by the client until it can, for at most HOLD_TICKS, then issued (it
# lands ~22 ticks later); held longer, it is dropped. Measured live on Nulls (2026-10-12, 18 short taps): every tap whose
# shortfall regenerated within 22.4 ticks landed (36-46 ticks after the tap), none needing 24+ ticks did. 22, not 23:
# over 136 v350/v400 games, holds over 22 ticks failed 8 of 28 times, up to 22 under 2% (docs/LIVE_LOSS_AUDIT.md).
HOLD_TICKS = 22


def regen(a, b):
    """Elixir (raw) regenerated over ticks (a, b]."""
    return sum(max(0, min(b, end) - max(a, start)) * mult * REGEN
               for start, end, mult in ((0, 2400, 1), (2400, 4800, 2), (4800, 1 << 30, 3)))


def hold_slack(tick):
    """How far short of a card's cost (raw elixir) a tap at `tick` may be and still be played."""
    return regen(tick, tick + HOLD_TICKS)


def elixir_rate(tick):
    return 3 if tick >= TRIPLE else 2 if tick >= DOUBLE else 1


def clock_features(tick):
    """Clock inputs: elixir rate one-hot (1x, 2x, 3x), overtime, time left in the period (minutes), ticks to the next
    rate change (minutes, 0 when none)."""
    rate = elixir_rate(tick)
    overtime = tick >= OVERTIME
    left = max(0, END - tick) if overtime else OVERTIME - tick
    nxt = DOUBLE - tick if rate == 1 else TRIPLE - tick if rate == 2 else 0
    return [float(rate == 1), float(rate == 2), float(rate == 3), float(overtime),
            left / (60 * TICKS_PER_SECOND), max(0, nxt) / (60 * TICKS_PER_SECOND)]


STANDARD_MATCH = '''{"battle":{"gmt":1,"plt":1,"t1s":0,"t2s":0,"gamemode":72000006,"hm":false,"lvlcap":0,"cardlvlmin":0,"stp":0,"trail":170000000,"te":-1,"tps":1,"deck0":{"hdr":"BD01","sp":[{"d":26000000},{"d":26000001},{"d":26000005},{"d":28000001},{"d":28000000},{"d":26000003},{"d":26000014},{"d":26000018}],"sc":[{"d":159000000,"l":1,"t":0,"c":1}]},"deck1":{"hdr":"BD01","sp":[{"d":26000000},{"d":26000001},{"d":26000005},{"d":28000001},{"d":28000000},{"d":26000003},{"d":26000014},{"d":26000018}],"sc":[{"d":159000000,"l":1,"t":0,"c":1}]},"avatar0":{"accountID.lo":1,"accountID.hi":0,"expLevel":1,"name":"Native-0","arena":54000001,"welo":-1,"scr":0,"lti":0,"npc":false},"avatar1":{"accountID.lo":2,"accountID.hi":0,"expLevel":1,"name":"Native-1","arena":54000001,"welo":-1,"scr":0,"lti":0,"npc":false},"location":15000199,"arena":54000001,"hbd":[{"em":{"ce":[],"de":[]},"sk":{"os":[],"ss":[],"rnd":false},"bn":{"bg":0,"fg":0},"kt":1},{"em":{"ce":[],"de":[]},"sk":{"os":[],"ss":[],"rnd":false},"bn":{"bg":0,"fg":0},"kt":1}]},"endTick":7200,"cmd":[],"evt":[],"rndSeed":1,"time":-1}'''


def standard_match(seed=1):
    match = json.loads(STANDARD_MATCH)
    match['rndSeed'] = seed
    return match


# ---- vocabulary --------------------------------------------------------------------------------------------------------
class Vocab:
    """Card, data and effect ids -> indices; 0 = padding, 1 = unknown (UNKNOWN cards and ids unseen when built)."""

    def __init__(self, ids):
        self.index = {int(i): n + 2 for n, i in enumerate(sorted(set(int(i) for i in ids)))}

    def __call__(self, i):
        if i is None:
            return 0
        if i == UNKNOWN:
            return 1
        return self.index.get(int(i) & 0xFFFFFFFF, 1)

    def __len__(self):
        return len(self.index) + 2

    @classmethod
    def load(cls, path=ASSETS / 'vocab.json'):
        return cls(json.loads(Path(path).read_text()))


def effect_matrix(vocab):
    """[len(vocab), 27]: the property vector of each effect id in the vocabulary (zeros for other ids), for the model's
    fixed effect table. Ids of both supported builds (effect_ids.json), vectors from the game data (effect_vectors.json)."""
    m = np.zeros((len(vocab), EFFECT_FIELDS), np.float32)
    for gid, name in _EFFECT_NAMES.items():
        v = EFFECT_VECTOR.get(name)
        if v is not None and vocab(gid) > 1:
            m[vocab(gid)] = v
    return m


def card_matrix(vocab):
    """[len(vocab), fields]: each card's and unit's fixed stats (assets/card_stats.json, from the build's logic tables
    by card_stats.py; zeros for ids without stats), for the model's --card_stats input."""
    table = json.loads((ASSETS / 'card_stats.json').read_text())
    m = np.zeros((len(vocab), len(table['fields'])), np.float32)
    for gid, v in table['stats'].items():
        if vocab(int(gid)) > 1:
            m[vocab(int(gid))] = v
    return m


# ---- one decision's inputs ---------------------------------------------------------------------------------------------
def _clip(v, lo, hi):
    return min(max(v, lo), hi)  # np.clip on one Python float costs ~10 us; this is the same double


def _ms_s(value, cap):
    return 0.0 if value is None else min(max(float(value), 0.0), cap * 1000.0) / 1000.0


def unit_state_values(u):
    """The 31 unit-state inputs of one unit_states() row."""
    buffs = u.get('buffs') or ()
    stops_move = stops_attack = False
    stopped = speed = hit = 0
    for b in buffs:
        m, a = b.get('stopsMovement'), b.get('stopsAttacks')
        stops_move = stops_move or bool(m)
        stops_attack = stops_attack or bool(a)
        if m or a:
            stopped = max(stopped, b.get('remainingMs') or 0)
        speed += b.get('speedPercent') or 0
        hit += b.get('hitSpeedPercent') or 0
    attack = u.get('attack') or {}
    ability = u.get('ability') or {}
    shot = attack.get('returnShot')
    hit_speed = attack.get('hitSpeedMs') or 0
    kind = u.get('kind')
    timer = u.get('remainingMs') if kind == 'area' else u.get('fuseMs') if kind == 'timed' else u.get('spawnTimerMs')
    return (
        float(bool(u.get('hidden'))), float(bool(u.get('immune'))), float(bool(u.get('invisible'))),
        float(bool(u.get('airborne'))), float(stops_move), float(stops_attack), _ms_s(stopped, 5),
        min(max(speed / 100.0, -1.0), 2.0), min(max(hit / 100.0, -1.0), 2.0),
        _ms_s(u.get('pushRemainingMs'), 5) if u.get('pushed') else 0.0,
        min(max((attack.get('cycleMs') or 0) / hit_speed, 0.0), 1.0) if hit_speed > 0 else 0.0,
        _ms_s(attack.get('loadMs'), 5), min(max(float(u.get('charge') or 0) / 10000.0, 0.0), 1.0),
        _ms_s(attack.get('dashCooldownMs'), 5), float(ability.get('phase') is not None), _ms_s(timer, 10),
        0.0, 0.0,                       # aim: filled per player (owner-relative)
        float(bool(ability.get('used'))),
        min(max((u.get('shield') or 0) / u['shieldMax'], 0.0), 1.0) if u.get('shieldMax') else 0.0,
        min(max(attack.get('stage') or 0, 0), 2) / 2.0, float(u.get('state') == 3), float(u.get('state') == 5),
        _ms_s(ability.get('sinceUseMs'), 10),
        min(max((ability.get('uses') or 0) / ability['maxCharges'], 0.0), 1.0) if ability.get('maxCharges') else 0.0,
        min(max(float(u.get('z') or 0) / 3000.0, 0.0), 2.0), _ms_s(u.get('jumpRemainingMs'), 2),
        min(max(shot['elapsedMs'] / shot['totalMs'], 0.0), 1.0) if shot else 0.0,
        _ms_s(attack.get('specialCooldownMs'), 5), _ms_s(u.get('invisibleAgainMs'), 5),
        min(float(attack.get('dashHop') or 0) / 3.0, 1.0),
    )


def ability_clock(last, tick, units):
    """Time since each hero / champion unit last used its ability (the engine keeps no such timer): clapha-engine
    crx/observe.AbilityClock's rule over the decisions. A use is placed exactly from the cooldown when the ability has
    one, otherwise at the first decision it is seen (at most 5 ticks late). Sets ability['sinceUseMs']."""
    for u in units:
        ability = u.get('ability')
        if not ability:
            continue
        count = ability['uses'] if ability.get('uses') is not None else int(bool(ability.get('used')))
        prev = last.get(u['nativeObjectId'])
        if prev is None or count != prev[0]:
            if count == 0:
                at = None
            elif (ability.get('configuredCooldownMs') or 0) > 0 and (ability.get('remainingCooldownMs') or 0) > 0:
                at = tick - (ability['configuredCooldownMs'] - ability['remainingCooldownMs']) // 50 + 1
            else:
                at = tick if prev is not None else None
            last[u['nativeObjectId']] = (count, at)
        at = last[u['nativeObjectId']][1]
        ability['sinceUseMs'] = 50 * (tick - at) if at is not None else None


def chef_clock(chef, tick, d):
    """Royal Chef: which owners have one (their towers' names) and the tick its cooking last started (7 s into the battle,
    then each time a pancake lands: the pancake effect appears on a new unit of that owner)."""
    owners = {o['nativeObjectId']: o['owner'] for o in d['objects']}
    if chef['has'] is None:
        chef['has'] = {0: False, 1: False}
        for u in d['units']:
            if u['nativeObjectId'] in TOWER_IDS and str(u.get('name', '')).startswith('ChefTower'):
                chef['has'][owners.get(u['nativeObjectId'], 0)] = True
    for u in d['units']:
        if any((b.get('globalId') or 0) & 0xFFFFFFFF == CHEF_PANCAKE for b in u.get('buffs') or ()):
            if u['nativeObjectId'] not in chef['fed']:
                chef['fed'].add(u['nativeObjectId'])
                owner = owners.get(u['nativeObjectId'])
                if owner in (0, 1):
                    chef['last'][owner] = tick


def rotate(x, y, owner):
    return (18000 - x, 32000 - y) if owner == 1 else (x, y)


def cell(x, y):
    return int(min(max(y // 1000, 0), H - 1) * W + min(max(x // 1000, 0), W - 1))


def _aim(u):
    if u.get('kind') == 'projectile' and u.get('aimX') is not None:
        return u['aimX'], u['aimY']
    if u.get('surfaceAt'):
        return tuple(u['surfaceAt'])
    return None


def decision_inputs(d, owner, vocab, prev_pos, state, shared=None):
    """One decision's object inputs. `prev_pos`: {object id: (x, y)} at the previous decision; `state`: per-game
    counters (PlayerView.state)."""
    if shared is None:
        shared = {}
    if not shared:
        shared.update(units={u['nativeObjectId']: u for u in d['units']},
                      objects=sorted(d['objects'], key=lambda o: (o['nativeObjectId'] not in TOWER_IDS, o['nativeObjectId'])),
                      positions={o['nativeObjectId']: (o['x'], o['y']) for o in d['objects']}, values={})
    units, objects = shared['units'], shared['objects']
    slots = state.get('object_slots', MAX_OBJECTS)
    n = min(len(objects), slots)
    state['dropped'] += max(0, len(objects) - MAX_OBJECTS)
    x = {
        'obj_card': np.zeros(slots, np.int64), 'obj_data': np.zeros(slots, np.int64),
        'obj_side': np.zeros(slots, np.int64), 'obj_kind': np.zeros(slots, np.int64),
        'obj_float': np.zeros((slots, OBJ_FLOATS), np.float32),
        'eff_id': np.zeros((slots, MAX_EFFECTS), np.int64), 'eff_left': np.zeros((slots, MAX_EFFECTS), np.float32),
    }
    positions = shared['positions']
    for i, o in enumerate(objects[:n]):
        oid = o['nativeObjectId']
        u = units.get(oid, {})
        ox, oy = rotate(o['x'], o['y'], owner)
        px, py = rotate(*prev_pos[oid], owner) if oid in prev_pos else (ox, oy)
        kind = 'tower' if oid in TOWER_IDS else u.get('kind', 'unit')
        x['obj_card'][i], x['obj_data'][i] = vocab(o['cardId']), vocab(o['dataGlobalId'])
        x['obj_side'][i] = 1 if o['owner'] == owner else 2
        x['obj_kind'][i] = KINDS.get(kind, 1)
        hp, mhp = o.get('hp'), o.get('maxHp')
        target = (o.get('targetEntityKey') or [None, None, None])[2]
        tpos = positions.get(target)
        tx, ty = rotate(*tpos, owner) if tpos else (ox, oy)
        phase = o.get('phaseRuntime') or {}
        f = [ox / 18000, oy / 32000, _clip((ox - px) / 1000, -3, 3), _clip((oy - py) / 1000, -3, 3),
             hp / mhp if hp is not None and mhp else 0.0, math.log1p(max(mhp or 0, 0)) / 16,
             _ms_s(phase.get('deployRemainingMs'), 5), float(tpos is not None),
             _clip((tx - ox) / 1000, -10, 10) / 10, _clip((ty - oy) / 1000, -10, 10) / 10,
             float(o['owner'] == owner), float(kind == 'tower')]
        if u and oid not in shared['values']:
            shared['values'][oid] = (_native_features.unit_state_values(u) if _native_features is not None
                                     else unit_state_values(u))
        us = list(shared['values'][oid]) if u else [0.0] * UNIT_STATES
        aim = _aim(u) if u else None
        if aim:
            ax, ay = rotate(aim[0], aim[1], owner)
            us[16], us[17] = _clip((ax - ox) / 1000, -20, 20) / 20, _clip((ay - oy) / 1000, -20, 20) / 20
        extra = [float(u.get('kingAwake', 0) or 0), (u.get('daggers') or 0) / 8 if u.get('daggers') is not None else 0.0,
                 0.0, 0.0]
        x['obj_float'][i] = np.array(f + us + extra, np.float32)
        buffs = sorted(u.get('buffs') or (), key=lambda b: -(b.get('remainingMs') or 0))
        state['effects_dropped'] += max(0, len(buffs) - MAX_EFFECTS)
        for k, b in enumerate(buffs[:MAX_EFFECTS]):
            x['eff_id'][i, k] = vocab(b.get('globalId'))
            x['eff_left'][i, k] = _ms_s(b.get('remainingMs'), 30)
    return x


def opponent_heroes(first_decision, forms, owner):
    """The opponent's hero / champion cards, known from the first tick: their deck's hero forms (`forms`: the
    opponent's form flag per deck slot, 2 = hero) and every Champion."""
    opp = [p for p in first_decision['lean']['state']['players'] if p['owner'] == 1 - owner][0]
    deck = [c['cardId'] for c in opp['deck']]
    return [c for i, c in enumerate(deck) if (forms and i < len(forms) and forms[i] == 2)
            or RARITY.get(str(c)) == 'Champion'][:2]


def crowns(d, owner):
    """Crowns `owner` has taken: the enemy's princess towers down, 3 once their King Tower is down."""
    other = 1 - owner
    alive = [(o['x'], o['y']) for o in d['objects'] if o['owner'] == other and o['nativeObjectId'] in TOWER_IDS]
    king = rotate(9000, 3000, other)
    if not any(abs(x - king[0]) <= 600 and abs(y - king[1]) <= 600 for x, y in alive):
        return 3
    return 3 - len(alive)


def destroyed_lanes(d, owner):
    """The enemy princess towers already down, named 'left' / 'right' by world x (3500 / 14500) for both owners
    (checked 2026-10-12 against 1,186 human plays: naming them from owner 1's view put its pockets on the wrong side)."""
    other = 1 - owner
    alive = {(o['x'], o['y']) for o in d['objects'] if o['owner'] == other and o['nativeObjectId'] in TOWER_IDS}
    lanes = []
    ky = 25500 if owner == 0 else 6500           # the enemy's princess towers, world positions; lanes named by world x
    for name, wx in (('left', 3500), ('right', 14500)):
        if not any(abs(ax - wx) <= 600 and abs(ay - ky) <= 600 for ax, ay in alive):
            lanes.append(name)
    return tuple(lanes)


# ---- own hand and the opponent's cycle ---------------------------------------------------------------------------------
QUEUE = 4


def _state(start, plays_before):
    hand, queue = set(start[0]), list(start[1])
    for card in plays_before:
        hand.remove(card)
        hand.add(queue.pop(0))
        queue.append(card)
    return frozenset(hand), tuple(queue)


def advance(states, plays):
    """The (hand, queue) states after `plays` more plays, duplicates merged: the starts that fit a replay become the same
    state once their differing cards have been played, so a game soon has one."""
    return {_state(s, plays) for s in states}


def certain(states):
    """What is certain over these (hand, queue) states: the cards in hand in all of them (a set), and the queue positions
    that agree (a tuple with None where they differ)."""
    states = list(states)
    if not states:
        return frozenset(), (None,) * QUEUE
    hand = frozenset.intersection(*(h for h, _ in states))
    queue = tuple(states[0][1][i] if all(q[i] == states[0][1][i] for _, q in states) else None for i in range(QUEUE))
    return hand, queue


def opponent_counted(plays):
    """For each card the opponent has played: how many more of their plays until it is back in their hand (0 = in hand
    now). Exact without knowing their deck or starting order: a card played k-th returns at play k + 4."""
    n, last = len(plays), {}
    for k, card in enumerate(plays):
        last[card] = k
    return {card: max(0, k + 1 + QUEUE - n) for card, k in last.items()}


class PlayerView:
    """One player's inputs, decision by decision: the same code for converted replays, the native arena and live play.
    step(d, landed) takes the raw record of one decision (the lean frame, objects(), unit_states()) and every card command
    issued so far as (execution tick, command); what the player may know of them at d's tick follows the rules here (own
    plays from the tap, landing - 22; the opponent's from 13 ticks before landing; plays landed by now). `candidates`: own
    starting hands that fit a replay's plays (convert.own_starts), where the hand is reconstructed; None where it is read
    (always known)."""

    def __init__(self, owner, vocab, opp_heroes, legal=None, candidates=None):
        self.owner, self.vocab, self.opp_heroes, self.legal, self.candidates = owner, vocab, opp_heroes, legal, candidates
        self.state = {'dropped': 0, 'effects_dropped': 0}
        self.prev, self.views, self.clock_state = {}, {}, {}
        self.states, self.played = (None, 0) if candidates is None else (set(candidates), 0)  # candidate states so far
        self.chef = {'has': None, 'last': {0: CHEF_START_TICKS, 1: CHEF_START_TICKS}, 'fed': set()}

    def step(self, d, landed, shared=None):
        owner, vocab, other = self.owner, self.vocab, 1 - self.owner
        t = d['tick']
        ability_clock(self.clock_state, t, d['units'])
        chef_clock(self.chef, t, d)
        x = decision_inputs(d, owner, vocab, self.prev, self.state, shared)
        self.prev = {o['nativeObjectId']: (o['x'], o['y']) for o in d['objects']}
        players = {p['owner']: p for p in d['lean']['state']['players']}
        lean = {p['owner']: p for p in d['lean']['players']}
        me, them = players[owner], players[other]
        before = sorted((tt, c['card_id']) for tt, c in landed if c['owner'] == owner and tt <= t)
        before = [c for _, c in before]
        if self.candidates is None:
            sure, queue = None, tuple(c['cardId'] for c in sorted(me.get('cycle') or (), key=lambda c: c['cycleIndex']))[:4]
        else:
            if len(before) not in self.views:  # a play at a time: the candidate states advance, merged as they converge
                self.states, self.played = advance(self.states, before[self.played:]), len(before)
                self.views[len(before)] = certain(self.states)
            sure, queue = self.views[len(before)]
        hand = [None] * 4
        for h in me['hand']:
            hand[h['handIndex']] = h['cardId'] if sure is None or h['cardId'] in sure else UNKNOWN
        last_own = before[-1] if before else None
        evo = {e['cardId']: e for e in lean[owner]['evolutionRuntime']}
        x['hand_card'] = np.array([vocab(c) for c in hand], np.int64)
        x['mirror_card'] = np.array([vocab(last_own) if c == MIRROR and last_own else 0 for c in hand], np.int64)
        hf = np.zeros((4, HAND_FLOATS), np.float32)
        for i, c in enumerate(hand):
            if c is None:
                continue
            real = [h['cardId'] for h in me['hand'] if h['handIndex'] == i][0]
            cost = (COSTS.get(last_own, 3) + 1) if real == MIRROR and last_own else COSTS.get(real, 3)
            e = evo.get(real) or {}
            hf[i] = (cost / 10, float(me['elixirRaw'] >= cost * 10000), float(bool(e.get('ready'))),
                     (e.get('progress') or 0) / max(1, e.get('cycleRequired') or 1), float(c == UNKNOWN), 0.0, 0.0)
        x['hand_float'] = hf
        x['cycle_card'] = np.array([vocab(q) if q is not None else 1 for q in queue] + [1] * (4 - len(queue)), np.int64)
        ab = np.zeros((2, 4), np.float32)
        for a in lean[owner]['abilityRuntime'][:2]:
            k = a['controllerSlot'] - 1
            ab[k] = (float(a['available']), float(a['buttonState'] in (2, 4)),
                     min(max(a['remainingChargesRaw'], 0), 3) / 3 if a['remainingChargesRaw'] is not None else 0.0,
                     (a['remainingCooldownMs'] / a['configuredCooldownMs']) if a['configuredCooldownMs'] else 0.0)
        x['ability'] = ab
        # plays in flight: own from the tap (landing - 22), the opponent's from 13 ticks before landing
        # a tap the client is still holding for elixir is in flight from the tap ('tap', when known), not only from
        # landing - 22: otherwise the model saw nothing in flight and tapped again (refused)
        own_f = [(tt, c) for tt, c in landed if c['owner'] == owner and c.get('tap', tt - DEPLOY_DELAY) <= t < tt]
        opp_f = [(tt, c) for tt, c in landed if c['owner'] == other and tt - OPP_VISIBLE <= t < tt][:2]

        def flight(tt, c):
            fx, fy = rotate(c['x'], c['y'], owner)
            return [fx / 18000, fy / 32000, (tt - t) / DEPLOY_DELAY, COSTS.get(c['card_id'], 3) / 10]
        x['own_flight_card'] = np.array(vocab(own_f[0][1]['card_id']) if own_f else 0, np.int64)
        x['own_flight_float'] = np.array(flight(*own_f[0]) if own_f else [0.0] * 4, np.float32)
        x['opp_flight_card'] = np.array([vocab(c['card_id']) for _, c in opp_f] + [0] * (2 - len(opp_f)), np.int64)
        x['opp_flight_float'] = np.array([flight(tt, c) for tt, c in opp_f] + [[0.0] * 4] * (2 - len(opp_f)), np.float32)
        seen = [c for _, c in sorted((tt, c['card_id']) for tt, c in landed if c['owner'] == other and tt <= t)]
        counted = opponent_counted(seen)
        last_index = {c: k for k, c in enumerate(seen)}
        order = sorted(counted, key=lambda c: -last_index[c])[:8]
        x['opp_seen_card'] = np.array([vocab(c) for c in order] + [0] * (8 - len(order)), np.int64)
        x['opp_seen_float'] = np.array([[counted[c] / 4, (len(seen) - 1 - last_index[c]) / 8, COSTS.get(c, 3) / 10]
                                        for c in order] + [[0.0] * 3] * (8 - len(order)), np.float32)
        x['opp_hero_card'] = np.array([vocab(c) for c in self.opp_heroes] + [0] * (2 - len(self.opp_heroes)), np.int64)
        crowns_now = [crowns(d, owner), crowns(d, other)]
        chef = self.chef
        cooking = [min(max(t - chef['last'][o], 0), 1200) / 20 / 30 if chef['has'][o] else 0.0 for o in (owner, other)]
        x['scalars'] = np.array([me['elixirRaw'] / 100000, them['elixirRaw'] / 100000, *clock_features(t),
                                 crowns_now[0] / 3, crowns_now[1] / 3, cooking[0], cooking[1], 0.0, 0.0], np.float32)
        if self.legal is not None:
            lanes = destroyed_lanes(d, owner)
            masks = []
            for i in range(4):
                c = ([h['cardId'] for h in me['hand'] if h['handIndex'] == i] or [None])[0]
                m = np.array(self.legal(c, owner, lanes, last_own), bool) if c is not None else np.zeros((H, W), bool)
                masks.append(m[::-1, ::-1] if owner == 1 else m)     # this player's view, like every other input
            x['legal'] = np.stack(masks)
        return x


# ---- legal placement ---------------------------------------------------------------------------------------------------
_PLACEMENT = None


def legal_rows(card_id, owner, lanes=(), last_card=None):
    """Exact frozen masks for the supported 1v1 map, in world coordinates. Mirror uses the previous card's rule."""
    global _PLACEMENT
    if owner not in (0, 1) or set(lanes) - {'left', 'right'}:
        raise ValueError('placement requires owner 0/1 and left/right destroyed tower lanes')
    if _PLACEMENT is None:
        table = json.loads((ASSETS / 'placement.json').read_text())
        masks = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(m), np.uint8)).reshape(H, W)
                          for m in table['masks']], bool)
        masks.flags.writeable = False
        _PLACEMENT = table['cards'], masks
    rule = int(card_id) if int(card_id) != MIRROR else last_card
    if rule is None:
        return np.zeros((H, W), bool)
    if int(rule) == MIRROR:
        return legal_rows(rule, owner, lanes)
    cards, masks = _PLACEMENT
    if str(rule) not in cards:
        raise ValueError(f'unsupported placement card: {rule}; update placement.json for the game build')
    pocket = int('left' in lanes) + 2 * int('right' in lanes)
    return masks[cards[str(rule)][owner * 4 + pocket]]


def cached_legal():
    """legal_rows with a per-process cache, in PlayerView's `legal` signature."""
    cache = {}

    def legal(card, owner, lanes, last):
        key = (card, owner, lanes, last if card == MIRROR else None)
        if key not in cache:
            cache[key] = np.array(legal_rows(card, owner, lanes=lanes, last_card=last), bool)
        return cache[key]
    return legal
