"""The emulator engine (Mac): a synchronous client of the probe in the game (`Engine`, headless; `RenderedEngine`,
drawn by the game UI), and the battle `State` every part of the project uses (clapha.py produces the same)."""
from dataclasses import dataclass
import json
import socket
import time

# The Training Camp standard deck at level 1 for both players; il.arena.make_match sets real decks and levels.
STANDARD_MATCH = '''{"battle":{"gmt":1,"plt":1,"t1s":0,"t2s":0,"gamemode":72000006,"hm":false,"lvlcap":0,"cardlvlmin":0,"stp":0,"trail":170000000,"te":-1,"tps":1,"deck0":{"hdr":"BD01","sp":[{"d":26000000},{"d":26000001},{"d":26000005},{"d":28000001},{"d":28000000},{"d":26000003},{"d":26000014},{"d":26000018}],"sc":[{"d":159000000,"l":1,"t":0,"c":1}]},"deck1":{"hdr":"BD01","sp":[{"d":26000000},{"d":26000001},{"d":26000005},{"d":28000001},{"d":28000000},{"d":26000003},{"d":26000014},{"d":26000018}],"sc":[{"d":159000000,"l":1,"t":0,"c":1}]},"avatar0":{"accountID.lo":1,"accountID.hi":0,"expLevel":1,"name":"Native-0","arena":54000001,"welo":-1,"scr":0,"lti":0,"npc":false},"avatar1":{"accountID.lo":2,"accountID.hi":0,"expLevel":1,"name":"Native-1","arena":54000001,"welo":-1,"scr":0,"lti":0,"npc":false},"location":15000199,"arena":54000001,"hbd":[{"em":{"ce":[],"de":[]},"sk":{"os":[],"ss":[],"rnd":false},"bn":{"bg":0,"fg":0},"kt":1},{"em":{"ce":[],"de":[]},"sk":{"os":[],"ss":[],"rnd":false},"bn":{"bg":0,"fg":0},"kt":1}]},"endTick":7200,"cmd":[],"evt":[],"rndSeed":1,"time":-1}'''


REGEN = 10000 / 56  # raw elixir per tick (1 per 2.8 s); x2 after tick 2400, x3 after 4800 (measured on the IL cache)
# A tap the elixir bar cannot pay for yet is held by the client until it can, for at most HOLD_TICKS, then issued (it
# lands ~22 ticks later); held longer, it is dropped. Measured live on Nulls (2026-10-12, 18 short taps): every tap whose
# shortfall regenerated within 22.4 ticks landed (36-46 ticks after the tap), none needing 24+ ticks did.
HOLD_TICKS = 23


def regen(a, b):
    """Elixir (raw) regenerated over ticks (a, b]."""
    return sum(max(0, min(b, end) - max(a, start)) * mult * REGEN
               for start, end, mult in ((0, 2400, 1), (2400, 4800, 2), (4800, 1 << 30, 3)))


def hold_slack(tick):
    """How far short of a card's cost (raw elixir) a tap at `tick` may be and still be played."""
    return regen(tick, tick + HOLD_TICKS)


def standard_match(seed=1):
    match = json.loads(STANDARD_MATCH)
    match['rndSeed'] = seed
    return match


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

    @classmethod
    def decode(cls, data):
        players = tuple(Player(p['owner'], p['elixir'], tuple(HandCard(*c) for c in p['hand']), tuple(p['cycle']),
                               tuple(Ability(*a[:5], tuple(a[5])) for a in p.get('abilities', ())),
                               tuple(Evolution(s, c, bool(e), n, r, bool(ok)) for s, c, e, n, r, ok in p.get('evolution', ())))
                        for p in data['players'])
        entities = tuple(Entity(*e) for e in data['entities'])
        return cls(data['generation'], data['tick'], data['finalized'], data['result'],
                   tuple(data['crowns']), players, entities,
                   tuple(CardPlay(*e) for e in data['plays']), data['next'])


@dataclass(frozen=True)
class Play:
    owner: int
    slot: int  # ABILITY: activate the hero / champion ability (x, y unused)
    x: int
    y: int


ABILITY = -1


@dataclass(frozen=True)
class Transition:
    state: State
    advanced: int
    executed: tuple[bool, ...]  # One result per submitted Play, in submission order.


class Engine:
    def __init__(self, host='127.0.0.1', port=26789, timeout=30):
        self.address = (host, port)
        self.timeout = timeout
        self.state = None
        self.wire_bytes = 0
        self.connection = None
        self.stream = None

    def connect(self):
        if self.connection:
            return
        self.connection = socket.create_connection(self.address, timeout=self.timeout)
        self.stream = self.connection.makefile('rb')
        self.connection.sendall(b'session-v1\n')
        response = self._receive()
        if not response.get('ok'):
            self.close()
            raise RuntimeError(response.get('error', 'control session failed'))

    def close(self):
        if self.stream:
            self.stream.close()
        if self.connection:
            self.connection.close()
        self.stream = self.connection = None

    def _receive(self, limit=65537):
        payload = self.stream.readline(limit)
        self.wire_bytes += len(payload)
        if not payload or not payload.endswith(b'\n'):
            raise ConnectionError(
                f'no battle engine responded on port {self.address[1]}; '
                f'check the lane is running and not in resident mode')
        return json.loads(payload)

    def request(self, command, limit=65537):
        """Raw diagnostic access; callers must not mutate a managed episode through it."""
        try:
            self.connect()
            self.connection.sendall((command + '\n').encode())
            return self._receive(limit)
        except Exception:
            self.close()
            raise

    def _state(self, command):
        response = self.request(command)
        if not response.get('ok'):
            raise RuntimeError(f"{command.split()[0]}: {response.get('error')}")
        return State.decode(response)

    def _capture(self, generation=0, cursor=0):
        return self._state(f'minimal {generation} {cursor}')

    def reset(self, seed=1, match=None):
        """Start a battle. `match` is a native replay dict (STANDARD_MATCH's layout); its
        rndSeed wins over `seed`. Defaults to the Training Camp standard deck."""
        self.state = None
        status = self.request('status')
        if status.get('mode') != 'headless':
            raise RuntimeError(
                f"headless reset requires headless mode, found {status.get('mode')!r}; "
                'run python -m native_engine.emulator bootstrap to restart the game')
        if match is None:
            match = standard_match(seed)
        configured = self.request('configure ' + json.dumps(match, separators=(',', ':')))
        if not configured.get('ok'):
            raise RuntimeError(configured.get('error', 'headless configuration failed'))
        stepped = self.request('step 90')  # First playable boundary; execute the first action at tick 91.
        if not stepped.get('ok'):
            raise RuntimeError(stepped.get('error', 'headless warmup failed'))
        self.state = self._capture()
        return self.state

    def step(self, actions=(), ticks=5):
        before = self.state
        actions = tuple(actions)
        fields = ['minimal-transition', str(before.generation), str(before.next_sequence),
                  str(ticks), str(len(actions))]
        for action in actions:
            fields.extend(map(str, (action.owner, action.slot, action.x, action.y)))
        # A partial transport failure invalidates this client until reset; never replay writes.
        self.state = None
        after = self._state(' '.join(fields))
        advanced = after.tick - before.tick
        # Not matched by card: evolved/champion forms are recorded under another id (e.g. 0).
        # ponytail: two plays by one owner in one step can't be told apart; callers send one each.
        executed = tuple(any(e.owner == a.owner and e.tick == before.tick + 1 for e in after.plays)
                         for a in actions)
        self.state = after
        return Transition(after, advanced, executed)

    def run(self, ticks, count):
        """Advance `count` chunks of `ticks` without actions in one round trip (scheduled
        replay commands still fire); return the state after each chunk, stopping at the end."""
        before = self.state
        self.state = None
        states = self.request(f'minimal-run {before.generation} {before.next_sequence} {ticks} {count}', limit=-1)
        if isinstance(states, dict):
            raise RuntimeError(f"minimal-run: {states.get('error')}")
        states = [State.decode(s) for s in states]
        self.state = states[-1]
        return states

    def advance(self, ticks):
        """Step without actions; scheduled replay commands still fire."""
        return self.step((), ticks)

    def pipeline(self, commands):
        """Send many commands in one write and return their responses in order."""
        try:
            self.connect()
            self.connection.sendall(''.join(command + '\n' for command in commands).encode())
            return [self._receive() for _ in commands]
        except Exception:
            self.close()
            raise

    @staticmethod
    def card_command(owner, card, x, y, tick):
        """Play `card` (by id) at the state boundary before `tick`; native resolves the hand slot."""
        return f'replay-schedule-card {owner} {card} {x} {y} {tick}'

    @staticmethod
    def ability_command(owner, hints, tick):
        """Activate the owner's Ready ability; `hints` are native name hints, () = the unique Ready one."""
        hints = dict.fromkeys(''.join(c for c in h.casefold() if c.isascii() and c.isalnum()) for h in hints)
        return f"replay-schedule-ability {owner} {','.join(h for h in hints if h) or '-'} {tick}"

    def schedule(self, commands):
        """Clear the replay schedule, queue `commands` in one round trip, return their sequences."""
        receipts = self.pipeline(['replay-schedule-clear', *commands])
        failed = [(c, r.get('error')) for c, r in zip(['clear', *commands], receipts) if not r.get('ok')]
        if failed:
            raise RuntimeError(f'schedule rejected: {failed[:3]}')
        return [r['sequence'] for r in receipts[1:]]

    def schedule_status(self, sequences):
        return self.pipeline([f'replay-schedule-status {s}' for s in sequences])


class RenderedEngine(Engine):
    """One paused, exactly stepped battle drawn by the stock game UI."""

    def reset(self, seed=1, match=None):
        self.state = None
        if match is None:
            match = standard_match(seed)
        status = self.request('status')
        if status.get('mode') == 'resident-headless':
            raise RuntimeError('the lane is in resident mode; restart the game (python -m native_engine.emulator bootstrap)')
        accepted = self.request('configure-native ' + json.dumps(match, separators=(',', ':')))
        if not accepted.get('ok'):
            raise RuntimeError(accepted.get('error', 'native renderer rejected the match'))
        sequence = accepted['sequence']
        deadline = time.monotonic() + self.timeout
        while True:
            status = self.request('status')
            if status.get('nativeRenderReady') and status.get('nativeRenderLoaded', 0) >= sequence:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('native renderer did not create the battle scene')
            time.sleep(0.05)
        paused = self.request('pause')
        current_tick = paused['tick']
        if current_tick < 90:
            self.request(f'advance-native {90 - current_tick}')
        self.state = self._capture()
        if self.state.tick != 90:
            raise RuntimeError(f'native renderer missed the first playable tick: {self.state.tick}')
        return self.state

    def step(self, actions=(), ticks=5):
        before = self.state
        if before is None:
            raise RuntimeError('reset or pause the rendered engine before stepping it')
        if not 1 <= ticks <= 1_000_000:
            raise ValueError('ticks must be between 1 and 1,000,000')
        actions = tuple(actions)
        cards = []
        for action in actions:
            player = before.players[action.owner]
            card = next((card for card in player.hand if card.slot == action.slot), None)
            if card is None:
                raise ValueError(f'hand slot {action.slot} is unavailable for owner {action.owner}')
            cards.append(card.card)
        self.state = None
        for action, card in zip(actions, cards):
            queued = self.request(
                f'replay-schedule-card {action.owner} {card} {action.x} {action.y} {before.tick + 1}'
            )
            if not queued.get('ok'):
                raise RuntimeError(queued.get('error', 'could not queue rendered action'))
        advanced = self.request(f'advance-native {ticks}')
        if not advanced.get('ok'):
            raise RuntimeError(advanced.get('error', 'could not advance native renderer'))
        after = self._capture(before.generation, before.next_sequence)
        executed = tuple(any(event.owner == action.owner and event.tick == before.tick + 1
                             for event in after.plays)
                         for action in actions)  # not by card: see Engine.step
        self.state = after
        return Transition(after, after.tick - before.tick, executed)

    def set_speed(self, multiplier):
        if multiplier not in (0.25, 0.5, 1, 2, 4):
            raise ValueError('speed must be 0.25, 0.5, 1, 2, or 4')
        return self.request(f'speed {multiplier:g}')

    def pause(self):
        result = self.request('pause')
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'could not pause native renderer'))
        self.state = self._capture()
        return self.state

    def resume(self):
        result = self.request('resume')
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'could not resume native renderer'))
        self.state = None
        return result
