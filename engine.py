"""Small synchronous client. One controller owns one battle."""
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import time

from .protocol import State


@dataclass(frozen=True)
class Play:
    owner: int
    slot: int
    x: int
    y: int


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
        """Start a battle. `match` is a native replay dict (see standard_match.json); its
        rndSeed wins over `seed`. Defaults to the Training Camp standard deck."""
        self.state = None
        status = self.request('status')
        if status.get('mode') != 'headless':
            raise RuntimeError(
                f"headless reset requires headless mode, found {status.get('mode')!r}; "
                'run python bootstrap_emulator.py to restart the game before training')
        if match is None:
            match = json.loads(Path(__file__).with_name('standard_match.json').read_text())
            match['rndSeed'] = seed
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

    def reset(self, seed=1):
        self.state = None
        match = json.loads(Path(__file__).with_name('standard_match.json').read_text())
        match['rndSeed'] = seed
        status = self.request('status')
        if status.get('mode') == 'resident-headless':
            raise RuntimeError('close the active Batch and call stop_resident() first')
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
