"""Resident batch client: several battles in one lane, stepped together."""
from dataclasses import dataclass
import json
import socket
import struct
from pathlib import Path

from .protocol import State


VERSION = 1
MAX_BATTLES = 16
MAX_ACTIONS = 8
MAX_PAYLOAD = 65536
HANDSHAKE = struct.Struct('<4sHHHH')
HEADER = struct.Struct('<4sHHQI')
REQUEST_ENTRY = struct.Struct('<HHI')
RESPONSE_ENTRY = struct.Struct('<HHHHI')
RECEIPT = struct.Struct('<BBHHHiIIiiiii')


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    kind: int
    owner: int
    action_index: int
    status: int
    execute_tick: int
    card_id: int
    card_parameter: int
    deck_slot: int
    cost: int
    form_code: int
    argument0: int
    argument1: int


def _decode_receipt(data):
    values = RECEIPT.unpack(data)
    if values[4]:
        raise RuntimeError('batch receipt reserved field is set')
    return ActionReceipt(*values[:4], *values[5:])


def _receive(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(min(1 << 16, count - len(data)))
        if not chunk:
            raise EOFError(f'battle engine closed at {len(data)}/{count} bytes')
        data += chunk
    return bytes(data)


def _encode_play(play):
    action = bytearray(20)
    action[0] = 1  # deploy
    action[1] = play.owner
    struct.pack_into('<H', action, 2, 1)  # execute on the next tick
    struct.pack_into('<i', action, 4, play.slot)
    struct.pack_into('<i', action, 8, play.x)
    struct.pack_into('<i', action, 12, play.y)
    return action


def stop_resident(port=26789):
    """Return a lane to single-battle mode after a batch session."""
    with socket.create_connection(('127.0.0.1', port), timeout=10) as sock:
        with sock.makefile('rb') as session:
            sock.sendall(b'session-v1\n')
            if not json.loads(session.readline()).get('ok'):
                return False
            sock.sendall(b'multi-stop\n')
            return json.loads(session.readline()).get('ok', False)


class Batch:
    """Several independent battles in one lane, advanced by one round trip per step.

    `states = batch.step(actions, ticks)` returns one State per battle. `actions[i]`
    is an iterable of Play for battle i; pass `()` for no actions.
    """

    def __init__(self, battles=16, port=26789, timeout=30):
        if not 1 <= battles <= MAX_BATTLES:
            raise ValueError(f'battles must be between 1 and {MAX_BATTLES}')
        self.battles = battles
        self.address = ('127.0.0.1', port)
        self.timeout = timeout
        self.connection = None
        self.sequence = 1
        self.last_receipts = tuple(() for _ in range(battles))

    def reset(self, seed=1):
        self.close()
        self._configure(seed)
        self._open()
        states = self._advance(90, ())  # first playable boundary, as Engine.reset
        self.last_receipts = tuple(() for _ in range(self.battles))
        return states

    def step(self, actions=(), ticks=5):
        return self._advance(ticks, actions)

    def close(self):
        if self.connection:
            self.connection.close()
        self.connection = None

    def _configure(self, seed):
        match = json.loads(Path(__file__).with_name('standard_match.json').read_text())
        with socket.create_connection(self.address, timeout=self.timeout) as sock:
            with sock.makefile('rb') as session:
                sock.sendall(b'session-v1\n')
                if not self._read(session).get('ok'):
                    raise RuntimeError('control session rejected')
                for battle in range(self.battles):
                    match['rndSeed'] = seed + battle
                    config = json.dumps(match, separators=(',', ':'))
                    sock.sendall(f'env {battle} configure {config}\n'.encode())
                    reply = self._read(session)
                    if not reply.get('ok'):
                        raise RuntimeError(f'battle {battle} configure failed: {reply}')

    def _read(self, session):
        line = session.readline()
        if not line:
            raise ConnectionError(
                f'no battle engine responded on port {self.address[1]}; '
                f'start a lane and pass its port')
        return json.loads(line)

    def _open(self):
        self.connection = socket.create_connection(self.address, timeout=self.timeout)
        self.connection.sendall(b'batch-fast-v1\n')
        magic, version, status, capacity, reserved = HANDSHAKE.unpack(
            _receive(self.connection, HANDSHAKE.size))
        if (magic, version, status, reserved) != (b'CRBH', VERSION, 0, 0) or capacity < self.battles:
            self.close()
            raise RuntimeError(
                f'batch handshake rejected: version={version} status={status} capacity={capacity}')

    def _advance(self, ticks, actions):
        if not self.connection:
            raise RuntimeError('reset the batch before stepping it')
        if not 1 <= ticks <= 1_000_000:
            raise ValueError('ticks must be between 1 and 1,000,000')
        groups = tuple(tuple(group) for group in actions) if actions else tuple(() for _ in range(self.battles))
        if len(groups) != self.battles:
            raise ValueError(f'expected {self.battles} action groups, got {len(groups)}')
        if any(len(group) > MAX_ACTIONS for group in groups):
            raise ValueError(f'each battle accepts at most {MAX_ACTIONS} actions')
        header = HEADER.pack(b'CRBQ', VERSION, 0, self.sequence, self.battles)
        body = bytearray()
        for battle, plays in enumerate(groups):
            body += REQUEST_ENTRY.pack(battle, len(plays), ticks)
            for play in plays:
                body += _encode_play(play)
        try:
            self.connection.sendall(header + body)
            response = HEADER.unpack(_receive(self.connection, HEADER.size))
            expected = (b'CRBS', VERSION, 0, self.sequence, self.battles)
            if response != expected:
                raise RuntimeError(f'batch response header mismatch: {response!r}')
            states = []
            all_receipts = []
            for expected_slot in range(self.battles):
                slot, status, count, flags, size = RESPONSE_ENTRY.unpack(
                    _receive(self.connection, RESPONSE_ENTRY.size))
                if slot != expected_slot or count > MAX_ACTIONS or flags or size > MAX_PAYLOAD:
                    raise RuntimeError(
                        f'invalid batch entry: slot={slot} status={status} '
                        f'receipts={count} flags={flags} size={size}')
                receipts = tuple(_decode_receipt(_receive(self.connection, RECEIPT.size))
                                 for _ in range(count))
                payload = json.loads(_receive(self.connection, size))
                if status:
                    raise RuntimeError(f'battle {slot} failed: {payload.get("error", payload)}')
                states.append(State.decode(payload))
                all_receipts.append(receipts)
            self.sequence += 1
            self.last_receipts = tuple(all_receipts)
            return tuple(states)
        except Exception:
            self.close()
            raise
