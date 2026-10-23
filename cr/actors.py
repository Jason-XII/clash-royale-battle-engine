"""PPO's actors: in each actor job, decision servers (batched GPU decisions for many games) and collectors (one native
game each); finished games go to the learner (ppo.py learn), which sends back its newest weights and opponents.

A collector plays one native game at a time (a second engine in one process crashes) and asks its job's decision
server for both seats' actions every decision. A server answers whichever collectors are waiting in one GPU call per
model, as soon as its last call is done (the old servers held each round for every collector: CPUs idled, 4k games/h).
"""
from collections import defaultdict
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener, wait
import os
from pathlib import Path
import queue
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

from cr import matchmaking, observation as C, policy
from cr.engine import Engine, Game, margin, potential, raise_thread_limit, tower_hp

OVERFLOW_PENALTY = 0.002
REPLY_TIMEOUT = 180  # s; a learner that says nothing for longer is gone (a failed node never closes its sockets)
PRIVATE = ('hand_card', 'mirror_card', 'hand_float', 'cycle_card', 'ability')  # the opponent's view: what only they know


def endpoint(run):
    """The learner's address, key and settings, waiting until a learner has written them."""
    while True:
        try:
            return json.loads((run / 'endpoint.json').read_text())
        except (OSError, ValueError):
            time.sleep(10)


def receive(conn):
    if not conn.poll(REPLY_TIMEOUT):
        raise EOFError(f'no reply for {REPLY_TIMEOUT} s')
    return conn.recv()


class Link:
    """A connection to one learner rank (collector `index` -> rank index % ranks, so every rank gets games; else one at
    random): finished games out, the opponents (and, with weights=True, the newest weights) back. Reconnects
    (re-reading endpoint.json) whenever the learner is gone or restarted; games it could not send are lost."""

    def __init__(self, run, weights=True, index=None):
        self.run, self.conn, self.version, self.weights, self.index = run, None, -1, weights, index

    def exchange(self, games, results, timing):
        while True:
            try:
                if self.conn is None:
                    ep = endpoint(self.run)
                    port = random.choice(ep['ports']) if self.index is None else ep['ports'][self.index % len(ep['ports'])]
                    self.conn = Client((ep['host'], port), authkey=bytes.fromhex(ep['authkey']))
                    self.version = -1  # a restarted learner may have rolled back: take its weights
                self.conn.send((self.version if self.weights else None, games, results, timing))
                version, weights, opponents = receive(self.conn)
                self.version = version
                return weights, opponents
            except (OSError, EOFError) as error:
                print(f'learner unreachable ({error!r}); retrying', flush=True)
                if self.conn is not None:
                    self.conn.close()
                self.conn, games, results = None, [], []
                time.sleep(15)


class Seat:
    """One learning seat's decisions in one game."""

    def __init__(self):
        self.rows, self.opp, self.action, self.logp, self.value, self.reward = [], [], [], [], [], []

    def trajectory(self, **extra):
        return dict(inputs={k: policy.pad_stack([r[k] for r in self.rows]) for k in self.rows[0]},
                    opp={k: np.stack([r[k] for r in self.opp]) for k in PRIVATE},
                    action=np.array(self.action, np.int64), logp=np.array(self.logp, np.float32),
                    value=np.array(self.value, np.float32), reward=np.array(self.reward, np.float32), **extra)


def redirect(run, name):
    job = os.environ.get('SLURM_JOB_ID', socket.gethostname())
    log = os.open(run / 'logs' / f'{name}-{job}.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log, 1)
    os.dup2(log, 2)


def collector(index, run, addresses, key, affinity=False):
    """One native game at a time (the engine allows one per process), decided by its job's decision server at
    `address`; finished games go to the learner."""
    redirect(run, f'collector-{index}')
    os.nice(5)  # the learner and the decision servers go first
    torch.set_num_threads(1)
    time.sleep(index % 8 * 4)  # engines boot a few at a time
    rng = random.Random(f'{socket.gethostname()}-{os.getpid()}-{time.time_ns()}')
    s = endpoint(run)['settings']
    pool = json.loads(Path(s['pool']).read_text())
    vocab = C.Vocab.load(run / 'vocab.json')
    legal = C.cached_legal()
    engine = Engine(s['engine'])
    link = Link(run, weights=False, index=index)
    _, opponents = link.exchange([], [], {})
    server, address = None, None
    timing = defaultdict(float)
    while True:
        # every game: a learner (re)started after this collector may have new settings (an actor that starts with its
        # learner can read the previous endpoint.json)
        s = endpoint(run)['settings']
        models, judged = matchmaking.pick_models(opponents, matchmaking.evaluation(run) if s.get('eval_frac') else [], s, rng)
        opponent = next((name for name, _ in models if name != 'current'), 'current')
        sid = rng.randrange(len(addresses)) if judged else \
            matchmaking.route_collector(opponent, opponents, rng, len(addresses)) if affinity else index % len(addresses)
        if address != addresses[sid]:
            if server is not None:
                server.close()
            address = addresses[sid]
            server = Client(address, authkey=key)
        pair = rng.choice(pool)
        if s.get('deck'):  # a specialist (settings.json "deck", as in the pool): one learning seat always plays it
            seat = matchmaking.deck_seat(models, judged, rng)
            deck = rng.sample(s['deck'], len(s['deck']))  # in a random order, as humans' decks come
            pair = dict(pair, decks=[deck if o == seat else d for o, d in enumerate(pair['decks'])])
        game = Game(engine, pair, rng.randrange(1 << 30), {0: vocab, 1: vocab}, legal)
        seats = {o: Seat() for o in (0, 1) if models[o][0] == 'current'}
        start = tower_hp(game.state)
        phi = [potential(game.state, start, o) for o in (0, 1)]
        version, shape = None, None
        while True:
            t0 = time.perf_counter()
            state = game.state
            rows = {o: policy.pack(x) for o, x in game.views().items()}
            if shape is None:
                shape = layout(rows[0])
                server.send(('reset', index, shape))  # a new game: zero LSTM states
            t1 = time.perf_counter()
            server.send(('act', index, [(o, *models[o], wire(rows[o], shape)) for o in (0, 1)]))
            replies, used = receive(server)
            version = used if version is None else version  # the game counts as played by its first weights
            t2 = time.perf_counter()
            choices = {}
            for o, (a, lp, v) in enumerate(replies):
                if o in seats:
                    seat = seats[o]
                    seat.rows.append(rows[o])
                    seat.opp.append(rows[1 - o])  # for the critic only
                    seat.action.append(a)
                    seat.logp.append(lp)
                    seat.value.append(v)
                    # ponytail: no overflow penalty when the board is quiet (no enemy troops) — avoids panic deploys like Golem at the back
                    enemy_on_board = any(e.owner != o and e.card != -1 for e in state.entities)
                    seat.reward.append(-OVERFLOW_PENALTY * (state.players[o].elixir >= 100000) * enemy_on_board)
                choices[o] = policy.decode(a, o)
            _, transition = game.step(choices)
            timing['refused'] += transition.executed.count(False)  # rare: refused plays are masked out (policy.playable)
            state = game.state
            ended = game.over
            for o, seat in seats.items():
                new = 0.0 if ended else potential(state, start, o)
                seat.reward[-1] += s['shaping'] * (s['gamma'] * new - phi[o])
                phi[o] = new
            timing['features'] += t1 - t0
            timing['wait'] += t2 - t1
            timing['engine'] += time.perf_counter() - t2
            timing['steps'] += 1
            if ended:
                break
        if judged:  # A's score: 1 win, 0.5 draw, 0 loss
            a = 0 if models[0][0] == judged['a'][0] else 1
            score = 0.5 if state.result not in (0, 1) else float(state.result == a)
            _, opponents = link.exchange([], [dict(eval=judged['id'], score=score, tick=state.tick)],
                                         dict(timing, games=1))
            timing = defaultdict(float)
            continue
        out = []
        for o, seat in seats.items():
            outcome = 0.0 if state.result not in (0, 1) else 1.0 if state.result == o else -1.0
            seat.reward[-1] += outcome + s.get('margin', 0.0) * margin(state, start, o)  # to the critic's short head
            out.append(seat.trajectory(version=version, opponent=models[1 - o][0], outcome=outcome))
        learner = min(seats)
        result = dict(opponent=models[1 - learner][0] if len(seats) == 1 else 'self', tick=state.tick,
                      outcome=0.0 if state.result not in (0, 1) else 1.0 if state.result == learner else -1.0)
        _, opponents = link.exchange(out, [result], dict(timing, games=1))
        timing = defaultdict(float)


ROW_BUCKETS = (8, 16, 32, 64, 128, 256, 512)
OBJECT_BUCKETS = (16, 32, 64, 128)


def _bucket(n, sizes):
    return next(b for b in sizes if b >= n)


def layout(row):
    """The byte layout of packed rows like `row`: (fixed-size keys, per-object keys), each [(key, dtype, shape)]."""
    fixed = [(k, v.dtype.str, v.shape) for k, v in sorted(row.items()) if k not in C.OBJECT_KEYS]
    objects = [(k, v.dtype.str, v.shape[1:]) for k, v in sorted(row.items()) if k in C.OBJECT_KEYS]
    return fixed, objects


def wire(row, shape):
    """A packed row -> (objects, fixed-size bytes, object bytes) for a decision server: unpickling ~20 arrays a row was
    most of a server's time."""
    fixed, objects = shape
    n = len(row['obj_side'])
    objs = np.concatenate([row[k].reshape(n, -1).view(np.uint8) for k, _, _ in objects], 1).tobytes() if n else b''
    return n, b''.join(row[k].tobytes() for k, _, _ in fixed), objs


def _size(dtype, shape):
    return np.dtype(dtype).itemsize * int(np.prod(shape))


def stage(wires, shape, B, O, out=None):
    """wire() rows -> inputs [B, 1, ...]; `out` reuses storage, clearing all padding on every call."""
    fixed, objects = shape
    n = len(wires)
    out = {} if out is None else out
    raw, off = np.frombuffer(b''.join(w[1] for w in wires), np.uint8).reshape(n, -1), 0
    for k, dtype, s in fixed:
        size = _size(dtype, s)
        values = raw[:, off:off + size].copy().view(dtype).reshape(n, *s)
        if k == 'legal':
            values = np.unpackbits(values, -1)[..., :4 * policy.CELLS].reshape(n, 4, C.H, C.W)
            dtype, s = bool, (4, C.H, C.W)
        target = np.int64 if np.dtype(dtype) == np.dtype(np.int32) else dtype
        if k not in out:
            out[k] = np.zeros((B, 1, *s), target)
        else:
            out[k].fill(0)
        out[k][:n, 0] = values
        off += size
    counts = np.array([w[0] for w in wires])
    block = np.zeros((B, O, sum(_size(d, s) for _, d, s in objects)), np.uint8)
    if counts.sum():
        rows = np.repeat(np.arange(n), counts)
        block[rows, np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)] = \
            np.frombuffer(b''.join(w[2] for w in wires), np.uint8).reshape(counts.sum(), -1)
    off = 0
    for k, dtype, s in objects:
        size = _size(dtype, s)
        if k not in out:
            out[k] = np.empty((B, 1, O, *s), np.int64 if np.dtype(dtype) == np.dtype(np.int32) else dtype)
        out[k][:, 0] = block[:, :, off:off + size].copy().view(dtype).reshape(B, O, *s)
        off += size
    return out


class Decider:
    """Batched decisions for many games: each seat's LSTM state is one row of a GPU tensor (one tensor per LSTM size, so
    a server can hold models of several sizes, e.g. medium benchmarks for a large run); one call per model. On a GPU
    each call replays a CUDA graph of the whole decision (forward pass and sampling) for its bucket of rows and objects:
    run op by op, a call launched ~280 kernels, 9 ms of CPU for 0.7 ms of GPU work, whatever the batch."""

    def __init__(self, models, device, core, capacity=1024, fast=True, channels_last=False, frozen_temperature=1.0):
        torch.backends.mha.set_fastpath_enabled(False)  # the learner's attention path: same numerics, and capturable
        self.models, self.device, self.capacity = models, device, capacity
        self.frozen_temperature = frozen_temperature  # card/tile temperature of every model but 'current' (as played)
        self.rows = {}  # (collector, owner) -> row of the LSTM states; row `capacity` is scratch (padding rows)
        self.states = {}  # LSTM size -> (h, c), each [1, capacity + 1, size]
        self.h, self.c = self.state(core)
        for model in models.values():
            self.state(model.config.core)
        self.graphs = {}  # (model name, rows, objects) -> graph and its owned I/O buffers
        self.pool = torch.cuda.graph_pool_handle() if device.type == 'cuda' else None
        self.fast = fast and device.type == 'cuda'
        self.channels_last = channels_last and device.type == 'cuda'
        if self.channels_last:
            for model in models.values():
                policy.convolution_layout(model)
        self.shape = None  # the collectors' wire layout (sent with each 'reset')

    def _rows(self, seats):
        for seat in seats:
            if seat not in self.rows:
                if len(self.rows) >= self.capacity:
                    raise RuntimeError(f'more than {self.capacity} seats on one decision server')
                self.rows[seat] = len(self.rows)
        return [self.rows[seat] for seat in seats]

    def state(self, core):
        """The LSTM states for models of size `core`, made on first use (outside any CUDA graph capture)."""
        if core not in self.states:
            h = torch.zeros(1, self.capacity + 1, core, device=self.device)
            self.states[core] = (h, torch.zeros_like(h))
        return self.states[core]

    def reset(self, collector):
        index = torch.tensor(self._rows([(collector, 0), (collector, 1)]), device=self.device)
        for h, c in self.states.values():
            h[:, index] = 0
            c[:, index] = 0

    def drop(self, name):
        del self.models[name]
        for key in [k for k in self.graphs if k[0] == name]:
            del self.graphs[key]

    def _decide(self, model, x, index):
        # bf16 on the GPU, as in the PPO update, so old and new log-probs come from the same arithmetic
        with torch.no_grad(), torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda',
                                             cache_enabled=False):
            h, c = self.states[model.config.core]
            out = model(x, (h.index_select(1, index), c.index_select(1, index)))
            # the press floor is the learner's exploration only: frozen models play as deployed (live has none)
            action, logp = policy.sample(model, out, x, policy.ABILITY_EXPLORE if model is self.models.get('current') else 0.0)
            h.index_copy_(1, index, out['state'][0].float())
            c.index_copy_(1, index, out['state'][1].float())
        return action, logp.float(), out['value'][:, 0].float()

    def _graph(self, name, staged):
        B, O = staged['obj_side'].shape[0], staged['obj_side'].shape[2]
        key = (name, B, O)
        if key not in self.graphs:
            model = self.models[name]
            x = {k: torch.from_numpy(v).to(self.device) for k, v in staged.items()}
            index = torch.full((B,), self.capacity, device=self.device)  # capture on the scratch row
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up (cuDNN, allocator) off the capture
                for _ in range(2):
                    self._decide(model, x, index)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                outputs = self._decide(model, x, index)
            self.graphs[key] = (x, index, outputs, graph)
        return self.graphs[key]

    def _fast_graph(self, name, staged):
        """One pinned input transfer and one packed output per graph; retain every backing allocation."""
        B, O = staged['obj_side'].shape[0], staged['obj_side'].shape[2]
        key = (name, B, O)
        if key not in self.graphs:
            fields, size = {}, 0
            for k, v in list(staged.items()) + [('_index', np.zeros(B, np.int64))]:
                size = (size + 7) // 8 * 8  # typed views must be naturally aligned
                fields[k] = (size, v.nbytes, v.dtype, v.shape)
                size += v.nbytes
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            device = torch.empty(size, dtype=torch.uint8, device=self.device)
            raw, arrays, x = host.numpy(), {}, {}
            for k, (offset, n, dtype, shape) in fields.items():
                arrays[k] = raw[offset:offset + n].view(dtype).reshape(shape)
                x[k] = device[offset:offset + n].view(torch.from_numpy(np.empty((), dtype)).dtype).reshape(shape)
                arrays[k][...] = self.capacity if k == '_index' else staged[k]
            device.copy_(host)
            index = x.pop('_index')
            model = self.models[name]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._decide(model, x, index)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            # Groups remain live until the aggregate readback. A shared capture pool can reuse a prior
            # group's storage when a new shape is first captured, corrupting its queued output.
            with torch.cuda.graph(graph):
                action, logp, value = self._decide(model, x, index)
                outputs = torch.stack((action.float(), logp, value), 1)
            self.graphs[key] = (host, device, arrays, outputs, graph)
        return self.graphs[key]

    def _readback(self, pending, results):
        """Finish every queued group together; also fences pinned buffers before they can be reused."""
        if not pending:
            return
        values = torch.cat([out for _, out in pending]).cpu().tolist()
        order = [i for part, _ in pending for i in part]
        for i, (action, logp, value) in zip(order, values):
            results[i] = (int(action), logp, value)
        pending.clear()

    def act(self, requests):
        """requests: [(seat, model name, its checkpoint, wire() row)] -> [(action, logp, value)], same order. A model
        is cached by its checkpoint's path (by name without one, as 'current'): names like v000100 recur across runs."""
        by_model = defaultdict(list)
        for i, (_, name, path, _) in enumerate(requests):
            name = path or name
            by_model[name].append(i)
            if name not in self.models:
                self.models[name] = policy.load(path, self.device)[0]
                self.models[name].temperature = self.frozen_temperature  # before its graphs are captured
                self.state(self.models[name].config.core)
                if self.channels_last:
                    policy.convolution_layout(self.models[name])
        results = [None] * len(requests)
        pending, active = [], set()
        for name, idx in by_model.items():
            for s in range(0, len(idx), ROW_BUCKETS[-1]):
                part = idx[s:s + ROW_BUCKETS[-1]]
                rows = [requests[i][3] for i in part]
                B = _bucket(len(rows), ROW_BUCKETS)
                O = _bucket(max(1, max(r[0] for r in rows)), OBJECT_BUCKETS)
                index = self._rows([requests[i][0] for i in part]) + [self.capacity] * (B - len(rows))
                if self.fast:
                    key = (name, B, O)
                    if key in active or key not in self.graphs:
                        # Fence both reused staging buffers and cold captures before touching their memory.
                        self._readback(pending, results)
                        active.clear()
                    if key in self.graphs:
                        host, device, arrays, outputs, graph = self.graphs[key]
                        stage(rows, self.shape, B, O, out=arrays)
                    else:
                        host, device, arrays, outputs, graph = self._fast_graph(name, stage(rows, self.shape, B, O))
                    arrays['_index'][...] = index
                    device.copy_(host, non_blocking=True)
                    graph.replay()
                    pending.append((part, outputs[:len(rows)]))
                    active.add(key)
                    continue
                elif self.pool is None:
                    staged = stage(rows, self.shape, B, O)
                    outputs = self._decide(self.models[name], {k: torch.from_numpy(v).to(self.device)
                                                               for k, v in staged.items()},
                                           torch.tensor(index, device=self.device))
                else:
                    staged = stage(rows, self.shape, B, O)
                    x, static_index, outputs, graph = self._graph(name, staged)
                    for k, v in staged.items():
                        x[k].copy_(torch.from_numpy(v), non_blocking=True)
                    static_index.copy_(torch.tensor(index), non_blocking=True)
                    graph.replay()
                action, logp, value = (o[:len(rows)].tolist() for o in outputs)
                for i, a, lp, v in zip(part, action, logp, value):
                    results[i] = (a, lp, v)
        self._readback(pending, results)
        return results


WEIGHTS_POLL = 5  # s between a decision server's asks for newer weights
SERVED = ('learner_temperature', 'learner_mix', 'frozen_temperature', 'channels_last')  # a server applies them once


def served(run):
    """The settings a decision server fixes at startup, as the learner in endpoint.json has them now."""
    settings = endpoint(run)['settings']
    return {k: settings.get(k) for k in SERVED}


def server(sid, run, listener, gpu, mps, fast=True):
    """A decision server: answers whichever collectors are waiting, in one batch, as soon as the last batch is done (it
    never holds a batch back for stragglers: collectors compute while the GPU decides)."""
    redirect(run, f'server-{sid}')
    torch.set_num_threads(1)
    if mps:  # before CUDA starts in this process
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = mps
    device = torch.device('cuda', gpu) if gpu is not None else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    current = policy.load(endpoint(run)['init'], device)[0]
    link = Link(run)
    weights, opponents = link.exchange([], [], {})
    settings = endpoint(run)['settings']
    applied = {k: settings.get(k) for k in SERVED}
    current.temperature = settings.get('learner_temperature', 1.0)
    current.mix = settings.get('learner_mix', 0.0)
    decider = Decider({'current': current}, device, current.config.core, fast=fast,
                      channels_last=settings.get('channels_last', False),
                      frozen_temperature=settings.get('frozen_temperature', 1.0))
    conns, arrivals, polled, stats = [], queue.Queue(), time.monotonic(), defaultdict(float)

    def accept():
        while True:
            try:
                arrivals.put(listener.accept())
            except Exception as error:  # noqa: BLE001
                print(f'refused a connection: {error!r}', flush=True)
    threading.Thread(target=accept, daemon=True).start()
    while True:
        if weights is not None:
            current.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            judged = {m[k][1] for m in matchmaking.evaluation(run) for k in ('a', 'b')}
            for name in set(decider.models) - {'current'} - {path for path, _ in opponents.values()} - judged:
                decider.drop(name)  # an opponent the learner no longer offers
            weights = None
        if time.monotonic() - polled > WEIGHTS_POLL:
            weights, opponents = link.exchange([], [], {})
            polled = time.monotonic()
            if served(run) != applied:  # a learner restarted with new ones: the actor job restarts this server with them
                print(f'settings changed ({applied} -> {served(run)}); exiting to restart', flush=True)
                os._exit(0)
            if stats['batches']:
                print(json.dumps({k: round(v, 3) for k, v in stats.items()}), flush=True)
            stats = defaultdict(float)
        while not arrivals.empty():
            conns.append(arrivals.get())
        if not conns:
            time.sleep(0.1)
            continue
        waiting = []
        for conn in wait(conns, timeout=1):
            try:
                kind, index, payload = conn.recv()
            except (EOFError, OSError):  # its collector died (and will reconnect when restarted)
                conns.remove(conn)
                continue
            if kind == 'reset':
                decider.shape = payload
                decider.reset(index)
            else:
                waiting.append((conn, [((index, o), name, path, row) for o, name, path, row in payload]))
        if waiting:
            t0 = time.perf_counter()
            requests = [r for _, payload in waiting for r in payload]
            replies = decider.act(requests)
            k = 0
            for conn, payload in waiting:
                conn.send((replies[k:k + len(payload)], link.version))
                k += len(payload)
            stats['batches'] += 1
            stats['rows'] += k
            stats['model_calls'] += len({r[2] or r[1] for r in requests})
            stats['busy_s'] += time.perf_counter() - t0


def start_mps():
    """NVIDIA MPS, so this job's processes on a GPU share it instead of taking turns. One daemon per job, ending with
    it. Returns its pipe directory, or None without MPS."""
    if not shutil.which('nvidia-cuda-mps-control'):
        return None
    root = Path(f"/tmp/cr-mps-{os.environ.get('SLURM_JOB_ID', os.getuid())}")
    for d in ('pipe', 'log'):
        (root / d).mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-cuda-mps-control', '-d'], capture_output=True,  # fails harmlessly if already running
                   env=dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(root / 'pipe'), CUDA_MPS_LOG_DIRECTORY=str(root / 'log')))
    return str(root / 'pipe')


def _child(target, *args):
    """target(*args); on any error its traceback in the log and a quick exit (no native exit-time destructors)."""
    try:
        target(*args)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def act(args):
    """This job's decision servers (one per --servers, on its GPUs) and collectors; restarts any that dies (the engine
    can segfault)."""
    raise_thread_limit()
    import resource
    print(f'thread limit {resource.getrlimit(resource.RLIMIT_NPROC)}', flush=True)
    run = args.run.expanduser().resolve()
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    while not (run / 'vocab.json').exists():  # written by the learner
        time.sleep(10)
    # this parent never touches CUDA (its forked children could not use it): the GPUs are counted, not opened
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = [int(g) for g in args.gpus.split(',')] if args.gpus else \
        list(range(len(visible.split(',')))) if visible else []
    servers = args.servers or max(1, len(gpus))
    mps = start_mps() if gpus and args.mps and servers > len(gpus) else None
    key = os.urandom(16)
    job = os.environ.get('SLURM_JOB_ID', os.getpid())
    listeners = [Listener(f'/tmp/cr-ppo-{job}-{sid}.sock', 'AF_UNIX', authkey=key) for sid in range(servers)]
    ctx = mp.get_context('fork')

    def launch(kind, i):
        target = (server, i, run, listeners[i], gpus[i % len(gpus)] if gpus else None, mps, args.fast_inference) if kind == 'server' else \
            (collector, i, run, [listener.address for listener in listeners], key, args.opponent_affinity)
        p = ctx.Process(target=_child, args=target, daemon=True)
        p.start()
        return p
    workers = {('server', i): launch('server', i) for i in range(servers)}
    workers.update({('collector', i): launch('collector', i) for i in range(args.collectors)})
    print(f'{servers} decision servers on GPUs {gpus}{" (MPS)" if mps else ""}, {args.collectors} collectors; '
          f'logs in {run}/logs', flush=True)
    while True:
        time.sleep(10)
        for (kind, i), w in workers.items():
            if w.exitcode is not None:
                print(f'{kind} {i} exited ({w.exitcode}); restarting it', flush=True)
                workers[(kind, i)] = launch(kind, i)
