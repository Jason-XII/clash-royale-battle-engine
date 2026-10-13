"""The playing side of il.rl: collectors (one engine each, the live deploy delay, FirstLight's tower shaping), decision
servers (batched GPU decisions for every collector, each seat's LSTM state kept on the GPU) and `il.rl join`, another
job's collectors (and, with --servers, decision servers) for a learner started with --remote.

One process per collector plays the learner's games: a spec (decks, seed, which model plays each seat, which server
answers) in, a trajectory per learning seat out. Each decision's inputs are built in the collector (il.policy.Inputs:
clapha-il's PlayerView on ClaphaEngine.record()) and sent packed (il.policy.pack). Decision servers (--servers
processes) answer their share of collectors in batched GPU calls with the learner's latest weights, published through
shared memory (one Python thread answering every decision was the bottleneck: GPU 17% busy, that thread 100%), and run
under NVIDIA MPS.
Reward: +1 win / -1 loss at the end, -OVERFLOW_PENALTY per decision at full elixir, and FirstLight's tower shaping:
spec shaping * (gamma * potential' - potential), the potential being crowns taken minus lost plus each tower's lost hp
fraction (king x3), + for theirs and - for ours, and 0 once the game ends (so it adds no bonus for a fast three-crown
or a crown margin, only credit for damage dealt and prevented when it happens).
"""
import argparse
from collections import defaultdict
import json
import multiprocessing as mp
from multiprocessing.connection import Client, wait
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from . import defense
from . import policy
from .defense import board_rows

REPLY_TIMEOUT = 180  # s; the learner can be busy with an update (~20 s). A node that dies sends nothing, so a
                     # collector that waits longer treats the connection as gone and reconnects (il.rl join).
KING_WEIGHT = 3.0  # the king tower's share of the shaping potential (FirstLight)
OVERFLOW_PENALTY = 0.002


def tower_hp(state):
    """{tower id: (owner, hp, king)}; a destroyed tower is absent."""
    return {e.id: (e.owner, max(e.hp or 0, 0), e.x == 9000) for e in state.entities
            if e.card == -1 and e.entity_type == 'building'}


def damage(state, start, owner):
    """(dealt, taken) for `owner`: crowns plus each tower's lost hp fraction (king x KING_WEIGHT), theirs and ours;
    `start` is tower_hp at the start."""
    towers = tower_hp(state)
    out = [float(state.crowns[owner]), float(state.crowns[1 - owner])]
    for tid, (o, full, king) in start.items():
        lost = 1.0 - (towers[tid][1] / max(full, 1) if tid in towers else 0.0)
        out[o == owner] += (KING_WEIGHT if king else 1.0) * lost
    return tuple(out)


def potential(state, start, owner):
    """FirstLight's shaping potential for `owner` (see the module docstring)."""
    dealt, taken = damage(state, start, owner)
    return dealt - taken


def opponent_dir(args):
    """Where the frozen opponents come from: this run's own snapshots, or --opponent-pool."""
    return Path(args.opponent_pool).expanduser() if args.opponent_pool else Path(args.out) / 'pool'


# ---- collectors ---------------------------------------------------------------------------------------------------------
def collector_engine(name, out, clapha, stagger):
    """A collector process's log, CPU priority and booted engine (+ the run's vocabulary and legal tiles)."""
    log = os.open(f'{out}/logs/collector-{name}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    time.sleep(stagger)  # engines boot a few at a time
    os.nice(5)  # the decision servers are the bottleneck on a busy node: let them run first
    torch.set_num_threads(1)
    from native_engine.clapha import ClaphaEngine
    return ClaphaEngine(clapha), policy.FE.Vocab.load(Path(out) / 'vocab.json'), policy.legal_fn()


def collector(index, args, learner, servers):
    """A collector on the learner's node, connected by pipes (one to every decision server)."""
    engine, vocab, legal = collector_engine(index, args.out, args.clapha, index // 4 * 8)
    play_games(index, args.delay, engine, vocab, legal, learner, servers)
    for conn in servers:
        conn.close()
    os._exit(0)  # the engine segfaults in its exit-time destructors


def receive(conn):
    """conn.recv(), but a peer that stays silent for REPLY_TIMEOUT counts as gone (a failed node never closes)."""
    if not conn.poll(REPLY_TIMEOUT):
        raise EOFError(f'no reply for {REPLY_TIMEOUT} s')
    return conn.recv()


def play_games(index, delay, engine, vocab, legal, learner, servers):
    """Plays the learner's games until it sends None: games and trajectories through the learner (`learner`),
    actions from the decision server the learner names for each game (one of `servers`: a frozen opponent lives on
    one server, so its games go there). Local pipes or TCP connections (il.rl join) alike."""
    from native_engine import ABILITY, Play
    from .arena import make_match
    lag = delay if isinstance(delay, int) else sum(delay) // 2  # for il.defense: about when plays land
    while True:
        learner.send(('new_game', index, None))
        spec = receive(learner)
        if spec is None:
            break
        match = make_match(spec['decks'], spec['seed'])
        models = spec['models']  # per seat: 'current' or a frozen opponent's key
        conn = servers[spec['server'] % len(servers)]  # il.rl join --servers may run fewer servers than the learner
        steps = [defaultdict(list) for _ in (0, 1)]
        state = engine.reset(match=match)
        inputs = policy.Inputs(engine, match, {0: vocab, 1: vocab}, legal)
        expired = engine.expired
        shaping, gamma = spec.get('shaping', (0.0, 1.0))
        start = tower_hp(state)
        phi = [potential(state, start, o) for o in (0, 1)]
        frames, landed = [], []  # for il.defense: the board every decision, our plays where they land
        timing = defaultdict(float)  # seconds per part of a decision, for the learner's log
        while True:
            t0 = time.perf_counter()
            rows = {o: policy.pack(x) for o, x in inputs(state).items()}
            frames.append((state.tick, board_rows(state)))
            t1 = time.perf_counter()
            conn.send(('act', index, [((index, o), models[o], rows[o]) for o in (0, 1)]))
            replies = receive(conn)
            t2 = time.perf_counter()
            timing['features'] += t1 - t0
            timing['wait'] += t2 - t1
            timing['decisions'] += 1
            plays = []
            for o, (action, logp, value) in enumerate(replies):
                full = state.players[o].elixir >= 100000
                s = steps[o]
                s['rows'].append(rows[o])
                s['action'].append(action)
                s['logp'].append(logp)
                s['value'].append(value)
                s['reward'].append(-OVERFLOW_PENALTY * full)
                s['full'].append(full)
                choice = policy.decode(action, o)
                if choice[0] == 'ability':
                    plays.append(Play(o, ABILITY, 0, 0))
                elif choice[0] == 'card':
                    plays.append(Play(o, *choice[1:]))
            if state.finalized or state.tick >= 7200:
                break
            t3 = time.perf_counter()
            transition = engine.step(plays, 5, delay=delay)
            timing['dropped'] += transition.executed.count(False)  # plays the engine refused
            landed += [(state.tick + lag, p.owner, p.x, p.y) for p, ok in zip(plays, transition.executed)
                       if ok and p.slot != ABILITY]
            state = transition.state
            timing['engine'] += time.perf_counter() - t3
            ended = state.finalized or state.tick >= 7200
            for o in (0, 1):
                new = 0.0 if ended else potential(state, start, o)
                steps[o]['reward'][-1] += shaping * (gamma * new - phi[o])
                phi[o] = new
        timing['expired'] = engine.expired - expired
        guard = defense.game(frames, landed)
        for o in (0, 1):
            outcome = 0.0 if state.result not in (0, 1) else (1.0 if state.result == o else -1.0)
            steps[o]['reward'][-1] += outcome
            if o in spec.get('train', [s for s in (0, 1) if models[s] == 'current']):
                trajectory = {k: np.array(v) for k, v in steps[o].items() if k != 'rows'}
                trajectory = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in trajectory.items()}
                trajectory['inputs'] = policy.stack(steps[o]['rows'])
                learner.send(('trajectory', index, dict(trajectory, outcome=outcome, opponent=models[1 - o], timing=timing,
                                                         defense=guard[o], tag=spec.get('tag'))))
                timing = {}  # counted once per game
        conn.send(('reset', index, [(index, 0), (index, 1)]))


# ---- decision servers ------------------------------------------------------------------------------------------------------------
def load_policy(path, device, vocab=None):
    """A checkpoint's model and saved dict; with `vocab`, the model must read its inputs with that vocabulary (every
    model of a run sees the same collectors' inputs)."""
    model, own, saved = policy.load(path, device)
    if vocab is not None and own.index != vocab.index:
        raise ValueError(f'{path} uses another vocabulary than this run')
    return model, saved


class Server:
    """Batched decisions for every collector. Each seat's LSTM state lives in one row of a preallocated GPU tensor. One
    decision of one model gathers its seats' states, runs the model, writes the states back and samples
    (il.policy.sample)."""

    def __init__(self, models, device, core, capacity=256):
        self.models, self.device, self.core = models, device, core
        self.rows = {}  # seat -> row in self.h / self.c
        self.h = self.c = None
        self._allocate(capacity)

    def _allocate(self, capacity):
        h = torch.zeros(1, capacity, self.core, device=self.device)
        c = torch.zeros_like(h)
        if self.h is not None:
            h[:, :self.capacity], c[:, :self.capacity] = self.h, self.c
        self.h, self.c, self.capacity = h, c, capacity

    def _rows(self, seats):
        for seat in seats:
            if seat not in self.rows:
                self.rows[seat] = len(self.rows)
        if len(self.rows) > self.capacity:
            self._allocate(2 * len(self.rows))
        return torch.tensor([self.rows[seat] for seat in seats], device=self.device)

    @torch.no_grad()
    def act(self, requests):
        """requests: [(seat, model key, packed input row)] -> [(action, logp, value)] in the same order."""
        by_model = defaultdict(list)
        for i, (seat, key, row) in enumerate(requests):
            by_model[key].append(i)
        results = [None] * len(requests)
        for key, idx in by_model.items():
            model, index = self.models[key], self._rows([requests[i][0] for i in idx])
            x = policy.tensors({k: v[:, None] for k, v in policy.stack([requests[i][2] for i in idx]).items()}, self.device)
            state = self.h.index_select(1, index), self.c.index_select(1, index)
            # bf16 on the GPU, as in the PPO update, so old and new log-probs come from the same arithmetic
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda'):
                out = model(x, state)
                action, logp = policy.sample(model, out, x, policy.ABILITY_EXPLORE)
            self.h.index_copy_(1, index, out['state'][0].float())
            self.c.index_copy_(1, index, out['state'][1].float())
            for i, a, lp, v in zip(idx, action.tolist(), logp.float().tolist(), out['value'][:, 0].float().tolist()):
                results[i] = (a, lp, v)
        return results

    def reset(self, seats):
        """A seat's game ended: its next game starts from a zero LSTM state."""
        known = [seat for seat in seats if seat in self.rows]
        if known:
            rows = torch.tensor([self.rows[seat] for seat in known], device=self.device)
            self.h[:, rows] = 0
            self.c[:, rows] = 0


def pick_device(name):
    """Called only after the fork: a process that touched CUDA (even torch.cuda.is_available()) cannot fork
    children that use CUDA ('Cannot re-initialize CUDA in forked subprocess')."""
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)


def start_mps():
    """NVIDIA MPS for the decision servers (the learner stays outside it). Processes sharing a GPU otherwise take turns:
    4 servers on an A100 answered 12.4k decisions/s, the same as one; under MPS their small CUDA graphs run side by
    side (24.7k; 8 servers 34.6k). One daemon per job, left running for the job's next run (it ends with the job);
    a server finding no daemon runs without MPS. Returns its pipe directory, or None on a node without MPS."""
    if not shutil.which('nvidia-cuda-mps-control'):
        return None
    root = Path(f"/tmp/cr-mps-{os.environ.get('SLURM_JOB_ID', os.getuid())}")
    for d in ('pipe', 'log'):
        (root / d).mkdir(parents=True, exist_ok=True)
    subprocess.run(['nvidia-cuda-mps-control', '-d'], capture_output=True,  # fails harmlessly if already running
                   env=dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(root / 'pipe'), CUDA_MPS_LOG_DIRECTORY=str(root / 'log')))
    return str(root / 'pipe')


def accept_into(listener, arrivals):
    """A daemon thread putting every connection `listener` accepts into the queue `arrivals`."""
    def run():
        while True:
            try:
                arrivals.put(listener.accept())
            except Exception as error:  # noqa: BLE001 - a failed handshake (wrong key, port scan) is not fatal
                print(f'refused a connection: {error!r}', flush=True)
    threading.Thread(target=run, daemon=True).start()


def reachable_ip():
    """This node's address on the cluster network (no packet is sent)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(('10.255.255.255', 1))
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


STATS = ('receive', 'decide', 'reply', 'batches', 'requests')  # per decision server, summed into the log


def decision_server(sid, args, conns, shared, version, lock, snapshots, stats, listener=None):
    """Answers its collectors' decisions in batched GPU calls with the learner's latest weights (`shared`,
    re-read whenever `version` changes) and the frozen opponents (`snapshots` announces new ones)."""
    log = os.open(f'{args.out}/logs/server-{socket.gethostname()}-{sid}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    torch.set_num_threads(1)
    if args.mps:  # before CUDA starts in this process
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = args.mps
    device = pick_device(args.server_device or args.device)
    if device.type == 'cuda':
        if device.index is None:  # 'cuda' (--device auto): torch.cuda.set_device needs an index
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)  # CUDA graphs and streams use the current device
    vocab = policy.FE.Vocab.load(Path(args.out) / 'vocab.json')
    model, _ = load_policy(args.init, device, vocab)  # the architecture; weights come from `shared`

    def refresh():
        with lock:
            model.load_state_dict(shared)
            seen = version.value
        print(f'weights version {seen}: first parameter sum {next(model.parameters()).sum().item():.6f}', flush=True)
        return seen

    seen = refresh()
    pool = {'il': args.il or args.init, **{q.stem: str(q) for q in sorted(opponent_dir(args).glob('*.pt'))}}
    server = Server({'current': model.eval(), **{k: load_policy(v, device, vocab)[0] for k, v in pool.items()}},
                    device, model.config.core)
    owners, first, window = [], None, args.batch_ms / 1000  # [(collector pipe, its requests)]
    # Collectors with a game on this server (asked since their last reset). Every collector is connected to every
    # server, so "all connections waiting" never happened and every round waited the whole window; a round is now
    # sent as soon as every collector playing here waits.
    active = set()
    arrivals = queue.Queue()  # remote collectors (il.rl join) joining at any time
    if listener is not None:
        accept_into(listener, arrivals)
    while True:
        while not arrivals.empty():
            conns.append(arrivals.get())
        if not conns:
            time.sleep(0.05)
            continue
        if version.value != seen:
            seen = refresh()
        while not snapshots.empty():
            path = Path(snapshots.get())
            server.models[path.stem] = load_policy(path, device, vocab)[0]
        t0 = time.perf_counter()
        timeout = 1 if first is None else max(0.0, window - (t0 - first))
        for conn in wait(conns, timeout=timeout):
            try:
                kind, index, payload = conn.recv()
            except (EOFError, OSError):  # its collector died, finished or (remote) lost its job
                conns.remove(conn)
                active.discard(conn)
                owners = [(c, r) for c, r in owners if c is not conn]
                continue
            if kind == 'act':
                owners.append((conn, payload))
                active.add(conn)
                first = first or time.perf_counter()
            elif kind == 'reset':
                server.reset(payload)
                active.discard(conn)
        t1 = time.perf_counter()
        stats[sid * len(STATS)] += t1 - t0
        if owners and (t1 - first >= window or len(owners) >= len(active)):
            requests = [r for _, payload in owners for r in payload]
            for key in {r[1] for r in requests} - server.models.keys():  # a snapshot newer than this server (il.rl join)
                server.models[key] = load_policy(opponent_dir(args) / f'{key}.pt', device, vocab)[0]
            replies = server.act(requests)
            t2 = time.perf_counter()
            k = 0
            for conn, payload in owners:
                conn.send(replies[k:k + len(payload)])
                k += len(payload)
            for j, value in enumerate((0, t2 - t1, time.perf_counter() - t2, 1, len(requests))):
                stats[sid * len(STATS) + j] += value
            owners, first = [], None


def endpoint(run):
    """Remote collectors: The learner's current address and key, waiting until a learner has written them."""
    path = run / 'endpoint.json'
    while True:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(10)


def remote(slot, run, local=None):
    """One remote collector, for good. `local`: pipes to this job's decision servers (il.rl join --servers), else it
    uses the learner's over the network."""
    from native_engine.clapha import raise_thread_limit
    raise_thread_limit()
    first = endpoint(run)
    name = f'{socket.gethostname()}-{os.getppid()}-{slot}'
    engine, vocab, legal = collector_engine(name, first['out'], first['clapha'], slot // 4 * 8)
    while True:
        learner, servers = None, []
        try:
            ep = endpoint(run)
            key = bytes.fromhex(ep['authkey'])
            learner = Client(tuple(ep['learner']), authkey=key)
            learner.send(('hello', None, None))
            hello = receive(learner)
            servers = local or [Client(tuple(address), authkey=key) for address in ep['servers']]
            print(f'connected to {ep["host"]} as collector {hello["index"]}', flush=True)
            play_games(hello['index'], ep['delay'], engine, vocab, legal, learner, servers)
            print('the learner stopped; waiting for the next one', flush=True)
        except (OSError, EOFError) as error:  # learner gone, restarting, or not up yet
            print(f'not connected ({error!r}); retrying', flush=True)
        finally:
            for c in [learner, *([] if local else servers)]:
                if c is not None:
                    c.close()
        time.sleep(15)


WEIGHTS_POLL = 5  # s between il.rl join --servers' asks for newer weights (an update takes ~30 s)


def local_servers(ctx, run, args):
    """il.rl join --servers: decision servers in this job, on its GPU, so its collectors' decisions take nothing from
    the learner's node. They keep the learner's latest weights in shared memory, as the learner's own servers do; the
    weights come over the network (no file: a shared file system may serve a stale latest.pt for a while). Returns each
    collector's pipes to the servers and fetch(), which copies newer weights in (call it every WEIGHTS_POLL s)."""
    link = {'conn': None, 'seen': None}  # connection to the learner, its weights version we hold

    def fetch():
        """Newer weights from the learner -> (ep, numpy state dict), or None (none newer, or not connected)."""
        try:
            if link['conn'] is None:
                ep = endpoint(run)
                link['conn'] = Client(tuple(ep['learner']), authkey=bytes.fromhex(ep['authkey']))
                link['seen'], link['ep'] = None, ep  # a new learner: its versions count from 0
            link['conn'].send(('weights', None, link['seen']))
            reply = receive(link['conn'])
        except (OSError, EOFError):  # learner gone or restarting
            link['conn'] = None
            return None
        if reply is None:
            return None
        link['seen'] = reply[0]
        return link['ep'], reply[1]

    while (first := fetch()) is None:
        time.sleep(WEIGHTS_POLL)
    link['conn'].close()  # reconnected after the fork: the children must not hold this socket
    link['conn'] = None
    ep, weights = first
    shared = {k: torch.from_numpy(v).share_memory_() for k, v in weights.items()}
    version, lock = ctx.Value('i', 0), ctx.Lock()
    mps = start_mps() if args.mps else None
    server_args = argparse.Namespace(out=ep['out'], init=ep['init'], il=ep['il'], opponent_pool=ep['opponent_pool'],
                                     batch_ms=ep['batch_ms'], server_device=args.server_device, device='auto', mps=mps)
    pipes = [[ctx.Pipe() for _ in range(args.servers)] for _ in range(args.collectors)]  # [collector][server]
    stats = ctx.Array('d', args.servers * len(STATS))  # read by nobody: the learner logs its own servers
    for sid in range(args.servers):
        ctx.Process(target=decision_server, daemon=True, args=(
            sid, server_args, [row[sid][0] for row in pipes], shared, version, lock, ctx.Queue(), stats)).start()

    def update():
        got = fetch()
        if got is not None:
            with lock:
                for k, v in got[1].items():
                    shared[k].copy_(torch.from_numpy(v))
                version.value += 1
    print(f"{args.servers} decision servers on {args.server_device}{' (MPS)' if mps else ''}", flush=True)
    return [[row[sid][1] for sid in range(args.servers)] for row in pipes], update


def join():
    """Collectors in another job: they find the learner through <run>/endpoint.json (written by il.rl --remote on a
    shared file system), connect over the cluster network and play its games. When the learner stops or restarts
    they wait and reconnect, re-reading the file; the learner simply loses them when this job ends. With --servers
    (a GPU job) their decisions come from this job's own decision servers, else from the learner's."""
    parser = argparse.ArgumentParser(prog='python -m il.rl join')
    parser.add_argument('run', help='the il.rl --remote output directory (on the shared file system)')
    parser.add_argument('--collectors', type=int, default=30)
    parser.add_argument('--servers', type=int, default=0, help="this job's own decision servers; 0: the learner's")
    parser.add_argument('--server-device', default='cuda')
    parser.add_argument('--mps', action=argparse.BooleanOptionalAction, default=True, help='NVIDIA MPS for --servers')
    args = parser.parse_args(sys.argv[2:])
    run = Path(args.run).expanduser().resolve()
    ctx = mp.get_context('fork')
    local, update = [None] * args.collectors, None
    if args.servers:
        local, update = local_servers(ctx, run, args)
    workers = [ctx.Process(target=remote, args=(i, run, local[i]), daemon=True) for i in range(args.collectors)]
    for w in workers:
        w.start()
    print(f'{args.collectors} collectors for {run}; logs in {run}/logs/collector-{socket.gethostname()}-*', flush=True)
    while True:  # keep the job alive; restart any collector that dies outright; keep this job's servers' weights fresh
        for i, w in enumerate(workers):
            if w.exitcode is not None:
                print(f'collector {i} exited ({w.exitcode}); restarting it', flush=True)
                workers[i] = ctx.Process(target=remote, args=(i, run, local[i]), daemon=True)
                workers[i].start()
        if update:
            update()
        time.sleep(WEIGHTS_POLL)
