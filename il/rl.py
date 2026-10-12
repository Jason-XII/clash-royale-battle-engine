"""PPO self-play on clapha engines (x86_64 Linux), starting from an IL checkpoint.

    PYTHONPATH=. python -m il.rl ~/clapha-engine ~/cr-engine/runs/il-002/best.pt ~/cr-engine/runs/ppo-general --collectors 24 --servers 4
    PYTHONPATH=. python -m il.rl join ~/cr-engine/runs/ppo-general --collectors 30   # more collectors from another job
                                                                                     # (the learner needs --remote)

One process per collector plays games on its own engine with the live deploy delay. Decision servers
(--servers processes, GPU) answer their share of collectors in batches and keep each seat's LSTM state;
the learner process picks opponents, runs PPO and publishes new weights to the servers through shared
memory (one Python thread answering every decision was the bottleneck: GPU 17% busy, that thread 100%). Half the games are self-play (both seats learn); the other half are against a frozen pool
(the IL model and snapshots), chosen by PFSP: weight (1 - win rate)^2 + 0.05.

Policy: wait with the model's p(wait); otherwise a card x cell sampled from the non-wait logits at
temperature TEMPERATURE (near the IL agent's "best card and cell", but stochastic so PPO can learn it).
The ability action is masked (clapha cannot play it). A KL penalty keeps the policy near the IL model.
Reward: +1 win / -1 loss at the end, +-CROWN_REWARD per crown taken / lost, -OVERFLOW_PENALTY per
decision at full elixir.
"""
import argparse
from collections import defaultdict, deque
import json
import math
import multiprocessing as mp
from multiprocessing.connection import Client, Listener, wait
import queue
import socket
import sys
import threading
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .model import INPUT_KEYS, Policy, load_weights
from .train import collate

TEMPERATURE = 0.5
HOLD_LIMIT = 60  # s a collector's next game may be held back for a full backlog (well under REPLY_TIMEOUT)
REPLY_TIMEOUT = 180  # s; the learner can be busy with an update (~20 s). A node that dies sends nothing, so a
                     # collector that waits longer treats the connection as gone and reconnects (il.rl join).
CROWN_REWARD = 0.1
OVERFLOW_PENALTY = 0.002


# ---- the policy distribution -----------------------------------------------------------------------------------------
ABILITY_EXPLORE = 0.01  # training: P(ability) >= this whenever it is ready (PPO stopped pressing it at all, unexplored)


def distribution(logits, temperature=TEMPERATURE, explore=0.0):
    """[..., ACTIONS] logits -> (log p(wait), log p(play), conditional log-probs over the play options [..., 1 + 4*CELLS]:
    the ability, then cards x cells). The model masks what cannot be played (-inf), the ability when it is not ready.
    A step with nothing playable waits with probability 1."""
    logits = logits.float()
    options = logits[..., 1:]
    log_wait = logits.log_softmax(-1)[..., 0]
    playable = torch.isfinite(options).any(-1)
    log_play = torch.where(playable, torch.log1p(-log_wait.exp().clamp(max=1 - 1e-6)), torch.full_like(log_wait, -1e9))
    log_wait = torch.where(playable, log_wait, torch.zeros_like(log_wait))
    cond = torch.where(playable.unsqueeze(-1), options / temperature, torch.zeros_like(options)).log_softmax(-1)
    if explore:  # mix explore x "press the ability" into the whole distribution where the ability is ready
        ready = torch.isfinite(options[..., 0])
        keep = math.log1p(-explore)  # Python floats: a tensor made here is a CPU -> GPU copy, which breaks CUDA graphs
        new_wait = keep + log_wait
        new_play = torch.log1p(-new_wait.exp().clamp(max=1 - 1e-6))
        ability = torch.logaddexp(keep + log_play + cond[..., 0], torch.full_like(log_wait, math.log(explore))) - new_play
        rest = keep + log_play.unsqueeze(-1) + cond[..., 1:] - new_play.unsqueeze(-1)
        cond = torch.where(ready.unsqueeze(-1), torch.cat([ability.unsqueeze(-1), rest], -1), cond)
        log_wait = torch.where(ready, new_wait, log_wait)
        log_play = torch.where(ready, new_play, log_play)
    return log_wait, log_play, cond


def log_prob(dist, action):
    log_wait, log_play, cond = dist
    option = (action - 1).clamp(min=0)
    return torch.where(action == D.WAIT, log_wait, log_play + cond.gather(-1, option.unsqueeze(-1)).squeeze(-1))


def entropy(dist):
    log_wait, log_play, cond = dist
    p_wait, p_play = log_wait.exp(), log_play.exp()
    cond_entropy = -(cond.exp() * torch.nan_to_num(cond, neginf=0.0)).sum(-1)
    return -(p_wait * log_wait + p_play * torch.where(p_play > 0, log_play, torch.zeros_like(log_play))) + p_play * cond_entropy


def kl(dist, ref):
    """KL(policy || reference), both in this factorised form."""
    (lw, lp, c), (rw, rp, rc) = dist, ref
    pw, pp = lw.exp(), lp.exp()
    cond_kl = (c.exp() * (torch.nan_to_num(c, neginf=0.0) - torch.nan_to_num(rc, neginf=0.0))).sum(-1)
    return pw * (lw - rw) + pp * torch.where(pp > 0, lp - rp + cond_kl, torch.zeros_like(lp))


def sample(dist):
    log_wait, log_play, cond = dist
    play = torch.rand_like(log_wait) < log_play.exp()
    option = (cond - torch.empty_like(cond).exponential_().log()).argmax(-1)  # Gumbel-max: a draw from softmax(cond)
    return torch.where(play, option + 1, torch.zeros_like(option))


# ---- collectors ---------------------------------------------------------------------------------------------------------
def collector_engine(name, out, clapha, stagger):
    """A collector process's log, CPU priority and booted engine (+ the run's vocab)."""
    log = os.open(f'{out}/logs/collector-{name}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    time.sleep(stagger)  # engines boot a few at a time
    os.nice(5)  # the decision servers are the bottleneck on a busy node: let them run first
    torch.set_num_threads(1)
    from native_engine.clapha import ClaphaEngine
    return ClaphaEngine(clapha), D.load_vocab(Path(out) / 'vocab.json')


def collector(index, args, learner, servers):
    """A collector on the learner's node, connected by pipes (one to every decision server)."""
    engine, vocab = collector_engine(index, args.out, args.clapha, index // 4 * 8)
    play_games(index, args.delay, engine, vocab, learner, servers)
    for conn in servers:
        conn.close()
    os._exit(0)  # the engine segfaults in its exit-time destructors


def receive(conn):
    """conn.recv(), but a peer that stays silent for REPLY_TIMEOUT counts as gone (a failed node never closes)."""
    if not conn.poll(REPLY_TIMEOUT):
        raise EOFError(f'no reply for {REPLY_TIMEOUT} s')
    return conn.recv()


def play_games(index, delay, engine, vocab, learner, servers):
    """Plays the learner's games until it sends None: games and trajectories through the learner (`learner`),
    actions from the decision server the learner names for each game (one of `servers`: a frozen opponent lives on
    one server, so its games go there). Local pipes or TCP connections (il.rl join) alike."""
    from native_engine import ABILITY, Play
    from .data import InFlight, Observer
    from .arena import make_match
    while True:
        learner.send(('new_game', index, None))
        spec = receive(learner)
        if spec is None:
            break
        match = make_match(spec['decks'], spec['seed'])
        models = spec['models']  # per seat: 'current' or a frozen opponent's key
        conn = servers[spec['server']]
        observer = Observer(vocab)  # both seats' rows, built once per state
        steps = [defaultdict(list) for _ in (0, 1)]
        state = engine.reset(match=match)
        expired = engine.expired
        crowns = (0, 0)
        timing = defaultdict(float)  # seconds per part of a decision, for the learner's log
        while True:
            t0 = time.perf_counter()
            # A player with a play or ability press in flight can do nothing else (ClaphaEngine drops it): the
            # Observer masks everything for that seat, so what is stored is what happened (wait, log-prob 0).
            rows = observer.observe(state, [InFlight(p['owner'], p['card'], p['cost'], p['due']) for p in engine.pending])
            t1 = time.perf_counter()
            conn.send(('act', index, [((index, o), models[o], pack(rows[o])) for o in (0, 1)]))
            replies = receive(conn)
            t2 = time.perf_counter()
            timing['features'] += t1 - t0
            timing['wait'] += t2 - t1
            timing['decisions'] += 1
            plays = []
            for o, (action, logp, value) in enumerate(replies):
                full = state.players[o].elixir >= 100000
                s = steps[o]
                for k, v in rows[o].items():
                    s[k].append(v)
                s['action'].append(action)
                s['logp'].append(logp)
                s['value'].append(value)
                s['reward'].append(-OVERFLOW_PENALTY * full)
                s['full'].append(full)
                if action == D.ABILITY:
                    plays.append(Play(o, ABILITY, 0, 0))
                elif action >= 2:
                    slot, cell = divmod(action - 2, D.CELLS)
                    x, y = D.cell_center(cell, o)
                    plays.append(Play(o, int(slot), int(x), int(y)))
            if state.finalized or state.tick >= 7200:
                break
            t3 = time.perf_counter()
            transition = engine.step(plays, 5, delay=delay)
            timing['dropped'] += transition.executed.count(False)  # plays the engine refused
            state = transition.state
            timing['engine'] += time.perf_counter() - t3
            for o in (0, 1):  # crowns taken minus crowns lost since the last decision
                gained = (state.crowns[o] - crowns[o]) - (state.crowns[1 - o] - crowns[1 - o])
                steps[o]['reward'][-1] += CROWN_REWARD * gained
            crowns = state.crowns
        timing['expired'] = engine.expired - expired
        for o in (0, 1):
            outcome = 0.0 if state.result not in (0, 1) else (1.0 if state.result == o else -1.0)
            steps[o]['reward'][-1] += outcome
            if o in spec.get('train', [s for s in (0, 1) if models[s] == 'current']):
                trajectory = {k: np.array(v) for k, v in steps[o].items()}
                trajectory = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in trajectory.items()}
                learner.send(('trajectory', index, dict(trajectory, outcome=outcome, opponent=models[1 - o], timing=timing)))
                timing = {}  # counted once per game
        conn.send(('reset', index, [(index, 0), (index, 1)]))


# ---- learner ------------------------------------------------------------------------------------------------------------
def load_policy(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = Policy(**saved['config'])
    load_weights(model, saved['model'])
    return model.to(device), saved


GRAPH_BUCKETS = (8, 16, 32, 64, 128)  # batch sizes a recorded decision is padded to

# A feature row travels as one byte row (8-byte aligned: int64 fields, then float32, then bool), so a server copies
# a model's whole batch to the GPU at once and views the fields back out of it.
LAYOUT = [(k, np.int64, (D.MAX_ENTITIES,)) for k in ('ent_obj', 'ent_side', 'ent_type', 'ent_cell')] + [
    ('rev_card', np.int64, (D.REVEALED,)), ('hand_card', np.int64, (4,)), ('next_card', np.int64, ()),
    ('cycle_card', np.int64, (3,)), ('pending_card', np.int64, ()), ('opp_deck', np.int64, (8,)),
    ('ent_float', np.float32, (D.MAX_ENTITIES, D.ENTITY_FLOATS)), ('rev_age', np.float32, (D.REVEALED,)),
    ('scalars', np.float32, (D.SCALARS,)), ('hand_cost', np.float32, (4,)), ('extras', np.float32, (D.EXTRAS,)),
    ('slot_ok', np.bool_, (4,)), ('ability_ok', np.bool_, ())]
assert sorted(k for k, _, _ in LAYOUT) == sorted(INPUT_KEYS)


def _fields():
    fields, start = [], 0
    for key, dtype, shape in LAYOUT:
        end = start + int(np.prod(shape)) * np.dtype(dtype).itemsize
        fields.append((key, dtype, shape, start, end))
        start = end
    return fields, -(-start // 8) * 8


FIELDS, ROW_BYTES = _fields()  # (key, dtype, shape, start byte, end byte); bytes per row
TORCH_DTYPES = {np.int64: torch.int64, np.float32: torch.float32, np.bool_: torch.bool}


def pack(row):
    """A feature row (dict of arrays, il.data.features) -> one uint8 array of ROW_BYTES."""
    out = np.zeros(ROW_BYTES, np.uint8)
    for key, dtype, shape, start, end in FIELDS:
        out[start:end] = np.ascontiguousarray(row[key], dtype).reshape(-1).view(np.uint8)
    return out


def unpack(rows):
    """[n, ROW_BYTES] uint8 tensor -> model inputs ([n, 1, ...] views of it, no copy)."""
    return {key: rows[:, start:end].view(TORCH_DTYPES[dtype]).reshape(len(rows), 1, *shape)
            for key, dtype, shape, start, end in FIELDS}


class Server:
    """Batched decisions for every collector. Each seat's LSTM state lives in one row of a preallocated GPU tensor
    (the last row is a scratch row for padding). One decision of one model - gather the seats' states, forward,
    write the states back, sample, log-prob - is one function; on a GPU it is recorded as a CUDA graph per batch
    bucket (checked against an eager pass when recorded; a failed check falls back to eager calls), so a model's
    share of a round is one input copy, one graph launch and one copy back."""

    def __init__(self, models, device, core, capacity=256):
        self.models, self.device, self.core = models, device, core
        self.rows = {}  # seat -> row in self.h / self.c
        self._allocate(capacity)

    def _allocate(self, capacity):
        h = torch.zeros(1, capacity + 1, self.core, device=self.device)
        c = torch.zeros_like(h)
        if hasattr(self, 'h'):
            h[:, :self.capacity], c[:, :self.capacity] = self.h[:, :self.capacity], self.c[:, :self.capacity]
        self.h, self.c, self.capacity = h, c, capacity
        self.graphs = {} if self.device.type == 'cuda' else None  # (model key, bucket) -> recorded decision

    def _decide(self, model, rows, index):
        """Decisions for `rows` (uint8 [n, ROW_BYTES]) of seats in state rows `index`: [n, 3] (action, logp, value)."""
        h, c = self.h.index_select(1, index), self.c.index_select(1, index)
        # bf16 on the GPU, as in the PPO update, so old and new log-probs come from the same arithmetic
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda', cache_enabled=False):
            logits, value, (h, c) = model(unpack(rows), (h, c))
        self.h.index_copy_(1, index, h.float())
        self.c.index_copy_(1, index, c.float())
        dist = distribution(logits[:, 0], explore=ABILITY_EXPLORE)
        action = sample(dist)
        return torch.stack([action.float(), log_prob(dist, action), value[:, 0].float()], 1), logits, value

    def _record(self, model, rows, index):
        """`rows` / `index` padded to a bucket -> a recorded decision for that bucket, or None (logged)."""
        saved = self.h.clone(), self.c.clone()

        def restore():
            self.h.copy_(saved[0])
            self.c.copy_(saved[1])
        try:
            g = {'rows': rows.clone(), 'index': index.clone(), 'graph': torch.cuda.CUDAGraph()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._decide(model, g['rows'], g['index'])
            torch.cuda.current_stream().wait_stream(side)
            restore()
            with torch.cuda.graph(g['graph']):
                g['out'], logits, value = self._decide(model, g['rows'], g['index'])
            g['graph'].replay()
            got = logits.float().clone(), value.float().clone()
            restore()
            want = [t.float() for t in self._decide(model, rows, index)[1:]]
            restore()
            finite = torch.isfinite(want[0])
            same = (torch.equal(finite, torch.isfinite(got[0]))
                    and torch.allclose(got[0][finite], want[0][finite], atol=0.05, rtol=0.05)
                    and torch.allclose(got[1], want[1], atol=0.05, rtol=0.05))
            print(f'CUDA graph for batch {len(index)}: {"ok" if same else "MISMATCH, using eager calls"}', flush=True)
            return g if same else None
        except Exception as error:  # noqa: BLE001 - any recording failure: keep the eager path
            restore()
            print(f'CUDA graph for batch {len(index)} failed ({error!r}); using eager calls', flush=True)
            return None

    def _rows(self, seats):
        for seat in seats:
            if seat not in self.rows:
                self.rows[seat] = len(self.rows)
        if len(self.rows) > self.capacity:  # more seats than rows: grow (recorded graphs point at the old rows)
            self._allocate(2 * len(self.rows))
        return np.array([self.rows[seat] for seat in seats], np.int64)

    @torch.no_grad()
    def act(self, requests):
        """requests: [(seat, model key, packed feature row)] -> [(action, logp, value)] in the same order."""
        by_model = defaultdict(list)
        for i, (seat, key, row) in enumerate(requests):
            by_model[key].append(i)
        pending = []
        for key, idx in by_model.items():  # every model's share launched before the first copy back waits
            n, index = len(idx), self._rows([requests[i][0] for i in idx])
            bucket = next((b for b in GRAPH_BUCKETS if b >= n), None) if self.graphs is not None else None
            rows = np.stack([requests[i][2] for i in idx])
            if bucket:
                rows = np.concatenate([rows, np.zeros((bucket - n, ROW_BYTES), np.uint8)])
                index = np.concatenate([index, np.full(bucket - n, self.capacity)])  # padding uses the scratch row
            rows = torch.from_numpy(rows).to(self.device, non_blocking=True)
            index = torch.from_numpy(index).to(self.device, non_blocking=True)
            if bucket and (key, bucket) not in self.graphs:
                self.graphs[(key, bucket)] = self._record(self.models[key], rows, index)
            g = self.graphs.get((key, bucket)) if bucket else None
            if g is None:
                out = self._decide(self.models[key], rows, index)[0]
            else:
                g['rows'].copy_(rows)
                g['index'].copy_(index)
                g['graph'].replay()
                out = g['out']
            pending.append((idx, out[:n]))
        results = [None] * len(requests)
        for idx, out in pending:
            for i, (a, logp, v) in zip(idx, out.cpu().tolist()):
                results[i] = (int(a), logp, v)
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
    log = os.open(f'{args.out}/logs/server-{sid}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    torch.set_num_threads(1)
    device = pick_device(args.server_device or args.device)
    if device.type == 'cuda':
        if device.index is None:  # 'cuda' (--device auto): torch.cuda.set_device needs an index
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)  # CUDA graphs and streams use the current device
    model, _ = load_policy(args.init, device)  # the architecture; weights come from `shared`

    def refresh():
        with lock:
            model.load_state_dict(shared)
            seen = version.value
        print(f'weights version {seen}: first parameter sum {next(model.parameters()).sum().item():.6f}', flush=True)
        return seen

    seen = refresh()
    pool = {'il': args.init, **{q.stem: str(q) for q in sorted(opponent_dir(args).glob('*.pt'))}}
    server = Server({'current': model.eval(), **{k: load_policy(v, device)[0].eval() for k, v in pool.items()}},
                    device, model.config['core'])
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
            server.models[path.stem] = load_policy(path, device)[0].eval()
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
            replies = server.act(requests)
            t2 = time.perf_counter()
            k = 0
            for conn, payload in owners:
                conn.send(replies[k:k + len(payload)])
                k += len(payload)
            for j, value in enumerate((0, t2 - t1, time.perf_counter() - t2, 1, len(requests))):
                stats[sid * len(STATS) + j] += value
            owners, first = [], None


def advantages(trajectory, gamma, lam):
    reward, value = trajectory['reward'], trajectory['value']
    adv = np.zeros_like(value)
    last = 0.0
    for t in reversed(range(len(reward))):
        next_value = value[t + 1] if t + 1 < len(value) else 0.0
        last = reward[t] + gamma * next_value - value[t] + gamma * lam * last
        adv[t] = last
    return adv.astype(np.float32), (adv + value).astype(np.float32)


def discounted(reward, gamma):
    """Discounted sum of the rewards still to come, at every step (what actually happened)."""
    out, running = np.zeros_like(reward), 0.0
    for t in reversed(range(len(reward))):
        running = reward[t] + gamma * running
        out[t] = running
    return out


def explained(target, value):
    """1 - Var(target - value) / Var(target): 1 = the value head predicts the target, 0 = no better than its mean."""
    return float(1 - np.var(target - value) / max(np.var(target), 1e-8))


def update(model, reference, optimizer, trajectories, args, device):
    for t in trajectories:
        t['adv'], t['ret'] = advantages(t, args.gamma, args.lam)
    flat = np.concatenate([t['adv'] for t in trajectories])
    mean, std = flat.mean(), flat.std() + 1e-8
    value = np.concatenate([t['value'] for t in trajectories])
    diagnostics = {  # how much the value head (the advantages' baseline) knows, before this update
        'value_ev_mc': explained(np.concatenate([discounted(t['reward'], args.gamma) for t in trajectories]), value),
        'value_ev_gae': explained(np.concatenate([t['ret'] for t in trajectories]), value)}
    keys = INPUT_KEYS + ('action', 'logp', 'adv', 'ret')
    stats = defaultdict(list)
    # The reference (IL) policy's distribution, once per trajectory: it does not change during the update, so the
    # epochs need not run its forward pass again (it was a quarter of the update's GPU time).
    with torch.no_grad():
        for i in range(0, len(trajectories), args.minibatch):
            chunk = trajectories[i:i + args.minibatch]
            batch = collate([{k: t[k] for k in INPUT_KEYS + ('action',)} for t in chunk])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
                ref_logits, _, _ = reference({k: batch[k] for k in INPUT_KEYS}, valid=batch['time_mask'])
            ref = distribution(ref_logits, explore=ABILITY_EXPLORE)
            for b, t in enumerate(chunk):
                t['ref'] = tuple(r[b, :len(t['action'])] for r in ref)
    for _ in range(args.epochs):
        random.shuffle(trajectories)
        for i in range(0, len(trajectories), args.minibatch):
            batch = collate([{k: t[k] for k in keys} for t in trajectories[i:i + args.minibatch]])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            m = batch['time_mask']
            x = {k: batch[k] for k in INPUT_KEYS}
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
                logits, value, _ = model(x, valid=m)
            dist = distribution(logits, explore=ABILITY_EXPLORE)
            ref = padded_ref(trajectories[i:i + args.minibatch], m.shape[1])
            new_logp = log_prob(dist, batch['action'])
            ratio = (new_logp - batch['logp']).exp()
            adv = (batch['adv'] - mean) / std
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)[m].mean()
            value_loss = F.mse_loss(value.float()[m], batch['ret'][m])
            ent = entropy(dist)[m].mean()
            kl_ref = kl(dist, ref)[m].mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy * ent + args.kl_coef * kl_ref
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(ent.item())
                stats['kl_ref'].append(kl_ref.item())
                stats['approx_kl'].append(((ratio - 1) - ratio.log())[m].mean().item())
                stats['clip_frac'].append(((ratio - 1).abs() > args.clip)[m].float().mean().item())
    for t in trajectories:
        del t['ref']
    return {**{k: float(np.mean(v)) for k, v in stats.items()}, **diagnostics}


def padded_ref(chunk, T):
    """The minibatch's reference distribution [B, T, ...] from its trajectories' stored ones: padded steps as a padded
    step's own (no playable card: wait with probability 1)."""
    first = chunk[0]['ref'][0]
    padded = torch.full((1, D.ACTIONS), float('-inf'), device=first.device)
    padded[:, 0] = 0  # a padded step's logits: wait finite; the ability (ability_ok False) and every card cell masked
    fills = [f[0] for f in distribution(padded)]
    out = []
    for j, fill in enumerate(fills):
        parts = [t['ref'][j] for t in chunk]
        full = fill.expand(len(parts), T, *fill.shape).clone()
        for b, part in enumerate(parts):
            full[b, :len(part)] = part
        out.append(full)
    return tuple(out)


def opponent_dir(args):
    """Where the frozen opponents come from: this run's own snapshots, or --opponent-pool."""
    return Path(args.opponent_pool).expanduser() if args.opponent_pool else Path(args.out) / 'pool'


def save(path, model, saved, optimizer=None, extra=None):
    """Same layout as IL checkpoints (il.model.load reads it); only latest.pt keeps the optimizer."""
    base = {k: v for k, v in saved.items() if k not in ('optimizer', 'update')}
    tmp = Path(f'{path}.tmp')  # written whole, then renamed: a kill mid-write (scancel, node failure) keeps the old file
    torch.save(dict(base, model=model.state_dict(), **({'optimizer': optimizer.state_dict()} if optimizer else {}),
                    **(extra or {})), tmp)
    os.replace(tmp, path)


def endpoint(run):
    """Remote collectors: The learner's current address and key, waiting until a learner has written them."""
    path = run / 'endpoint.json'
    while True:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(10)


def remote(slot, run):
    """One remote collector, for good."""
    from native_engine.clapha import raise_thread_limit
    raise_thread_limit()
    first = endpoint(run)
    name = f'{socket.gethostname()}-{os.getppid()}-{slot}'
    engine, vocab = collector_engine(name, first['out'], first['clapha'], slot // 4 * 8)
    while True:
        learner, servers = None, []
        try:
            ep = endpoint(run)
            key = bytes.fromhex(ep['authkey'])
            learner = Client(tuple(ep['learner']), authkey=key)
            learner.send(('hello', None, None))
            hello = receive(learner)
            servers = [Client(tuple(address), authkey=key) for address in ep['servers']]
            print(f'connected to {ep["host"]} as collector {hello["index"]}', flush=True)
            play_games(hello['index'], ep['delay'], engine, vocab, learner, servers)
            print('the learner stopped; waiting for the next one', flush=True)
        except (OSError, EOFError) as error:  # learner gone, restarting, or not up yet
            print(f'not connected ({error!r}); retrying', flush=True)
        finally:
            for c in [learner, *servers]:
                if c is not None:
                    c.close()
        time.sleep(15)


def join():
    """Collectors in another job: they find the learner through <run>/endpoint.json (written by il.rl --remote on a
    shared file system), connect over the cluster network and play its games. When the learner stops or restarts
    they wait and reconnect, re-reading the file; the learner simply loses them when this job ends."""
    parser = argparse.ArgumentParser(prog='python -m il.rl join')
    parser.add_argument('run', help='the il.rl --remote output directory (on the shared file system)')
    parser.add_argument('--collectors', type=int, default=30)
    args = parser.parse_args(sys.argv[2:])
    run = Path(args.run).expanduser().resolve()
    ctx = mp.get_context('fork')
    workers = [ctx.Process(target=remote, args=(i, run), daemon=True) for i in range(args.collectors)]
    for w in workers:
        w.start()
    print(f'{args.collectors} collectors for {run}; logs in {run}/logs/collector-{socket.gethostname()}-*', flush=True)
    while True:  # keep the job alive; restart any collector that dies outright
        for i, w in enumerate(workers):
            if w.exitcode is not None:
                print(f'collector {i} exited ({w.exitcode}); restarting it', flush=True)
                workers[i] = ctx.Process(target=remote, args=(i, run), daemon=True)
                workers[i].start()
        time.sleep(30)


def main():
    if sys.argv[1:2] == ['join']:
        return join()
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('init', help='IL checkpoint to start from (also the KL reference and first frozen opponent)')
    parser.add_argument('out')
    parser.add_argument('--collectors', type=int, default=24)
    parser.add_argument('--servers', type=int, default=3, help='decision server processes (each one Python thread)')
    parser.add_argument('--batch-ms', type=float, default=2.0, help='how long a server gathers decisions per GPU call')
    parser.add_argument('--games-per-update', type=int, default=128)
    parser.add_argument('--max-backlog', type=int, help='finished trajectories waiting for the learner before collectors '
                        "are held back (default: --games-per-update; 0: no limit)")
    parser.add_argument('--updates', type=int, default=100000)
    parser.add_argument('--self-play', type=float, default=0.5)
    parser.add_argument('--snapshot-every', type=int, default=150, help='updates between frozen snapshots')
    parser.add_argument('--delay', type=int, default=D.DEPLOY_DELAY)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--minibatch', type=int, default=8, help='games per gradient step')
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy', type=float, default=0.002)
    parser.add_argument('--kl-coef', type=float, default=0.05)
    parser.add_argument('--pool', default='data/deck_pool.json', help='deck pairs as played (build_cache --deck-pool)')
    parser.add_argument('--cache', default='data/cache', help='deck pairs when there is no pool file')
    parser.add_argument('--device', default='auto', help="'auto': cuda if available, decided in each process after forking")
    parser.add_argument('--learner-deck', help="specialist: the learning seat always plays this deck (8 cards, id or "
                        "id:el, as il.arena --deck), against real ladder decks; only that seat's games train")
    parser.add_argument('--opponent-pool', help="frozen opponents from this directory of snapshots instead of the run's own "
                        "(specialist: the general run's pool, models that play every ladder deck)")
    parser.add_argument('--pool-size', type=int, default=6, help='frozen opponents: the starting model + this many latest '
                        'snapshots')
    parser.add_argument('--server-device', help="the decision servers' device (default: --device), e.g. cuda:1 "
                        "with two GPUs: the learner's update and the servers stop sharing one")
    parser.add_argument('--remote', action='store_true',
                        help='accept collectors from other jobs (python -m il.rl join <out>); address in <out>/endpoint.json')
    args = parser.parse_args()
    out = Path(args.out)
    from native_engine.clapha import raise_thread_limit
    raise_thread_limit()  # before any engine process starts, so every process (the learner too) gets it
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    (out / 'pool').mkdir(exist_ok=True)
    resume = out / 'latest.pt'
    start_from = resume if resume.exists() else args.init
    cpu_model, saved = load_policy(start_from, 'cpu')
    (out / 'vocab.json').write_text(json.dumps({str(k): v for k, v in saved['vocab'].items()}))
    from .arena import deck_pairs as sample_decks, mirror
    learner_deck = mirror(args.learner_deck)['decks'][0] if args.learner_deck else None
    deck_pairs = sample_decks(args.pool, args.cache, 30000, seed=random.randrange(1 << 20))

    # Every child is forked before CUDA starts here; the servers start CUDA themselves.
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA started before forking the decision servers; they could not use it')
    ctx = mp.get_context('fork')
    if args.remote:  # listeners before forking: each server inherits its own
        authkey = os.urandom(16)
        learner_listener = Listener(('0.0.0.0', 0), authkey=authkey)
        server_listeners = [Listener(('0.0.0.0', 0), authkey=authkey) for _ in range(args.servers)]
    else:
        learner_listener, server_listeners = None, [None] * args.servers
    shared = {k: v.clone().share_memory_() for k, v in cpu_model.state_dict().items()}
    version, lock = ctx.Value('i', 0), ctx.Lock()
    stats = ctx.Array('d', args.servers * len(STATS))
    snapshot_queues = [ctx.Queue() for _ in range(args.servers)]
    to_learner = [ctx.Pipe() for _ in range(args.collectors)]  # (learner end, collector end)
    to_server = [[ctx.Pipe() for _ in range(args.servers)] for _ in range(args.collectors)]  # [collector][server]
    every = [c for pair in to_learner + [p for row in to_server for p in row] for c in pair]

    def start(target, keep, *target_args):
        def run():
            for c in every:  # a pipe end left open in another process never reads as closed
                if not any(c is k for k in keep):
                    c.close()
            target(*target_args)
        ctx.Process(target=run, daemon=True).start()

    for sid in range(args.servers):
        mine = [to_server[i][sid][0] for i in range(args.collectors)]
        start(decision_server, mine, sid, args, mine, shared, version, lock, snapshot_queues[sid], stats,
              server_listeners[sid])
    for i in range(args.collectors):
        ends = [pipe[1] for pipe in to_server[i]]
        start(collector, [to_learner[i][1], *ends], i, args, to_learner[i][1], ends)
    conns = [learner_end for learner_end, _ in to_learner]
    arrivals, next_remote = queue.Queue(), 100000  # remote collectors' numbers start here
    self_play_games = 0  # spreads self-play games over the servers
    if args.remote:
        accept_into(learner_listener, arrivals)
        ip = reachable_ip()
        endpoint = {'learner': [ip, learner_listener.address[1]],
                    'servers': [[ip, listener.address[1]] for listener in server_listeners],
                    'authkey': authkey.hex(), 'host': socket.gethostname(), 'clapha': args.clapha,
                    'out': str(out.resolve()), 'delay': args.delay, 'started': time.time()}
        path = out / 'endpoint.json'
        path.write_text(json.dumps(endpoint))
        path.chmod(0o600)  # the key admits collectors
        print(f'remote collectors: python -m il.rl join {out.resolve()} (learner at {ip}:{endpoint["learner"][1]})', flush=True)
    for c in every:
        if not any(c is k for k in conns):
            c.close()

    torch.set_num_threads(2)
    device = pick_device(args.device)
    model = cpu_model.to(device)
    reference, _ = load_policy(args.init, device)
    reference.eval().requires_grad_(False)
    if device.type == 'cuda':  # convs in channels-last layout: no layout conversion before each conv (~7% of an update)
        model.to(memory_format=torch.channels_last)
        reference.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if 'optimizer' in saved:
        try:
            optimizer.load_state_dict(saved['optimizer'])
        except ValueError:  # saved before the model gained layers: a fresh Adam
            print('the saved optimizer state does not fit the model; starting Adam afresh', flush=True)
        for group in optimizer.param_groups:  # the saved state would restore the old learning rate
            group['lr'] = args.lr
    start_update = saved.get('update', 0)
    frozen = ['il'] + [q.stem for q in sorted(opponent_dir(args).glob('*.pt'))]
    results = defaultdict(lambda: deque(maxlen=200))  # opponent -> recent learner outcomes

    def pick_opponent():
        # the starting model and the --pool-size latest snapshots: older ones are beaten easily and each extra model
        # in a server's round costs a graph launch and a copy back (14 frozen models cut throughput by a quarter)
        active = frozen[:1] + frozen[1:][-args.pool_size:]
        weights = [(1 - (np.mean(results[k]) + 1) / 2 if results[k] else 0.5) ** 2 + 0.05 for k in active]
        return random.choices(active, weights)[0]

    def publish():
        with lock:
            for k, v in model.state_dict().items():
                shared[k].copy_(v.detach().cpu())
            version.value += 1
        print(f'published weights version {version.value}: first parameter sum '
              f'{next(model.parameters()).sum().item():.6f}', flush=True)

    ready, failed = queue.Queue(), []  # finished trajectories; why collecting stopped
    played = 0
    held, held_time = [], [0.0]  # collectors waiting for their next game (conn, since); seconds held this cycle
    max_backlog = args.games_per_update if args.max_backlog is None else args.max_backlog

    def new_game(conn):
        """The next game's spec for a collector. A frozen opponent's games all go to its home server (pool position %
        servers): a server then runs the current model and its own opponents, not every model of the pool, each round."""
        nonlocal self_play_games
        learner_seat = random.randrange(2)
        models = ['current', 'current']
        if random.random() >= args.self_play:
            models[1 - learner_seat] = pick_opponent()
            server = frozen.index(models[1 - learner_seat]) % args.servers
        else:
            server = self_play_games % args.servers
            self_play_games += 1
        decks, train = random.choice(deck_pairs), None
        if learner_deck is not None:  # specialist: our deck vs a real ladder deck; only our seat learns
            decks = dict(decks, decks=[None, None])
            decks['decks'][learner_seat] = learner_deck
            decks['decks'][1 - learner_seat] = random.choice(random.choice(deck_pairs)['decks'])
            train = [learner_seat]
        conn.send(dict(decks=decks, seed=random.randrange(1 << 30), models=models, server=server,
                       **({'train': train} if train else {})))

    def serve_collectors():
        """A thread that keeps the collectors busy (new games, trajectories) while the main thread trains."""
        nonlocal next_remote, played
        try:
            while True:
                while not arrivals.empty():
                    conns.append(arrivals.get())
                if not conns:
                    if not args.remote:
                        raise RuntimeError('every collector died')
                    time.sleep(0.5)
                    continue
                # With a full update's worth of trajectories already waiting (--max-backlog), collectors' next games
                # are held back until the learner takes them (at most HOLD_LIMIT s): otherwise, when collectors
                # outrun the update, the queue grows without end (memory, and data from ever older weights).
                now = time.time()
                while held and (not max_backlog or ready.qsize() < max_backlog or now - held[0][1] > HOLD_LIMIT):
                    conn, since = held.pop(0)
                    held_time[0] += now - since
                    if conn in conns:
                        try:
                            new_game(conn)
                        except (EOFError, OSError):
                            pass  # gone meanwhile; the read below notices
                for conn in wait(conns, timeout=0.2 if held else 1):
                    try:
                        kind, index, payload = conn.recv()
                    except (EOFError, OSError):  # a collector died, or a remote one's job ended; its log says why
                        conns.remove(conn)
                        held[:] = [h for h in held if h[0] is not conn]
                        print(f'a collector left ({len(conns)} connected); see {out}/logs/', flush=True)
                        continue
                    if kind == 'hello':  # a remote collector: its number
                        conn.send({'index': next_remote})
                        next_remote += 1
                    elif kind == 'new_game':
                        if max_backlog and ready.qsize() >= max_backlog:
                            held.append((conn, time.time()))
                        else:
                            new_game(conn)
                    elif kind == 'trajectory':
                        if payload['opponent'] != 'current':
                            results[payload['opponent']].append(payload['outcome'])
                        played += 1
                        ready.put(payload)
        except BaseException as error:  # noqa: BLE001 - reported by the main thread
            failed.append(error)

    threading.Thread(target=serve_collectors, daemon=True).start()
    started = last = update_end = time.time()
    played_before = 0
    metrics = open(out / 'metrics.jsonl', 'a')
    for update_index in range(start_update, args.updates):
        # Games keep being played (with the previous weights) while an update runs; PPO's ratio uses the log-probs
        # they were played with.
        trajectories = []
        while len(trajectories) < args.games_per_update:
            if failed:
                raise failed[0]
            try:
                trajectories.append(ready.get(timeout=1))
            except queue.Empty:
                pass
        waited = time.time() - update_end  # the learner's wait for this batch after the last update
        backlog = ready.qsize()  # trajectories still waiting once this batch is taken
        collect_time, last = time.time() - last, time.time()  # wall clock per update: collection overlaps updates
        update_start = time.time()
        model.train()
        stats_row = update(model, reference, optimizer, trajectories, args, device)
        model.eval()
        publish()
        step = update_index + 1
        save(out / 'latest.pt', model, saved, optimizer, {'update': step})
        update_end = time.time()
        if step % args.snapshot_every == 0:
            path = out / 'pool' / f'update-{step:05d}.pt'
            save(path, model, saved)
            if opponent_dir(args) == out / 'pool':  # with --opponent-pool, our own snapshots are not opponents
                frozen.append(path.stem)
            for q in snapshot_queues if opponent_dir(args) == out / 'pool' else ():
                q.put(str(path))
        served = {name: sum(stats[s * len(STATS) + j] for s in range(args.servers)) for j, name in enumerate(STATS)}
        for j in range(len(stats)):
            stats[j] = 0.0
        timed = [t['timing'] for t in trajectories if t['timing']]
        decisions = sum(t['decisions'] for t in timed) or 1
        outcomes = [t['outcome'] for t in trajectories if t['opponent'] != 'current']
        collected, played_before = played - played_before, played
        held_s, held_time[0] = held_time[0], 0.0
        row = dict(update=step, collectors=len(conns), trajectories=played, games_per_hour=round(3600 * len(timed) / max(collect_time, 1e-9)),
                   # games_per_hour: trajectories trained on per hour; collected_per_hour: finished by collectors
                   # (higher when collectors outrun the update; then backlog / held_s show the learner is the limit)
                   collected_per_hour=round(3600 * collected / max(collect_time, 1e-9)), backlog=backlog, held_s=round(held_s, 1),
                   server_ms_per_call=round(1000 * served['decide'] / max(served['batches'], 1), 2),
                   collector_ms={k: round(1000 * sum(t[k] for t in timed) / decisions, 2) for k in ('features', 'wait', 'engine')},
                   dropped_per_game=round(sum(t.get('dropped', 0) for t in timed) / max(len(timed), 1), 2),
                   expired_per_game=round(sum(t.get('expired', 0) for t in timed) / max(len(timed), 1), 2),
                   server_s={k: round(served[k] / args.servers, 1) for k in ('receive', 'decide', 'reply')},
                   mean_batch=round(served['requests'] / max(served['batches'], 1), 1),
                   hours=(time.time() - started) / 3600, cycle_s=collect_time, wait_s=waited, update_s=time.time() - update_start,
                   steps=int(sum(len(t['action']) for t in trajectories)),
                   plays_per_game=float(np.mean([(t['action'] >= 2).sum() for t in trajectories])),
                   abilities_per_game=float(np.mean([(t['action'] == D.ABILITY).sum() for t in trajectories])),
                   full_elixir=float(np.mean(np.concatenate([t['full'] for t in trajectories]))),
                   score_vs_pool=float(np.mean(outcomes)) if outcomes else None,
                   pool={k: round(float(np.mean(v)), 3) for k, v in list(results.items()) if v}, **stats_row)
        metrics.write(json.dumps(row) + '\n')
        metrics.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    for conn in conns:
        conn.send(None)


if __name__ == '__main__':
    main()
