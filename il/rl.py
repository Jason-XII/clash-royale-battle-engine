"""PPO self-play on clapha engines (x86_64 Linux), starting from an IL checkpoint.

    PYTHONPATH=. python -m il.rl ~/clapha-engine ~/cr-engine/runs/il-002/best.pt ~/cr-engine/runs/ppo-general --collectors 24 --servers 4

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
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .model import INPUT_KEYS, Policy
from .train import collate

TEMPERATURE = 0.5
CROWN_REWARD = 0.1
OVERFLOW_PENALTY = 0.002


# ---- the policy distribution -----------------------------------------------------------------------------------------
def distribution(logits, temperature=TEMPERATURE):
    """[..., ACTIONS] logits -> (log p(wait), log p(play), conditional log-probs over cards x cells [..., 4*CELLS]).
    The ability action is excluded. A step with no playable card waits with probability 1."""
    logits = logits.float()
    cards = logits[..., 2:]
    joint = torch.cat([logits[..., :1], cards], -1)  # wait + cards, ability dropped
    log_wait = joint.log_softmax(-1)[..., 0]
    playable = torch.isfinite(cards).any(-1)
    log_play = torch.where(playable, torch.log1p(-log_wait.exp().clamp(max=1 - 1e-6)), torch.full_like(log_wait, -1e9))
    log_wait = torch.where(playable, log_wait, torch.zeros_like(log_wait))
    cond = torch.where(playable.unsqueeze(-1), cards / temperature, torch.zeros_like(cards)).log_softmax(-1)
    return log_wait, log_play, cond


def log_prob(dist, action):
    log_wait, log_play, cond = dist
    card = (action - 2).clamp(min=0)
    return torch.where(action == D.WAIT, log_wait, log_play + cond.gather(-1, card.unsqueeze(-1)).squeeze(-1))


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
    card = (cond - torch.empty_like(cond).exponential_().log()).argmax(-1)  # Gumbel-max: a draw from softmax(cond)
    return torch.where(play, card + 2, torch.zeros_like(card))


# ---- collectors ---------------------------------------------------------------------------------------------------------
def collector(index, args, learner, conn):
    """Plays games; asks its decision server (`conn`) for actions and the learner for games and trajectories."""
    log = os.open(f'{args.out}/logs/collector-{index}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    time.sleep(index // 4 * 8)  # boot 4 engines at a time
    torch.set_num_threads(1)
    from native_engine import Play
    from native_engine.clapha import ClaphaEngine
    from .agent import Agent
    from .arena import make_match
    engine = ClaphaEngine(args.clapha)
    vocab = D.load_vocab(Path(args.out) / 'vocab.json')
    while True:
        learner.send(('new_game', index, None))
        spec = learner.recv()
        if spec is None:
            break
        match = make_match(spec['decks'], spec['seed'])
        models = spec['models']  # per seat: 'current' or a frozen opponent's key
        agents = [Agent(None, vocab, owner) for owner in (0, 1)]
        steps = [defaultdict(list) for _ in (0, 1)]
        state = engine.reset(match=match)
        crowns = (0, 0)
        timing = defaultdict(float)  # seconds per part of a decision, for the learner's log
        while True:
            t0 = time.perf_counter()
            rows = [{k: v[0, 0].numpy() for k, v in agents[o].observe(state).items()} for o in (0, 1)]
            t1 = time.perf_counter()
            conn.send(('act', index, [((index, o), models[o], rows[o]) for o in (0, 1)]))
            replies = conn.recv()
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
                if action >= 2:
                    slot, cell = divmod(action - 2, D.CELLS)
                    x, y = D.cell_center(cell, o)
                    plays.append(Play(o, int(slot), int(x), int(y)))
            if state.finalized or state.tick >= 7200:
                break
            t3 = time.perf_counter()
            transition = engine.step(plays, 5, delay=args.delay)
            state = transition.state
            timing['engine'] += time.perf_counter() - t3
            for o in (0, 1):  # crowns taken minus crowns lost since the last decision
                gained = (state.crowns[o] - crowns[o]) - (state.crowns[1 - o] - crowns[1 - o])
                steps[o]['reward'][-1] += CROWN_REWARD * gained
            crowns = state.crowns
        for o in (0, 1):
            outcome = 0.0 if state.result not in (0, 1) else (1.0 if state.result == o else -1.0)
            steps[o]['reward'][-1] += outcome
            if models[o] == 'current':
                trajectory = {k: np.array(v) for k, v in steps[o].items()}
                trajectory = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in trajectory.items()}
                learner.send(('trajectory', index, dict(trajectory, outcome=outcome, opponent=models[1 - o], timing=timing)))
                timing = {}  # counted once per game
        conn.send(('reset', index, [(index, 0), (index, 1)]))
    conn.close()
    os._exit(0)  # the engine segfaults in its exit-time destructors


# ---- learner ------------------------------------------------------------------------------------------------------------
def load_policy(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = Policy(**saved['config'])
    model.load_state_dict(saved['model'])
    return model.to(device), saved


class Server:
    """Batched decisions for every collector. Each seat's LSTM state lives in one row of a preallocated GPU
    tensor (gathered and written back by index); results come back to the CPU in one transfer per call."""

    def __init__(self, models, device, core):
        self.models, self.device, self.core = models, device, core
        self.rows = {}  # seat -> row in self.h / self.c
        self.h = self.c = torch.zeros(1, 0, core, device=device)

    def _rows(self, seats):
        for seat in seats:
            if seat not in self.rows:
                self.rows[seat] = len(self.rows)
        if len(self.rows) > self.h.shape[1]:  # grow, keeping the existing rows
            grow = torch.zeros(1, max(64, len(self.rows)), self.core, device=self.device)
            self.h, self.c = torch.cat([self.h, grow], 1), torch.cat([self.c, grow.clone()], 1)
        return torch.tensor([self.rows[seat] for seat in seats], device=self.device)

    @torch.no_grad()
    def act(self, requests):
        """requests: [(seat, model key, feature row)] -> [(action, logp, value)] in the same order."""
        out = [None] * len(requests)
        by_model = defaultdict(list)
        for i, (seat, key, row) in enumerate(requests):
            by_model[key].append(i)
        for key, idx in by_model.items():
            rows = self._rows([requests[i][0] for i in idx])
            x = {k: torch.from_numpy(np.stack([requests[i][2][k] for i in idx]))[:, None].to(self.device, non_blocking=True)
                 for k in INPUT_KEYS}
            # bf16 on the GPU, as in the PPO update, so old and new log-probs come from the same arithmetic
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda'):
                logits, value, (h, c) = self.models[key](x, (self.h[:, rows], self.c[:, rows]))
            self.h[:, rows], self.c[:, rows] = h.float(), c.float()
            dist = distribution(logits[:, 0])
            action = sample(dist)
            result = torch.stack([action.float(), log_prob(dist, action), value[:, 0].float()], 1).cpu().tolist()
            for j, i in enumerate(idx):
                a, logp, v = result[j]
                out[i] = (int(a), logp, v)
        return out

    def reset(self, seats):
        """A seat's game ended: its next game starts from a zero LSTM state."""
        known = [seat for seat in seats if seat in self.rows]
        if known:
            rows = torch.tensor([self.rows[seat] for seat in known], device=self.device)
            self.h[:, rows] = 0
            self.c[:, rows] = 0


STATS = ('receive', 'decide', 'reply', 'batches', 'requests')  # per decision server, summed into the log


def decision_server(sid, args, conns, shared, version, lock, snapshots, stats):
    """Answers its collectors' decisions in batched GPU calls with the learner's latest weights (`shared`,
    re-read whenever `version` changes) and the frozen opponents (`snapshots` announces new ones)."""
    log = os.open(f'{args.out}/logs/server-{sid}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model, _ = load_policy(args.init, device)  # the architecture; weights come from `shared`

    def refresh():
        with lock:
            model.load_state_dict(shared)
            seen = version.value
        print(f'weights version {seen}: first parameter sum {next(model.parameters()).sum().item():.6f}', flush=True)
        return seen

    seen = refresh()
    pool = {'il': args.init, **{q.stem: str(q) for q in sorted((Path(args.out) / 'pool').glob('*.pt'))}}
    server = Server({'current': model.eval(), **{k: load_policy(v, device)[0].eval() for k, v in pool.items()}},
                    device, model.config['core'])
    owners, first, window = [], None, args.batch_ms / 1000  # [(collector pipe, its requests)]
    while conns:
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
            except EOFError:  # its collector died or finished
                conns.remove(conn)
                owners = [(c, r) for c, r in owners if c is not conn]
                continue
            if kind == 'act':
                owners.append((conn, payload))
                first = first or time.perf_counter()
            elif kind == 'reset':
                server.reset(payload)
        t1 = time.perf_counter()
        stats[sid * len(STATS)] += t1 - t0
        if owners and (t1 - first >= window or len(owners) >= len(conns)):
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
    os._exit(0)


def advantages(trajectory, gamma, lam):
    reward, value = trajectory['reward'], trajectory['value']
    adv = np.zeros_like(value)
    last = 0.0
    for t in reversed(range(len(reward))):
        next_value = value[t + 1] if t + 1 < len(value) else 0.0
        last = reward[t] + gamma * next_value - value[t] + gamma * lam * last
        adv[t] = last
    return adv.astype(np.float32), (adv + value).astype(np.float32)


def update(model, reference, optimizer, trajectories, args, device):
    for t in trajectories:
        t['adv'], t['ret'] = advantages(t, args.gamma, args.lam)
    flat = np.concatenate([t['adv'] for t in trajectories])
    mean, std = flat.mean(), flat.std() + 1e-8
    keys = INPUT_KEYS + ('action', 'logp', 'adv', 'ret')
    stats = defaultdict(list)
    for _ in range(args.epochs):
        random.shuffle(trajectories)
        for i in range(0, len(trajectories), args.minibatch):
            batch = collate([{k: t[k] for k in keys} for t in trajectories[i:i + args.minibatch]])
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            m = batch['time_mask']
            x = {k: batch[k] for k in INPUT_KEYS}
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
                logits, value, _ = model(x)
                with torch.no_grad():
                    ref_logits, _, _ = reference(x)
            dist, ref = distribution(logits), distribution(ref_logits)
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
    return {k: float(np.mean(v)) for k, v in stats.items()}


def save(path, model, saved, optimizer=None, extra=None):
    """Same layout as IL checkpoints (il.agent.load / il.arena read it); only latest.pt keeps the optimizer."""
    base = {k: v for k, v in saved.items() if k not in ('optimizer', 'update')}
    torch.save(dict(base, model=model.state_dict(), **({'optimizer': optimizer.state_dict()} if optimizer else {}),
                    **(extra or {})), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('init', help='IL checkpoint to start from (also the KL reference and first frozen opponent)')
    parser.add_argument('out')
    parser.add_argument('--collectors', type=int, default=24)
    parser.add_argument('--servers', type=int, default=4, help='decision server processes (each one Python thread)')
    parser.add_argument('--batch-ms', type=float, default=2.0, help='how long a server gathers decisions per GPU call')
    parser.add_argument('--games-per-update', type=int, default=128)
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
    parser.add_argument('--pool', default='data/deck_pool.json', help='deck pairs as played (tools/build_deck_pool.py)')
    parser.add_argument('--cache', default='data/cache', help='deck pairs when there is no pool file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
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
    from .arena import deck_pairs as sample_decks
    deck_pairs = sample_decks(args.pool, args.cache, 30000, seed=random.randrange(1 << 20))

    # Every child is forked before CUDA starts here; the servers start CUDA themselves.
    ctx = mp.get_context('fork')
    shared = {k: v.clone().share_memory_() for k, v in cpu_model.state_dict().items()}
    version, lock = ctx.Value('i', 0), ctx.Lock()
    stats = ctx.Array('d', args.servers * len(STATS))
    snapshot_queues = [ctx.Queue() for _ in range(args.servers)]
    to_learner = [ctx.Pipe() for _ in range(args.collectors)]  # (learner end, collector end)
    to_server = [ctx.Pipe() for _ in range(args.collectors)]   # (server end, collector end)
    every = [c for pair in to_learner + to_server for c in pair]

    def start(target, keep, *target_args):
        def run():
            for c in every:  # a pipe end left open in another process never reads as closed
                if not any(c is k for k in keep):
                    c.close()
            target(*target_args)
        ctx.Process(target=run, daemon=True).start()

    for sid in range(args.servers):
        mine = [to_server[i][0] for i in range(args.collectors) if i % args.servers == sid]
        start(decision_server, mine, sid, args, mine, shared, version, lock, snapshot_queues[sid], stats)
    for i in range(args.collectors):
        start(collector, [to_learner[i][1], to_server[i][1]], i, args, to_learner[i][1], to_server[i][1])
    conns = [learner_end for learner_end, _ in to_learner]
    for c in every:
        if not any(c is k for k in conns):
            c.close()

    torch.set_num_threads(2)
    device = torch.device(args.device)
    model = cpu_model.to(device)
    reference, _ = load_policy(args.init, device)
    reference.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if 'optimizer' in saved:
        optimizer.load_state_dict(saved['optimizer'])
    start_update = saved.get('update', 0)
    frozen = ['il'] + [q.stem for q in sorted((out / 'pool').glob('*.pt'))]
    results = defaultdict(lambda: deque(maxlen=200))  # opponent -> recent learner outcomes

    def pick_opponent():
        weights = [(1 - (np.mean(results[k]) + 1) / 2 if results[k] else 0.5) ** 2 + 0.05 for k in frozen]
        return random.choices(frozen, weights)[0]

    def publish():
        with lock:
            for k, v in model.state_dict().items():
                shared[k].copy_(v.detach().cpu())
            version.value += 1
        print(f'published weights version {version.value}: first parameter sum '
              f'{next(model.parameters()).sum().item():.6f}', flush=True)

    buffer, played, started = [], 0, time.time()
    metrics = open(out / 'metrics.jsonl', 'a')
    for update_index in range(start_update, args.updates):
        collect_start = time.time()
        while len(buffer) < args.games_per_update:
            for conn in wait(conns, timeout=1):
                try:
                    kind, index, payload = conn.recv()
                except EOFError:  # the collector died; its log says why
                    conns.remove(conn)
                    print(f'a collector died ({len(conns)} left); see {out}/logs/', flush=True)
                    if not conns:
                        raise RuntimeError('every collector died')
                    continue
                if kind == 'new_game':
                    learner_seat = random.randrange(2)
                    models = ['current', 'current']
                    if random.random() >= args.self_play:
                        models[1 - learner_seat] = pick_opponent()
                    conn.send(dict(decks=random.choice(deck_pairs), seed=random.randrange(1 << 30), models=models))
                elif kind == 'trajectory':
                    buffer.append(payload)
                    if payload['opponent'] != 'current':
                        results[payload['opponent']].append(payload['outcome'])
                    played += 1
        collect_time = time.time() - collect_start
        trajectories, buffer = buffer[:args.games_per_update], buffer[args.games_per_update:]
        update_start = time.time()
        model.train()
        stats_row = update(model, reference, optimizer, trajectories, args, device)
        model.eval()
        publish()
        step = update_index + 1
        save(out / 'latest.pt', model, saved, optimizer, {'update': step})
        if step % args.snapshot_every == 0:
            path = out / 'pool' / f'update-{step:05d}.pt'
            save(path, model, saved)
            frozen.append(path.stem)
            for q in snapshot_queues:
                q.put(str(path))
        served = {name: sum(stats[s * len(STATS) + j] for s in range(args.servers)) for j, name in enumerate(STATS)}
        for j in range(len(stats)):
            stats[j] = 0.0
        timed = [t['timing'] for t in trajectories if t['timing']]
        decisions = sum(t['decisions'] for t in timed) or 1
        outcomes = [t['outcome'] for t in trajectories if t['opponent'] != 'current']
        row = dict(update=step, trajectories=played, games_per_hour=round(3600 * len(timed) / max(collect_time, 1e-9)),
                   collector_ms={k: round(1000 * sum(t[k] for t in timed) / decisions, 2) for k in ('features', 'wait', 'engine')},
                   server_s={k: round(served[k] / args.servers, 1) for k in ('receive', 'decide', 'reply')},
                   mean_batch=round(served['requests'] / max(served['batches'], 1), 1),
                   hours=(time.time() - started) / 3600, collect_s=collect_time, update_s=time.time() - update_start,
                   steps=int(sum(len(t['action']) for t in trajectories)),
                   plays_per_game=float(np.mean([(t['action'] >= 2).sum() for t in trajectories])),
                   full_elixir=float(np.mean(np.concatenate([t['full'] for t in trajectories]))),
                   score_vs_pool=float(np.mean(outcomes)) if outcomes else None,
                   pool={k: round(float(np.mean(v)), 3) for k, v in results.items() if v}, **stats_row)
        metrics.write(json.dumps(row) + '\n')
        metrics.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    for conn in conns:
        conn.send(None)


if __name__ == '__main__':
    main()
