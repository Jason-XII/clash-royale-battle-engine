"""PPO on clapha engines (x86_64 Linux), starting from an IL checkpoint: the learner.

    PYTHONPATH=. python -m il.rl ~/clapha-engine ~/cr-engine/runs/il-002/best.pt ~/cr-engine/runs/ppo-general --collectors 24 --servers 4
    PYTHONPATH=. python -m il.rl join ~/cr-engine/runs/ppo-general --collectors 36 --servers 3   # another (GPU) job's
                                       # collectors and decision servers (the learner needs --remote); tools/*.sbatch

The learner forks the decision servers and collectors (il.actors), hands each collector its next game (Matchmaker),
runs PPO on the finished trajectories and publishes new weights to the servers through shared memory. Games are
self-play (--self-play, both seats learn) or against a frozen pool (the IL model and snapshots) chosen by PFSP:
weight (1 - win rate)^2 + 0.05.

Policy (il.model.distribution): wait with the model's p(wait); otherwise a card x cell sampled from the non-wait
logits at temperature TEMPERATURE (near the IL agent's "best card and cell", but stochastic so PPO can learn it).
A KL penalty keeps the policy near the IL model. Reward: see il.actors.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
import multiprocessing as mp
from multiprocessing.connection import Listener, wait
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
from . import defense
from .actors import (STATS, accept_into, collector, decision_server, join, load_policy, opponent_dir, pick_device,
                     reachable_ip, start_mps)
from .model import ABILITY_EXPLORE, INPUT_KEYS, distribution, entropy, kl, load_weights, log_prob
from .train import collate

HOLD_LIMIT = 60  # s a collector's next game may be held back for a full backlog (well under REPLY_TIMEOUT)
FOCUS_DECKS = 20  # --deck-mix 'focus': the most played ladder decks of the pool
HARD_DECAY, HARD_TOP, HARD_GAMES = 0.8, 10, 8  # hard cards: the HARD_TOP worst of those in >= HARD_GAMES decayed games
ADAM_WARMUP = 20  # updates of learning-rate ramp after a fresh Adam: its first steps are full-size on every weight
                  # (a fresh Adam at lr 2e-4 broke the specialist: approx_kl 0.06, 25x usual, and a collapse)
AUX_HORIZONS = (40, 120)  # decisions (5 ticks each) ahead: the auxiliary head's tower-damage targets


def rollback_scale(out):
    """The learning rate's factor from il.autopilot's rollbacks so far (kept across restarts)."""
    return float(np.prod([json.loads(f.read_text())['lr_scale'] for f in Path(out).glob('rollback-done-*.json')]))


def deck_key(deck):
    return tuple(sorted((int(c['d']), int(c.get('el', 0))) for c in deck))  # an evolved card plays differently


class Matchmaker:
    """Who plays whom with which decks: the next game's spec for a collector, steered by the results so far.

    Opponents: half the games (--self-play) both seats are the current model; otherwise a frozen opponent by PFSP over
    the IL model and the --pool-size latest snapshots, or (--il-share, FirstLight's specialist mix) IL that often and
    PFSP over the snapshots otherwise. A frozen opponent's games all go to its home server (pool position % servers):
    a server then runs the current model and its own opponents, not every model of the pool, each round.
    Specialist (learner_deck): our seat plays it against a real ladder deck, by --deck-mix shares of the most played
    ladder decks, any pool deck, and hard decks: a ladder deck with one of the cards our seat scores worst against
    lately (results decay by HARD_DECAY every update; by card, as nearly every ladder deck is played once); only our
    seat's games train."""

    def __init__(self, args, deck_pairs, learner_deck, frozen):
        self.args, self.deck_pairs, self.learner_deck, self.frozen = args, deck_pairs, learner_deck, frozen
        self.results = defaultdict(lambda: deque(maxlen=200))  # opponent -> recent learner outcomes
        self.self_play_games = 0  # spreads self-play games over the servers
        self.deck_mix = [float(x) for x in args.deck_mix.split(',')]
        ladder = [d for pair in deck_pairs for d in pair['decks']] if learner_deck is not None else []
        self.by_key = {deck_key(d): d for d in ladder}
        self.played_keys = [deck_key(d) for d in ladder]
        self.focus = [k for k, _ in Counter(self.played_keys).most_common(FOCUS_DECKS)]
        self.by_card = defaultdict(list)  # card -> the ladder decks with it, as often as they are played
        for key in self.played_keys:
            for card in key:
                self.by_card[card].append(key)
        self.card_results = {}  # card -> [decayed sum of outcomes, decayed games] against decks with it

    def pick_opponent(self):
        # the starting model and the --pool-size latest snapshots: older ones are beaten easily and each extra model
        # in a server's round costs a graph launch and a copy back (14 frozen models cut throughput by a quarter)
        args, frozen = self.args, self.frozen
        if args.il_share is not None:
            if random.random() < args.il_share or len(frozen) == 1:
                return 'il'
            active = frozen[1:][-args.pool_size:]
        else:
            active = frozen[:1] + frozen[1:][-args.pool_size:]
        results = self.results
        weights = [(1 - (np.mean(results[k]) + 1) / 2 if results[k] else 0.5) ** 2 + 0.05 for k in active]
        return random.choices(active, weights)[0]

    def hard_cards(self):
        rated = [(total / games, card) for card, (total, games) in list(self.card_results.items())  # the serve thread adds
                 if games >= HARD_GAMES]
        return [card for _, card in sorted(rated)[:HARD_TOP]]

    def opponent_deck(self):
        kind = random.choices(('focus', 'pool', 'hard'), self.deck_mix)[0]
        hard = self.hard_cards() if kind == 'hard' else []
        if kind == 'focus':
            key = random.choice(self.focus)
        elif hard:
            key = random.choice(self.by_card[random.choice(hard)])
        else:
            key = random.choice(self.played_keys)
        return self.by_key[key], key

    def spec(self):
        args = self.args
        learner_seat = random.randrange(2)
        models = ['current', 'current']
        if random.random() >= args.self_play:
            models[1 - learner_seat] = self.pick_opponent()
            server = self.frozen.index(models[1 - learner_seat]) % args.servers
        else:
            server = self.self_play_games % args.servers
            self.self_play_games += 1
        decks, train, tag = random.choice(self.deck_pairs), None, None
        if self.learner_deck is not None:
            decks = dict(decks, decks=[None, None])
            decks['decks'][learner_seat] = self.learner_deck
            decks['decks'][1 - learner_seat], tag = self.opponent_deck()
            train = [learner_seat]
        return dict(decks=decks, seed=random.randrange(1 << 30), models=models, server=server,
                    shaping=(args.shaping, args.gamma), tag=tag, **({'train': train} if train else {}))

    def record(self, trajectory):
        if trajectory['opponent'] != 'current':
            self.results[trajectory['opponent']].append(trajectory['outcome'])
        if trajectory.get('tag') is not None:
            for card in trajectory['tag']:
                entry = self.card_results.setdefault(card, [0.0, 0.0])
                entry[0] += trajectory['outcome']
                entry[1] += 1

    def decay(self):
        """Once per update: older results count less."""
        for entry in list(self.card_results.values()):
            entry[0] *= HARD_DECAY
            entry[1] *= HARD_DECAY


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


def aux_targets(towers):
    """[T, 2] damage dealt / taken so far (il.actors.damage) -> [T, 4]: how much more of each AUX_HORIZONS decisions
    later (or at the game's end)."""
    T = len(towers)
    return np.concatenate([towers[np.minimum(np.arange(T) + h, T - 1)] - towers for h in AUX_HORIZONS], 1)


def update(model, reference, optimizer, trajectories, args, device):
    for t in trajectories:
        t['adv'], t['ret'] = advantages(t, args.gamma, args.lam)
        t['aux'] = aux_targets(t['towers'])
    flat = np.concatenate([t['adv'] for t in trajectories])
    mean, std = flat.mean(), flat.std() + 1e-8
    value = np.concatenate([t['value'] for t in trajectories])
    diagnostics = {  # how much the value head (the advantages' baseline) knows, before this update
        'value_ev_mc': explained(np.concatenate([discounted(t['reward'], args.gamma) for t in trajectories]), value),
        'value_ev_gae': explained(np.concatenate([t['ret'] for t in trajectories]), value)}
    keys = INPUT_KEYS + ('action', 'logp', 'adv', 'ret', 'aux')
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
                logits, value, _, aux = model(x, valid=m, aux=True)
            dist = distribution(logits, explore=ABILITY_EXPLORE)
            ref = padded_ref(trajectories[i:i + args.minibatch], m.shape[1])
            new_logp = log_prob(dist, batch['action'])
            ratio = (new_logp - batch['logp']).exp()
            adv = (batch['adv'] - mean) / std
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)[m].mean()
            value_loss = F.mse_loss(value.float()[m], batch['ret'][m])
            aux_loss = F.mse_loss(aux.float()[m], batch['aux'][m])
            ent = entropy(dist)[m].mean()
            kl_ref = kl(dist, ref)[m].mean()
            loss = (policy_loss + args.value_coef * value_loss + args.aux_coef * aux_loss - args.entropy * ent
                    + args.kl_coef * kl_ref)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['aux_loss'].append(aux_loss.item())
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


def save(path, model, saved, optimizer=None, extra=None):
    """Same layout as IL checkpoints (il.model.load reads it); only latest.pt keeps the optimizer."""
    base = {k: v for k, v in saved.items() if k not in ('optimizer', 'update')}
    tmp = Path(f'{path}.tmp')  # written whole, then renamed: a kill mid-write (scancel, node failure) keeps the old file
    torch.save(dict(base, model=model.state_dict(), **({'optimizer': optimizer.state_dict()} if optimizer else {}),
                    **(extra or {})), tmp)
    os.replace(tmp, path)


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
    parser.add_argument('--aux-coef', type=float, default=0.02, help="the auxiliary tower-damage head's loss "
                        "weight (its loss is ~35x the value loss: 0.5 swamped the policy gradient)")
    parser.add_argument('--entropy', type=float, default=0.002)
    parser.add_argument('--kl-coef', type=float, default=0.05)
    parser.add_argument('--pool', default='data/deck_pool.json', help='deck pairs as played (build_cache --deck-pool)')
    parser.add_argument('--cache', default='data/cache', help='deck pairs when there is no pool file')
    parser.add_argument('--device', default='auto', help="'auto': cuda if available, decided in each process after forking")
    parser.add_argument('--learner-deck', help="specialist: the learning seat always plays this deck (8 cards, id or "
                        "id:el, as il.arena --deck), against real ladder decks; only that seat's games train")
    parser.add_argument('--opponent-pool', help="frozen opponents from this directory of snapshots instead of the run's own "
                        "(specialist: the general run's pool, models that play every ladder deck)")
    parser.add_argument('--il', help="the frozen opponent called 'il' (default: init); e.g. the real IL model when init is "
                        "the General, the specialist's KL reference")
    parser.add_argument('--shaping', type=float, default=0.1, help="tower shaping's weight (FirstLight: 0.05-0.15; 0: off)")
    parser.add_argument('--il-share', type=float, help='games against the IL model among frozen-opponent games (FirstLight '
                        'specialist: 0.4); the rest PFSP over the snapshots. Default: IL is one PFSP entry')
    parser.add_argument('--deck-mix', default='0,1,0', help='specialist opponent decks: shares of the most played decks, '
                        'any pool deck, and hard decks (FirstLight: 0.3,0.4,0.3)')
    parser.add_argument('--pool-size', type=int, default=6, help='frozen opponents: the starting model + this many latest '
                        'snapshots')
    parser.add_argument('--server-device', help="the decision servers' device (default: --device), e.g. cuda:1 "
                        "with two GPUs: the learner's update and the servers stop sharing one")
    parser.add_argument('--remote', action='store_true',
                        help='accept collectors from other jobs (python -m il.rl join <out>); address in <out>/endpoint.json')
    parser.add_argument('--mps', action=argparse.BooleanOptionalAction, default=True,
                        help='NVIDIA MPS for the decision servers (where the node has it)')
    args = parser.parse_args()
    args.mps = start_mps() if args.mps else None
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
    if args.remote:
        accept_into(learner_listener, arrivals)
        ip = reachable_ip()
        endpoint = {'learner': [ip, learner_listener.address[1]],
                    'servers': [[ip, listener.address[1]] for listener in server_listeners],
                    'authkey': authkey.hex(), 'host': socket.gethostname(), 'clapha': args.clapha,
                    'out': str(out.resolve()), 'delay': args.delay, 'started': time.time(),
                    # for il.rl join --servers: the models its servers load
                    'init': str(Path(args.init).expanduser().resolve()), 'il': args.il and str(Path(args.il).expanduser().resolve()),
                    'opponent_pool': str(opponent_dir(args).resolve()), 'batch_ms': args.batch_ms}
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
    base_lr = args.lr * rollback_scale(out)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    start_update = saved.get('update', 0)
    fresh_at = start_update  # the update Adam started afresh at (warmed up from there), None: its state was loaded
    if 'optimizer' in saved:
        try:
            optimizer.load_state_dict(saved['optimizer'])
            fresh_at = None
        except ValueError:  # saved before the model gained layers: a fresh Adam
            print('the saved optimizer state does not fit the model; starting Adam afresh', flush=True)
    frozen = ['il'] + [q.stem for q in sorted(opponent_dir(args).glob('*.pt'))]
    matchmaker = Matchmaker(args, deck_pairs, learner_deck, frozen)

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
                            conn.send(matchmaker.spec())
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
                    elif kind == 'weights':  # il.rl join --servers: the current weights, unless it has them (payload)
                        with lock:
                            current = version.value
                            weights = None if payload == current else {k: v.numpy().copy() for k, v in shared.items()}
                        conn.send(None if weights is None else (current, weights))  # numpy: tensors would pickle
                                                                                     # as shared-memory handles
                    elif kind == 'new_game':
                        if max_backlog and ready.qsize() >= max_backlog:
                            held.append((conn, time.time()))
                        else:
                            conn.send(matchmaker.spec())
                    elif kind == 'trajectory':
                        matchmaker.record(payload)
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
        rollback = out / 'rollback.json'  # written by il.autopilot: restart from its best checkpoint at a lower rate
        if rollback.exists():
            order = json.loads(rollback.read_text())
            load_weights(model, load_policy(order['checkpoint'], 'cpu')[1]['model'])
            base_lr *= order['lr_scale']
            optimizer, fresh_at = torch.optim.Adam(model.parameters(), lr=base_lr), update_index + 1
            rollback.rename(out / f'rollback-done-{update_index}.json')
            publish()  # this batch was played by the weights we left: dropped
            print(f"rolled back to {order['checkpoint']}, lr {base_lr:.2e}", flush=True)
            continue
        for group in optimizer.param_groups:  # the saved state would restore the old learning rate
            group['lr'] = base_lr * (1 if fresh_at is None else min(1, (update_index - fresh_at + 1) / ADAM_WARMUP))
        matchmaker.decay()
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
                   pool={k: round(float(np.mean(v)), 3) for k, v in list(matchmaker.results.items()) if v},
                   hard_cards=[card[0] for card in matchmaker.hard_cards()],
                   lr=optimizer.param_groups[0]['lr'],
                   defense=defense.summary([t['defense'] for t in trajectories if t.get('defense')]), **stats_row)
        metrics.write(json.dumps(row) + '\n')
        metrics.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    for conn in conns:
        conn.send(None)


if __name__ == '__main__':
    main()
