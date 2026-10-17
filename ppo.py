"""PPO from the imitation model: a learner (DDP over its node's GPUs) and actors (native games and batched GPU
decisions), in as many Slurm jobs as there are CPUs to get.

    torchrun --standalone --nproc_per_node=3 ppo.py learn --init IL.pt --out RUN --engine ~/clapha-engine --pool POOL
    python ppo.py act RUN --collectors 18 [--servers 2] [--gpus 0,1]   # any job; finds the learner via RUN/endpoint.json

A collector plays one native game at a time (a second engine in one process crashes) and asks its job's decision
server for both seats' actions every decision. A server answers whichever collectors are waiting in one GPU call per
model, as soon as its last call is done (the old servers held each round for every collector: CPUs idled, 4k games/h).
Finished games go to the learner; servers fetch its newest weights. The learner trains on whatever has arrived (games more than --max-lag
versions old are dropped; the PPO ratio uses the log-probs they were played with). Rerunning the learner resumes from
RUN/latest.pt with RUN/settings.json; actors reconnect by themselves.

Opponents: self-play (--self-play of the games, both seats learn) or a frozen one by PFSP, weight (1 - win rate)^2 + 0.05,
over the IL model and the --pool-size latest snapshots. Reward: +1 win / -1 loss, -OVERFLOW_PENALTY per decision at full
elixir, -REFUSED_PENALTY per play the engine refuses, and FirstLight's tower shaping, shaping * (gamma * potential' -
potential), the potential being crowns plus lost tower hp fractions (king x3), theirs minus ours, and 0 once the game ends
(no bonus for a fast or a big win).
"""
import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener, wait
import os
import pickle
from pathlib import Path
import queue
import random
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

import common as C
import train as policy  # the inference helpers live in train.py

KING_WEIGHT = 3.0  # the king tower's share of the shaping potential (FirstLight)
OVERFLOW_PENALTY = 0.002
# Per play the engine refuses. 0 since refused plays are masked out (train.playable: unaffordable within the hold) and a
# held tap counts as in flight from the tap (PlayerView): what is left are rare mismatches, counted in refused_per_game.
REFUSED_PENALTY = 0.0
ADAM_WARMUP = 20  # updates of learning-rate ramp after a fresh Adam (a fresh Adam at full lr collapsed ppo-ebarbs2)
REPLY_TIMEOUT = 180  # s; a learner that says nothing for longer is gone (a failed node never closes its sockets)
MAX_TICKS = 7200


# ---- the PPO update -----------------------------------------------------------------------------------------------------
PRIVATE = ('hand_card', 'mirror_card', 'hand_float', 'cycle_card', 'ability')  # the opponent's view: what only they know


def advantages(reward, value, gamma, lam):
    """GAE -> (advantages, TD(lambda) returns)."""
    adv = np.zeros_like(value)
    last = 0.0
    for t in reversed(range(len(reward))):
        next_value = value[t + 1] if t + 1 < len(value) else 0.0
        last = reward[t] + gamma * next_value - value[t] + gamma * lam * last
        adv[t] = last
    return adv.astype(np.float32), (adv + value).astype(np.float32)


def upgo(reward, value, gamma):
    """UPGO (AlphaStar): the return that follows the actual path while it beats the critic's estimate (one-step
    look-ahead r' + gamma V'' >= V') and bootstraps from V' where it doesn't -> advantages (that return - value)."""
    out = np.zeros_like(value)
    for t in reversed(range(len(reward))):
        if t + 1 == len(reward):
            out[t] = reward[t]
            continue
        after = value[t + 2] if t + 2 < len(value) else 0.0
        follow = reward[t + 1] + gamma * after >= value[t + 1]
        out[t] = reward[t] + gamma * (out[t + 1] if follow else value[t + 1])
    return (out - value).astype(np.float32)


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


def minibatches(trajectories, args, device):
    """Trajectories -> [(chunk of --minibatch trajectories, model inputs [B, T, ...], the opponent's private inputs,
    the mask of real decisions)], built once per update (rebuilding them every pass made the update CPU-bound)."""
    random.shuffle(trajectories)
    out = []
    for i in range(0, len(trajectories), args.minibatch):
        chunk = trajectories[i:i + args.minibatch]
        x = policy.tensors({k: policy.pad_stack([t['inputs'][k] for t in chunk]) for k in chunk[0]['inputs']}, device)
        opp = policy.tensors({k: policy.pad_stack([t['opp'][k] for t in chunk]) for k in PRIVATE}, device)
        T = x['scalars'].shape[1]
        valid = torch.from_numpy(np.arange(T) < np.array([len(t['action']) for t in chunk])[:, None]).to(device)
        out.append((chunk, x, opp, valid))
    return out


def targets(chunk, keys, device):
    """The per-decision arrays `keys` -> [B, T] tensors."""
    return {k: torch.from_numpy(policy.pad_stack([t[k] for t in chunk])).to(device, non_blocking=True) for k in keys}


def _entropy(logp):
    return -(logp.exp() * logp.masked_fill(torch.isinf(logp), 0)).sum(-1)


def _kl(logp, ref):
    return (logp.exp() * (logp.masked_fill(torch.isinf(logp), 0) - ref.masked_fill(torch.isinf(ref), 0))).sum(-1)


def evaluate(model, x, action, valid, search=None):
    """A forward pass on a padded batch -> per real decision ([n], in the mask's order): log p(action), the what and
    card heads' log-probs [n, 4], the log-prob of the tile taken and its head's entropy (0 where no card was played),
    whether a card was played, and the value. search: (flat rows b * T + t, hand slots) -> also 'search', the tile
    log-probs [S, CELLS] of those decisions for those cards."""
    full = model(x)
    out, x = policy.flat(full, x)
    v = valid.reshape(-1)
    xv = {k: x[k][v] for k in ('own_flight_card', 'legal', 'hand_float', 'scalars')}
    what, slot, cell, play = policy.split(action.reshape(-1)[v])
    what_lp = policy.what_logp(out['what'][v], xv, policy.ABILITY_EXPLORE, out['which'][v])
    which_lp = policy.which_logp(out['which'][v], xv)
    tiles = policy.tile_logp(model, out, x, slot[play], v.nonzero().squeeze(1)[play])
    tile, tile_entropy = what_lp.new_zeros(len(what)), what_lp.new_zeros(len(what))
    tile[play] = tiles.gather(1, cell[play].unsqueeze(1)).squeeze(1)
    tile_entropy[play] = _entropy(tiles)
    pick = lambda lp, i: lp.gather(1, i.unsqueeze(1)).squeeze(1)  # noqa: E731
    logp = pick(what_lp, what) + torch.where(play, pick(which_lp, slot) + tile, torch.zeros_like(tile))
    result = dict(logp=logp, what=what_lp, which=which_lp, tile=tile, tile_entropy=tile_entropy, play=play,
                  value=full['value'].reshape(-1)[v])
    if search is not None:
        result['search'] = policy.tile_logp(model, out, x, search[1], search[0])
    return result


class Evaluate(nn.Module):
    """evaluate() as one module, so DDP sees the tile head (called outside Policy.forward) too."""

    def __init__(self, model):
        super().__init__()
        self.policy = model

    def forward(self, x, action, valid, search=None):
        return evaluate(self.policy, x, action, valid, search)


class Critic(nn.Module):
    """The values for training only (never needed to play): the policy's network (a copy, from the IL weights) over
    this seat's view, plus what only the opponent knows (hand, next cards, ability). Two heads: the logit of P(win),
    for the outcome (no discount, Monte Carlo targets), and the shaping and overflow rewards still to come (TD(lambda))."""

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk  # a Policy; its action heads go unused
        c, d = trunk.config, trunk.config.card_dim
        self.private = policy.mlp(4 * (2 * d + C.HAND_FLOATS) + 4 * d + 8, 128, 128)
        self.win = policy.mlp(c.core + C.SCALARS + 128, 256, 1)
        self.short = policy.mlp(c.core + C.SCALARS + 128, 256, 1)

    def forward(self, x, opp):
        B, T = x['scalars'].shape[:2]
        card = self.trunk.card
        private = torch.cat([card(opp['hand_card']).flatten(2), card(opp['mirror_card']).flatten(2),
                             opp['hand_float'].flatten(2), card(opp['cycle_card']).flatten(2), opp['ability'].flatten(2)], -1)
        h = torch.cat([self.trunk(x)['core'].reshape(B, T, -1), x['scalars'], F.relu(self.private(private))], -1)
        return self.win(h).squeeze(-1), self.short(h).squeeze(-1)


def critic_targets(batches, critic, args, device):
    """The critic's values for each trajectory (before it learns from them) -> advantages and targets in place;
    returns how much it knew."""
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    with torch.no_grad():
        for chunk, x, opp, valid in batches:
            with torch.autocast(**autocast):
                win, short = critic(x, opp)
            for t, w, s, n in zip(chunk, win.float().cpu().numpy(), short.float().cpu().numpy(), valid.sum(1).tolist()):
                t['v_win'], t['v_short'] = np.tanh(w[:n] / 2), s[:n]  # 2 P(win) - 1 = tanh(logit / 2)
    out = defaultdict(list)
    trajectories = [t for chunk, *_ in batches for t in chunk]
    for t in trajectories:
        n = len(t['action'])
        r_win = np.zeros(n, np.float32)
        r_win[-1] = t['outcome']
        r_short = t['reward'] - r_win
        a_win, _ = advantages(r_win, t['v_win'], 1.0, args.lam)
        a_short, t['ret_short'] = advantages(r_short, t['v_short'], args.gamma, args.lam)
        t['adv'] = a_win + a_short
        if args.upgo:
            t['adv'] += args.upgo * (upgo(r_win, t['v_win'], 1.0) + upgo(r_short, t['v_short'], args.gamma))
        t['p_win'] = np.full(n, (t['outcome'] + 1) / 2, np.float32)
        late = np.arange(n) >= 3 * n // 4
        out['outcome'].append(np.full(n, t['outcome'], np.float32))
        out['v_win'].append(t['v_win'])
        out['late'].append(late)
        out['short'].append(discounted(r_short, args.gamma))
        out['v_short'].append(t['v_short'])
        out['ret_short'].append(t['ret_short'])
    if not trajectories:
        return {}
    o = {k: np.concatenate(v) for k, v in out.items()}
    return {  # before this update: 1 = knows, 0 = no better than the mean
        'value_ev_mc': explained(o['outcome'] + o['short'], o['v_win'] + o['v_short']),
        'win_ev': explained(o['outcome'], o['v_win']),
        'win_ev_late': explained(o['outcome'][o['late']], o['v_win'][o['late']]),  # the last quarter of each game
        'short_ev_mc': explained(o['short'], o['v_short']),
        'short_ev_td': explained(o['ret_short'], o['v_short'])}


def train_critic(critic, optimizer, batches, args, device, objective=None):
    """--critic-epochs passes over the batch: P(win) by cross-entropy, the short-term returns by squared error."""
    objective = objective or critic
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    stats = defaultdict(list)
    joined = nn.parallel.distributed.Join([objective]) if isinstance(objective, nn.parallel.DistributedDataParallel) \
        else nullcontext()
    with joined:
        for _ in range(args.critic_epochs):
            for chunk, x, opp, valid in random.sample(batches, len(batches)):
                y = targets(chunk, ('p_win', 'ret_short'), device)
                with torch.autocast(**autocast):
                    win, short = objective(x, opp)
                win_loss = F.binary_cross_entropy_with_logits(win.float()[valid], y['p_win'][valid])
                short_loss = F.mse_loss(short.float()[valid], y['ret_short'][valid])
                optimizer.zero_grad(set_to_none=True)
                (win_loss + short_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
                stats['win_loss'].append(win_loss.item())
                stats['short_loss'].append(short_loss.item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def update(model, reference, optimizer, batches, args, device, objective=None):
    """One PPO update on `trajectories` (this rank's share, with critic_targets' advantages). `objective`:
    Evaluate(model), DDP-wrapped on several ranks; ranks with fewer minibatches join early."""
    objective = objective or Evaluate(model)
    trajectories = [t for chunk, *_ in batches for t in chunk]
    if trajectories:
        flat = np.concatenate([t['adv'] for t in trajectories])
        mean, std = flat.mean(), flat.std() + 1e-8
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    stats = defaultdict(list)
    # The reference (IL) policy once per trajectory: its what / card heads and the tile taken. It does not change during
    # the update, so the epochs need not run its forward pass again.
    with torch.no_grad():
        for chunk, x, _, valid in batches:
            with torch.autocast(**autocast):
                ref = evaluate(reference, x, targets(chunk, ('action',), device)['action'], valid)
            lengths = [len(t['action']) for t in chunk]
            for t, *parts in zip(chunk, *(ref[k].float().split(lengths) for k in ('what', 'which', 'tile'))):
                t['ref'] = parts
    joined = nn.parallel.distributed.Join([objective]) if isinstance(objective, nn.parallel.DistributedDataParallel) \
        else nullcontext()
    with joined:
        for _ in range(args.epochs):
            for chunk, x, _, valid in random.sample(batches, len(batches)):
                y = targets(chunk, ('action', 'logp', 'adv'), device)
                T = x['scalars'].shape[1]
                searched = sum(len(t.get('search_t', ())) for t in chunk)
                search = None
                if searched:  # the search's targets for this chunk's searched decisions
                    rows = np.concatenate([b * T + t['search_t'] for b, t in enumerate(chunk)])
                    search = (torch.from_numpy(rows).to(device), torch.from_numpy(np.concatenate(
                        [t['search_slot'] for t in chunk])).to(device))
                    cells = torch.from_numpy(np.concatenate([t['search_cells'] for t in chunk])).to(device)
                    target = torch.from_numpy(np.concatenate([t['search_target'] for t in chunk])).to(device)
                with torch.autocast(**autocast):
                    new = objective(x, y['action'], valid, search)
                ref_what, ref_which, ref_tile = (torch.cat(parts) for parts in zip(*(t['ref'] for t in chunk)))
                ratio = (new['logp'] - y['logp'][valid]).exp()
                adv = (y['adv'][valid] - mean) / std
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                p_play = new['what'][:, policy.PLAY].exp()
                ent = (_entropy(new['what']) + p_play * _entropy(new['which']) + new['tile_entropy']).mean()
                log_r = ref_tile - new['tile']  # the tile taken: KL's k3 estimate (r - 1 - log r), 0 where no card played
                kl_ref = (_kl(new['what'], ref_what) + p_play * _kl(new['which'], ref_which)
                          + torch.where(new['play'], log_r.exp() - 1 - log_r, torch.zeros_like(log_r))).mean()
                # the policy's own value head is left alone (0 x keeps a gradient on every parameter DDP expects)
                loss = policy_loss - args.entropy * ent + args.kl_coef * kl_ref + 0 * new['value'].sum()
                if search is not None:  # cross-entropy of the tile head to the search's distribution over its candidates
                    lp = new['search'].float().gather(1, cells.clamp(min=0))
                    ok = (cells >= 0) & torch.isfinite(lp)
                    search_loss = -(target * lp.masked_fill(~ok, 0)).sum(1).mean()
                    loss = loss + getattr(args, 'search_coef', 0.0) * search_loss
                    stats['search_loss'].append(search_loss.item())
                    stats['search_agree'].append((lp.masked_fill(~ok, float('-inf')).argmax(1)
                                                  == target.argmax(1)).float().mean().item())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    stats['policy_loss'].append(policy_loss.item())
                    stats['entropy'].append(ent.item())
                    stats['kl_ref'].append(kl_ref.item())
                    stats['approx_kl'].append(((ratio - 1) - ratio.log()).mean().item())
                    stats['clip_frac'].append(((ratio - 1).abs() > args.clip).float().mean().item())
    for t in trajectories:
        del t['ref']
    return {k: float(np.mean(v)) for k, v in stats.items()}


# ---- rewards ------------------------------------------------------------------------------------------------------------
def tower_hp(state):
    """{tower id: (owner, hp, king)}; a destroyed tower is absent."""
    return {e.id: (e.owner, max(e.hp or 0, 0), e.x == 9000) for e in state.entities
            if e.card == -1 and e.entity_type == 'building'}


def potential(state, start, owner):
    """Crowns plus each tower's lost hp fraction (king x KING_WEIGHT), theirs minus ours; `start` is tower_hp at the
    start."""
    towers = tower_hp(state)
    dealt, taken = float(state.crowns[owner]), float(state.crowns[1 - owner])
    for tid, (o, full, king) in start.items():
        lost = (KING_WEIGHT if king else 1.0) * (1.0 - (towers[tid][1] / max(full, 1) if tid in towers else 0.0))
        if o == owner:
            taken += lost
        else:
            dealt += lost
    return dealt - taken


# ---- search ------------------------------------------------------------------------------------------------------------
SEARCH_K = 8  # the policy's top tiles a search compares (plus the one it sampled)
SEARCH_TIMEOUT = 30  # s for one search's rollouts; a copy that hangs is killed and the search dropped


def rollout_score(engine, play, owner, start, princess, horizon, material):
    """In a forked copy of the game: drop the opponent's plays `owner` cannot see yet, make `play`, run `horizon` ticks
    with no further plays -> tower damage (the shaping potential) + material x (our units' hp - theirs, in princess
    towers)."""
    from arena import PLAY_DELAY
    now = engine.state.tick
    engine.pending[:] = [p for p in engine.pending if p['owner'] == owner or p['due'] - now <= C.OPP_VISIBLE]
    state = engine.step([play], 5, delay=sum(PLAY_DELAY) // 2).state
    while state.tick - now < horizon and not state.finalized:
        state = engine.step((), 5).state
    units = sum((e.hp if e.owner == owner else -e.hp) for e in state.entities if e.card != -1 and e.hp)
    return potential(state, start, owner) + material * units / princess


def search_tiles(engine, owner, slot, cells, start, princess, horizon, material):
    """Score playing hand slot `slot` on each of `cells`, each in its own forked copy of the game (all at once; os.fork
    copies the native engine exactly: ~12 ms) -> [score], or None if any copy failed."""
    from arena import Play
    kids = []
    for cell in cells:
        row, col = divmod(cell, C.W)
        play = Play(owner, slot, *C.rotate(col * 1000 + 500, row * 1000 + 500, owner))
        rd, wr = os.pipe()
        pid = os.fork()
        if pid == 0:  # the copy: never touches the sockets it inherited, and exits without native destructors
            try:
                os.close(rd)
                os.write(wr, pickle.dumps(rollout_score(engine, play, owner, start, princess, horizon, material)))
            finally:
                os._exit(0)
        os.close(wr)
        kids.append((pid, rd))
    scores, deadline = [], time.time() + SEARCH_TIMEOUT
    for pid, rd in kids:
        buf = b''
        while time.time() < deadline:
            ready, _, _ = select.select([rd], [], [], max(0.0, deadline - time.time()))
            chunk = os.read(rd, 1 << 16) if ready else b''
            if not chunk:
                break
            buf += chunk
        if time.time() >= deadline:
            os.kill(pid, signal.SIGKILL)
        os.close(rd)
        os.waitpid(pid, 0)
        scores.append(pickle.loads(buf) if buf else None)
    return None if any(not isinstance(x, float) for x in scores) else scores


def search_target(scores, tau):
    """Scores -> a distribution over the candidates (softmax at temperature tau), or None when they do not differ."""
    scores = np.asarray(scores, np.float64)
    if scores.max() - scores.min() < 1e-3:
        return None
    w = np.exp((scores - scores.max()) / tau)
    return (w / w.sum()).astype(np.float32)


# ---- actors -------------------------------------------------------------------------------------------------------------
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
    """A connection to one learner rank (chosen at random): finished games out, the opponents (and, with weights=True,
    the newest weights) back. Reconnects (re-reading endpoint.json) whenever the learner is gone or restarted; games it
    could not send are lost."""

    def __init__(self, run, weights=True):
        self.run, self.conn, self.version, self.weights = run, None, -1, weights

    def exchange(self, games, results, timing):
        while True:
            try:
                if self.conn is None:
                    ep = endpoint(self.run)
                    self.conn = Client((ep['host'], random.choice(ep['ports'])), authkey=bytes.fromhex(ep['authkey']))
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
        self.search = []  # (decision index, hand slot, candidate cells, target distribution)

    def trajectory(self, **extra):
        n = SEARCH_K + 1
        cells = np.full((len(self.search), n), -1, np.int64)
        target = np.zeros((len(self.search), n), np.float32)
        for i, (_, _, c, p) in enumerate(self.search):
            cells[i, :len(c)], target[i, :len(p)] = c, p
        return dict(inputs={k: policy.pad_stack([r[k] for r in self.rows]) for k in self.rows[0]},
                    opp={k: np.stack([r[k] for r in self.opp]) for k in PRIVATE},
                    action=np.array(self.action, np.int64), logp=np.array(self.logp, np.float32),
                    value=np.array(self.value, np.float32), reward=np.array(self.reward, np.float32),
                    search_t=np.array([x[0] for x in self.search], np.int64),
                    search_slot=np.array([x[1] for x in self.search], np.int64), search_cells=cells,
                    search_target=target, **extra)


def redirect(run, name):
    job = os.environ.get('SLURM_JOB_ID', socket.gethostname())
    log = os.open(run / 'logs' / f'{name}-{job}.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log, 1)
    os.dup2(log, 2)


def collector(index, run, address, key):
    """One native game at a time (the engine allows one per process), decided by its job's decision server at
    `address`; finished games go to the learner."""
    from arena import ABILITY, PLAY_DELAY, Engine, Inputs, Play, make_match
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
    link = Link(run, weights=False)
    _, opponents = link.exchange([], [], {})
    server = Client(address, authkey=key)
    timing = defaultdict(float)
    while True:
        # every game: a learner (re)started after this collector may have new settings (an actor that starts with its
        # learner can read the previous endpoint.json)
        s = endpoint(run)['settings']
        names = list(opponents)
        if names and rng.random() >= s['self_play']:
            name = rng.choices(names, [opponents[n][1] for n in names])[0]
            models = [('current', None), (name, opponents[name][0])]
            if rng.random() < 0.5:
                models.reverse()
        else:
            models = [('current', None)] * 2
        match = make_match(rng.choice(pool), rng.randrange(1 << 30))
        state = engine.reset(match=match)
        inputs = Inputs(engine, match, {0: vocab, 1: vocab}, legal)
        seats = {o: Seat() for o in (0, 1) if models[o][0] == 'current'}
        start = tower_hp(state)
        princess = max(hp for _, hp, king in start.values() if not king)
        phi = [potential(state, start, o) for o in (0, 1)]
        version, shape = None, None
        while True:
            t0 = time.perf_counter()
            rows = {o: policy.pack(x) for o, x in inputs(state).items()}
            if shape is None:
                shape = layout(rows[0])
                server.send(('reset', index, shape))  # a new game: zero LSTM states
            t1 = time.perf_counter()
            server.send(('act', index, [(o, *models[o], wire(rows[o], shape)) for o in (0, 1)]))
            replies, used = receive(server)
            version = used if version is None else version  # the game counts as played by its first weights
            t2 = time.perf_counter()
            plays, searched = [], 0.0
            for o, (a, lp, v, top) in enumerate(replies):
                if o in seats and a >= policy.CARD and rng.random() < s.get('search_frac', 0.0):
                    t3 = time.perf_counter()
                    slot, sampled = divmod(a - policy.CARD, policy.CELLS)
                    cells = list(dict.fromkeys([sampled] + [c for c in top if c >= 0]))
                    scores = search_tiles(engine, o, slot, cells, start, princess, s['search_horizon'],
                                          s['search_material'])
                    target = search_target(scores, s['search_tau']) if scores else None
                    if target is not None:
                        seats[o].search.append((len(seats[o].action), slot, cells, target))
                    timing['searches'] += 1
                    timing['search_failed'] += scores is None
                    timing['search_labels'] += target is not None
                    timing['search_changed'] += target is not None and int(np.argmax(target)) != 0
                    searched += time.perf_counter() - t3
                if o in seats:
                    seat = seats[o]
                    seat.rows.append(rows[o])
                    seat.opp.append(rows[1 - o])  # for the critic only
                    seat.action.append(a)
                    seat.logp.append(lp)
                    seat.value.append(v)
                    seat.reward.append(-OVERFLOW_PENALTY * (state.players[o].elixir >= 100000))
                choice = policy.decode(a, o)
                if choice[0] == 'ability':
                    plays.append(Play(o, ABILITY, 0, 0))
                elif choice[0] == 'card':
                    plays.append(Play(o, *choice[1:]))
            transition = engine.step(plays, 5, delay=PLAY_DELAY)
            timing['refused'] += transition.executed.count(False)
            for play, ok in zip(plays, transition.executed):
                if not ok and play.owner in seats:
                    seats[play.owner].reward[-1] -= REFUSED_PENALTY
            state = transition.state
            ended = state.finalized or state.tick >= MAX_TICKS
            for o, seat in seats.items():
                new = 0.0 if ended else potential(state, start, o)
                seat.reward[-1] += s['shaping'] * (s['gamma'] * new - phi[o])
                phi[o] = new
            timing['features'] += t1 - t0
            timing['wait'] += t2 - t1
            timing['engine'] += time.perf_counter() - t2 - searched
            timing['search'] += searched
            timing['steps'] += 1
            if ended:
                break
        out = []
        for o, seat in seats.items():
            outcome = 0.0 if state.result not in (0, 1) else 1.0 if state.result == o else -1.0
            seat.reward[-1] += outcome
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


def stage(wires, shape, B, O):
    """wire() rows -> model inputs [B, 1, ...] as numpy (zero rows after len(wires), objects padded to O)."""
    fixed, objects = shape
    n = len(wires)
    out = {}
    raw, off = np.frombuffer(b''.join(w[1] for w in wires), np.uint8).reshape(n, -1), 0
    for k, dtype, s in fixed:
        size = _size(dtype, s)
        out[k] = np.zeros((B, *s), dtype)
        out[k][:n] = raw[:, off:off + size].copy().view(dtype).reshape(n, *s)
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
        out[k] = block[:, :, off:off + size].copy().view(dtype).reshape(B, O, *s)
        off += size
    out['legal'] = np.unpackbits(out['legal'], -1)[..., :4 * policy.CELLS].reshape(B, 4, C.H, C.W).astype(bool)
    return {k: (a.astype(np.int64) if a.dtype == np.int32 else a)[:, None] for k, a in out.items()}


class Decider:
    """Batched decisions for many games: each seat's LSTM state is one row of a GPU tensor; one call per model. On a GPU
    each call replays a CUDA graph of the whole decision (forward pass and sampling) for its bucket of rows and objects:
    run op by op, a call launched ~280 kernels, 9 ms of CPU for 0.7 ms of GPU work, whatever the batch."""

    def __init__(self, models, device, core, capacity=1024, top=True):
        torch.backends.mha.set_fastpath_enabled(False)  # the learner's attention path: same numerics, and capturable
        self.models, self.device, self.capacity, self.top = models, device, capacity, top
        self.rows = {}  # (collector, owner) -> row of self.h / self.c; row `capacity` is scratch (padding rows)
        self.h = torch.zeros(1, capacity + 1, core, device=device)
        self.c = torch.zeros_like(self.h)
        self.graphs = {}  # (model name, rows, objects) -> (static inputs, static row index, static outputs, graph)
        self.pool = torch.cuda.graph_pool_handle() if device.type == 'cuda' else None
        self.shape = None  # the collectors' wire layout (sent with each 'reset')

    def _rows(self, seats):
        for seat in seats:
            if seat not in self.rows:
                if len(self.rows) >= self.capacity:
                    raise RuntimeError(f'more than {self.capacity} seats on one decision server')
                self.rows[seat] = len(self.rows)
        return [self.rows[seat] for seat in seats]

    def reset(self, collector):
        index = torch.tensor(self._rows([(collector, 0), (collector, 1)]), device=self.device)
        self.h[:, index] = 0
        self.c[:, index] = 0

    def drop(self, name):
        del self.models[name]
        for key in [k for k in self.graphs if k[0] == name]:
            del self.graphs[key]

    def _decide(self, model, x, index):
        # bf16 on the GPU, as in the PPO update, so old and new log-probs come from the same arithmetic
        with torch.no_grad(), torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda',
                                             cache_enabled=False):
            out = model(x, (self.h.index_select(1, index), self.c.index_select(1, index)))
            action, logp = policy.sample(model, out, x, policy.ABILITY_EXPLORE)
            self.h.index_copy_(1, index, out['state'][0].float())
            self.c.index_copy_(1, index, out['state'][1].float())
            if not self.top:  # no search: -1s, at no cost
                return action, logp.float(), out['value'][:, 0].float(), action.new_full((len(action), SEARCH_K), -1)
            # the sampled card's most likely tiles (what a collector's search compares); -1 for illegal ones
            f, fx = policy.flat(out, x)
            tiles = policy.tile_logp(model, f, fx, ((action - policy.CARD).clamp(min=0) // policy.CELLS))
            best, top = tiles.topk(SEARCH_K, -1)
            top = torch.where(torch.isfinite(best), top, torch.full_like(top, -1))
        return action, logp.float(), out['value'][:, 0].float(), top

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

    def act(self, requests):
        """requests: [(seat, model name, its checkpoint, wire() row)] -> [(action, logp, value, the sampled card's top
        tiles)], same order."""
        by_model = defaultdict(list)
        for i, (_, name, path, _) in enumerate(requests):
            by_model[name].append(i)
            if name not in self.models:
                self.models[name] = policy.load(path, self.device)[0]
        results = [None] * len(requests)
        for name, idx in by_model.items():
            for s in range(0, len(idx), ROW_BUCKETS[-1]):
                part = idx[s:s + ROW_BUCKETS[-1]]
                rows = [requests[i][3] for i in part]
                B = _bucket(len(rows), ROW_BUCKETS)
                O = _bucket(max(1, max(r[0] for r in rows)), OBJECT_BUCKETS)
                staged = stage(rows, self.shape, B, O)
                index = self._rows([requests[i][0] for i in part]) + [self.capacity] * (B - len(rows))
                if self.pool is None:
                    outputs = self._decide(self.models[name], {k: torch.from_numpy(v).to(self.device)
                                                               for k, v in staged.items()},
                                           torch.tensor(index, device=self.device))
                else:
                    x, static_index, outputs, graph = self._graph(name, staged)
                    for k, v in staged.items():
                        x[k].copy_(torch.from_numpy(v), non_blocking=True)
                    static_index.copy_(torch.tensor(index), non_blocking=True)
                    graph.replay()
                action, logp, value, top = (o[:len(rows)].tolist() for o in outputs)
                for i, a, lp, v, t in zip(part, action, logp, value, top):
                    results[i] = (a, lp, v, t)
        return results


WEIGHTS_POLL = 5  # s between a decision server's asks for newer weights


def server(sid, run, listener, gpu, mps):
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
    # the top tiles cost ~2 ms a step; a server started while search is off skips them
    decider = Decider({'current': current}, device, current.config.core,
                      top=endpoint(run)['settings'].get('search_frac', 0.0) > 0)
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
            for name in set(decider.models) - {'current'} - set(opponents):
                decider.drop(name)  # an opponent the learner no longer offers
            weights = None
        if time.monotonic() - polled > WEIGHTS_POLL:
            weights, opponents = link.exchange([], [], {})
            polled = time.monotonic()
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
            replies = decider.act([r for _, payload in waiting for r in payload])
            k = 0
            for conn, payload in waiting:
                conn.send((replies[k:k + len(payload)], link.version))
                k += len(payload)
            stats['batches'] += 1
            stats['rows'] += k
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
    from arena import raise_thread_limit
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
        target = (server, i, run, listeners[i], gpus[i % len(gpus)] if gpus else None, mps) if kind == 'server' else \
            (collector, i, run, listeners[i % servers].address, key)
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


# ---- the learner --------------------------------------------------------------------------------------------------------
class Inbox:
    """This rank's actors: finished games in, the newest weights out. A thread answers them, so actors never wait for
    an update to finish."""

    def __init__(self, authkey, reply):
        self.listener = Listener(('0.0.0.0', 0), authkey=authkey)
        self.port = self.listener.address[1]
        self.lock, self.reply = threading.Lock(), reply
        self.games, self.results, self.timing, self.conns = [], [], [], []
        self.arrivals = queue.Queue()
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._serve, daemon=True).start()

    def _accept(self):
        while True:
            try:
                self.arrivals.put(self.listener.accept())
            except Exception as error:  # noqa: BLE001 - a failed handshake (wrong key, port scan) is not fatal
                print(f'refused a connection: {error!r}', flush=True)

    def _serve(self):
        while True:
            while not self.arrivals.empty():
                self.conns.append(self.arrivals.get())
            if not self.conns:
                time.sleep(0.1)
                continue
            for conn in wait(self.conns, timeout=0.5):
                try:
                    held, games, results, timing = conn.recv()
                    with self.lock:
                        self.games += games
                        self.results += results
                        if timing:
                            self.timing.append(timing)
                        version, weights, opponents = self.reply
                    conn.send((version, None if held in (version, None) else weights, opponents))
                except (EOFError, OSError):  # its actor died, or its job ended
                    self.conns.remove(conn)

    def count(self):
        with self.lock:
            return len(self.games)

    def take(self, oldest):
        """-> (games played from version `oldest` on, results, actor timings, games dropped as too old)."""
        with self.lock:
            games, results, timing = self.games, self.results, self.timing
            self.games, self.results, self.timing = [], [], []
        fresh = [g for g in games if g['version'] >= oldest]
        return fresh, results, timing, len(games) - len(fresh)


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


DEFAULTS = dict(updates=100000, batch=512, max_lag=4, lr=2e-5, epochs=2, minibatch=8, clip=0.2, gamma=0.999, lam=0.95,
                critic_lr=1e-4, critic_epochs=4, entropy=0.002, kl_coef=0.05, value_warmup=20, self_play=0.5,
                pool_size=6, snapshot_every=100, checkpoint_every=10, shaping=0.1, upgo=0.0,
                search_frac=0.0, search_horizon=160, search_material=0.3, search_tau=0.02, search_coef=1.0)


def learn(args, settings):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = torch.cuda.is_available()
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo')
    out, init = args.out, str(args.init.expanduser().resolve())
    s = argparse.Namespace(**settings)
    model, vocab, saved = policy.load(init, device)
    reference = policy.load(str(args.reference.expanduser().resolve()) if args.reference else init, device)[0].requires_grad_(False)
    critic = Critic(policy.load(init, device)[0]).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=s.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=s.critic_lr)
    version = fresh = 0
    if (out / 'latest.pt').exists():
        latest = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(latest['model'])
        optimizer.load_state_dict(latest['optimizer'])
        critic.load_state_dict(latest['critic'])
        critic_optimizer.load_state_dict(latest['critic_optimizer'])
        version = latest['version']
        fresh = version - ADAM_WARMUP  # its Adam moments are back: no warm-up
    model.train()
    objective, critic_objective = Evaluate(model), critic
    if world > 1:
        objective, critic_objective = (
            nn.parallel.DistributedDataParallel(m, device_ids=[device.index] if cuda else None,
                                                find_unused_parameters=True) for m in (objective, critic))

    def weights():
        return {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}

    results = defaultdict(lambda: deque(maxlen=200))  # opponent -> the learning seat's recent outcomes

    def opponents():
        """{name: (checkpoint, PFSP weight)}: the IL model and the latest snapshots."""
        names = {'il': init, **{p.stem: str(p) for p in sorted((out / 'pool').glob('v*.pt'))[-s.pool_size:]}}
        win = {n: (np.mean(results[n]) + 1) / 2 if results[n] else 0.5 for n in names}
        return {n: (path, (1 - win[n]) ** 2 + 0.05) for n, path in names.items()}

    key = [os.urandom(16).hex()]
    if world > 1:
        dist.broadcast_object_list(key)
    inbox = Inbox(bytes.fromhex(key[0]), (version, weights(), opponents()))
    ports = [None] * world
    if world > 1:
        dist.all_gather_object(ports, inbox.port)
    else:
        ports = [inbox.port]
    if rank == 0:
        (out / 'vocab.json').write_text(json.dumps(saved['vocab']))
        tmp = out / 'endpoint.tmp'
        tmp.write_text(json.dumps(dict(host=reachable_ip(), ports=ports, authkey=key[0], init=init, settings=dict(
            settings, engine=str(args.engine.expanduser().resolve()), pool=str(args.pool.expanduser().resolve())))))
        tmp.replace(out / 'endpoint.json')
        print(f'learner on {socket.gethostname()} ports {ports}, version {version}', flush=True)
    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None
    started = last = time.monotonic()
    total_games = 0
    while version < s.updates:
        t0 = time.monotonic()
        while True:  # until the ranks hold a batch between them
            n = torch.tensor(inbox.count(), device=device)
            if world > 1:
                dist.all_reduce(n)
            if int(n) >= s.batch:
                break
            time.sleep(1)
        games, done, timing, stale = inbox.take(version - s.max_lag)
        # ponytail: a learner slower than its actors would take ever bigger batches; the newest 2 x --batch are kept
        cap = 2 * s.batch // world
        stale, games = stale + max(0, len(games) - cap), games[-cap:]
        lag = [version - g['version'] for g in games]
        t1 = time.monotonic()
        ramp = min(1.0, (version - fresh + 1) / ADAM_WARMUP)
        lr = s.lr * ramp
        for group in optimizer.param_groups:
            group['lr'] = lr
        for group in critic_optimizer.param_groups:
            group['lr'] = s.critic_lr * ramp
        batches = minibatches(games, s, device)
        stats = dict(prepare_s=time.monotonic() - t1)
        stats.update(critic_targets(batches, critic, s, device))
        if version >= s.value_warmup:  # before that, the critic alone learns (the policy's advantages need it)
            stats.update(update(model, reference, optimizer, batches, s, device, objective))
        t_policy = time.monotonic()
        stats.update(train_critic(critic, critic_optimizer, batches, s, device, critic_objective),
                     critic_s=time.monotonic() - t_policy)
        del batches
        version += 1
        t2 = time.monotonic()
        mine = dict(stats=stats, results=done, timing=timing, stale=stale, lag=lag, games=len(games),
                    decisions=sum(len(g['action']) for g in games), actors=len(inbox.conns))
        everyone = [None] * world
        if world > 1:
            dist.all_gather_object(everyone, mine)
        else:
            everyone = [mine]
        for r in everyone:
            for row in r['results']:
                if row['opponent'] != 'self':
                    results[row['opponent']].append(row['outcome'])
        if rank == 0 and version % s.snapshot_every == 0:
            torch.save(dict(config=saved['config'], model=model.state_dict(), vocab=saved['vocab'], version=version),
                       out / 'pool' / 'snapshot.tmp')
            (out / 'pool' / 'snapshot.tmp').replace(out / 'pool' / f'v{version:06d}.pt')
        if rank == 0 and version % s.checkpoint_every == 0:
            torch.save(dict(config=saved['config'], model=model.state_dict(), optimizer=optimizer.state_dict(),
                            critic=critic.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                            vocab=saved['vocab'], version=version, step=version), out / 'latest.tmp')
            (out / 'latest.tmp').replace(out / 'latest.pt')
        if world > 1:
            dist.barrier()  # the new snapshot is on disk before any rank offers it
        with inbox.lock:
            inbox.reply = (version, weights(), opponents())
        if rank == 0:
            now = time.monotonic()
            finished = [row for r in everyone for row in r['results']]
            total_games += len(finished)
            times = [t for r in everyone for t in r['timing']]
            steps = max(1.0, sum(t.get('steps', 0) for t in times))
            keys = everyone[0]['stats'].keys() | set().union(*(r['stats'].keys() for r in everyone))
            row = dict(
                version=version, hours=round((now - started) / 3600, 4), games=total_games,
                games_per_hour=round(len(finished) / max(now - last, 1e-6) * 3600),
                trajectories=sum(r['games'] for r in everyone), decisions=sum(r['decisions'] for r in everyone),
                stale=sum(r['stale'] for r in everyone), lag=float(np.mean([x for r in everyone for x in r['lag']] or [0])),
                actors=sum(r['actors'] for r in everyone), wait_s=round(t1 - t0, 1), update_s=round(t2 - t1, 1), lr=lr,
                value_only=version - 1 < s.value_warmup,
                actor_ms={k: round(1000 * sum(t.get(k, 0) for t in times) / steps, 2)
                          for k in ('features', 'wait', 'engine', 'search')},
                search={k: round(sum(t.get(k, 0) for t in times) / max(1, sum(t.get('games', 0) for t in times)), 2)
                        for k in ('searches', 'search_labels', 'search_changed', 'search_failed')},
                refused_per_game=round(sum(t.get('refused', 0) for t in times) / max(1, sum(t.get('games', 0) for t in times)), 2),
                score={n: round(float(np.mean(results[n])), 3) for n in sorted(results) if results[n]},
                ticks=float(np.mean([row['tick'] for row in finished] or [0])),
                **{k: float(np.mean([r['stats'][k] for r in everyone if k in r['stats']])) for k in sorted(keys)})
            last = now
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('learn')
    p.add_argument('--init', type=Path, required=True, help='the IL checkpoint: start, KL reference, frozen opponent')
    p.add_argument('--reference', type=Path, help='the KL reference instead of --init (e.g. a recent snapshot)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True, help='the clapha-engine checkout')
    p.add_argument('--pool', type=Path, required=True, help='deck pairs as played (arena.py --pool)')
    for k, v in DEFAULTS.items():
        p.add_argument('--' + k.replace('_', '-'), dest=k, type=type(v), default=v)
    p = sub.add_parser('act')
    p.add_argument('run', type=Path)
    p.add_argument('--collectors', type=int, default=16, help='one native game each')
    p.add_argument('--servers', type=int, help='decision servers (default one per GPU)')
    p.add_argument('--gpus', help='indices of the visible GPUs to use (default all; none: CPU)')
    p.add_argument('--mps', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.command == 'act':
        return act(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'pool').mkdir(exist_ok=True)
    saved = args.out / 'settings.json'
    settings = {k: getattr(args, k) for k in DEFAULTS}
    if saved.exists():
        settings = {**settings, **json.loads(saved.read_text())}  # a resumed run keeps its settings
    elif int(os.environ.get('RANK', 0)) == 0:
        saved.write_text(json.dumps(settings, indent=1))
    learn(args, settings)


if __name__ == '__main__':
    main()
