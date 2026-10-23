"""PPO from the imitation model: a learner (DDP over its node's GPUs) and actors (native games and batched GPU
decisions), in as many Slurm jobs as there are CPUs to get.

    torchrun --standalone --nproc_per_node=3 ppo.py learn --init IL.pt --out RUN --engine ~/clapha-engine --pool POOL
    python ppo.py act RUN --collectors 18 [--servers 2] [--gpus 0,1]   # any job; finds the learner via RUN/endpoint.json

The actors (cr/actors.py: collectors and decision servers) send finished games; servers fetch the newest weights. The
learner trains on whatever has arrived (games more than --max-lag versions old are dropped; the PPO ratio uses the
log-probs they were played with). Rerunning the learner resumes from
RUN/latest.pt with RUN/settings.json; actors reconnect by themselves.

Opponents: self-play (--self-play of the games, both seats learn) or a frozen one by PFSP, weight (1 - win rate)^2 + 0.05,
over the IL model and the --pool-size latest snapshots. Reward: +1 win / -1 loss, -OVERFLOW_PENALTY per decision at full
elixir when enemy troops are on the board (no penalty on a quiet board: avoids panic deploys), and tower shaping, shaping * (gamma * potential' - potential), the potential being crowns plus lost
tower hp fractions (all towers weighted equally), theirs minus ours, and 0 once the game ends
(no bonus for a fast win). At the end, + margin x (our lowest tower's hp fraction - theirs; a destroyed tower is 0):
a close loss is worth more than a 3-crown one, and a draw is no longer flat.

The critic is its own network, from --critic-init (default --init): it is never needed to play, so it can be bigger than
the policy.

loop.py drives a run through small files in RUN: champion.json (the KL reference: the champion, replaced when a
challenger is promoted), eval.json (evaluation matches between frozen models: --eval-frac of the collectors' games
while it lists any; results in evals.jsonl) and rollback.json (back to a champion's full checkpoint, ckpt/vN.pt, saved
with every snapshot). Promoted champions in RUN/league stay PFSP opponents for good. A minibatch whose gradient is not
finite is skipped (metrics skipped / critic_skipped: the fraction of steps).

Frozen models (opponents, evaluation and benchmark games) choose cards and tiles at --frozen-temperature: models are
deployed sharpened (a lower temperature), so they are judged and met that way. The learner plays at
--learner-temperature, and its PPO update uses log-probs at that temperature, so the policy trained is the one played;
--learner-mix of the untempered distribution mixed in lets it explore around that deployed policy (deploy without it).

An exploiter: "against": the main run's directory and "self_play": 0 in its own RUN/settings.json; it then plays only
the main run's champion (RUN/champion.json there), to find what beats it.

A specialist: put "deck" (a deck as in the pool) and "self_play": 0 in RUN/settings.json before the first start; a
learning seat then always plays that deck, against the pool's decks.
"""
import argparse
from collections import defaultdict, deque
import json
from multiprocessing.connection import Listener, wait
import os
from pathlib import Path
import queue
import random
import socket
import threading
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from cr import actors, matchmaking, observation as C, policy
from cr.actors import PRIVATE

ADAM_WARMUP = 20  # updates of learning-rate ramp after a fresh Adam (a fresh Adam at full lr collapsed ppo-ebarbs2)


# ---- the PPO update -----------------------------------------------------------------------------------------------------
def advantages(reward, value, gamma, lam):
    """GAE -> (advantages, TD(lambda) returns)."""
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


def evaluate(model, x, action, valid):
    """A forward pass on a padded batch -> per real decision ([n], in the mask's order): log p(action), the what and
    card heads' log-probs [n, 4], the log-prob of the tile taken and its head's entropy (0 where no card was played),
    whether a card was played, and the value."""
    full = model(x)
    out, x = policy.flat(full, x)
    v = valid.reshape(-1)
    xv = {k: x[k][v] for k in ('own_flight_card', 'legal', 'hand_float', 'scalars')}
    what, slot, cell, play = policy.split(action.reshape(-1)[v])
    what_lp = policy.what_logp(out['what'][v], xv, policy.ABILITY_EXPLORE, out['which'][v])
    t, mix = getattr(model, 'temperature', 1.0), getattr(model, 'mix', 0.0)  # as it plays: the distribution it sampled
    which_lp = policy.sharpen(policy.which_logp(out['which'][v], xv), t, mix)
    tiles = policy.sharpen(policy.tile_logp(model, out, x, slot[play], v.nonzero().squeeze(1)[play]), t, mix)
    tile, tile_entropy = what_lp.new_zeros(len(what)), what_lp.new_zeros(len(what))
    tile[play] = tiles.gather(1, cell[play].unsqueeze(1)).squeeze(1)
    tile_entropy[play] = _entropy(tiles)
    pick = lambda lp, i: lp.gather(1, i.unsqueeze(1)).squeeze(1)  # noqa: E731
    logp = pick(what_lp, what) + torch.where(play, pick(which_lp, slot) + tile, torch.zeros_like(tile))
    return dict(logp=logp, what=what_lp, which=which_lp, tile=tile, tile_entropy=tile_entropy, play=play,
                value=full['value'].reshape(-1)[v])


class Evaluate(nn.Module):
    """evaluate() as one module, so DDP sees the tile head (called outside Policy.forward) too."""

    def __init__(self, model):
        super().__init__()
        self.policy = model

    def forward(self, x, action, valid):
        return evaluate(self.policy, x, action, valid)


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


def finite_step(parameters, optimizer, max_norm, stats, key):
    """Clip the gradient and step only if it is finite -> whether it stepped. After DDP's all-reduce every rank sees the
    same norm, so all skip together: one bad minibatch is dropped instead of turning every weight into NaN (the fork
    test of 19 October lost its policy that way). stats[key]: the fraction of steps skipped."""
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    ok = bool(torch.isfinite(norm))
    if ok:
        optimizer.step()
    stats[key].append(float(not ok))
    return ok


def schedule(batches, passes):
    """`passes` shuffled passes over this rank's minibatches, as many steps on every rank: the most any rank has, a rank
    with fewer repeating some of its own. (DDP's Join let a rank that ran out skip optimizer steps: only the weights are
    re-synced when Join ends, so its Adam moments and step count then differed from the others' for good.)"""
    counts = [len(batches)]
    if dist.is_initialized():
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, len(batches))
    if min(counts) == 0:  # learn() waits until every rank has games; skipping silently here once stalled training
        raise RuntimeError(f'a rank has no minibatches: {counts}')
    return [b for _ in range(passes)
            for b in random.sample(batches, len(batches)) + random.choices(batches, k=max(counts) - len(batches))]


def train_critic(critic, optimizer, batches, args, device, objective=None):
    """--critic-epochs passes over the batch: P(win) by cross-entropy, the short-term returns by squared error."""
    objective = objective or critic
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    stats = defaultdict(list)
    for chunk, x, opp, valid in schedule(batches, args.critic_epochs):
        y = targets(chunk, ('p_win', 'ret_short'), device)
        with torch.autocast(**autocast):
            win, short = objective(x, opp)
        win_loss = F.binary_cross_entropy_with_logits(win.float()[valid], y['p_win'][valid])
        short_loss = F.mse_loss(short.float()[valid], y['ret_short'][valid])
        optimizer.zero_grad(set_to_none=True)
        (win_loss + short_loss).backward()
        finite_step(critic.parameters(), optimizer, 1.0, stats, 'critic_skipped')
        stats['win_loss'].append(win_loss.item())
        stats['short_loss'].append(short_loss.item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def update(model, reference, optimizer, batches, args, device, objective=None):
    """One PPO update on `trajectories` (this rank's share, with critic_targets' advantages). `objective`:
    Evaluate(model), DDP-wrapped on several ranks, every rank taking the same number of steps (schedule)."""
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
    for chunk, x, _, valid in schedule(batches, args.epochs):
        y = targets(chunk, ('action', 'logp', 'adv'), device)
        with torch.autocast(**autocast):
            new = objective(x, y['action'], valid)
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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite_step(model.parameters(), optimizer, 0.5, stats, 'skipped')
        with torch.no_grad():
            stats['policy_loss'].append(policy_loss.item())
            stats['entropy'].append(ent.item())
            stats['kl_ref'].append(kl_ref.item())
            stats['approx_kl'].append(((ratio - 1) - ratio.log()).mean().item())
            stats['clip_frac'].append(((ratio - 1).abs() > args.clip).float().mean().item())
    for t in trajectories:
        del t['ref']
    return {k: float(np.mean(v)) for k, v in stats.items()}


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

    def count(self, oldest=0):
        """Games played from version `oldest` on (what take(oldest) returns at least)."""
        with self.lock:
            return sum(g['version'] >= oldest for g in self.games)

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
                pool_size=6, snapshot_every=100, checkpoint_every=10, shaping=0.1, margin=0.2, eval_frac=0.25,
                channels_last=True, against='', frozen_temperature=1.0, learner_temperature=1.0,
                learner_mix=0.0)



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
    trunk, critic_vocab, _ = policy.load(str(args.critic_init.expanduser().resolve()) if args.critic_init else init, device)
    assert critic_vocab.index == vocab.index, 'the critic reads the same inputs: same vocabulary'
    critic = Critic(trunk).to(device).train()
    model.temperature = reference.temperature = s.learner_temperature  # trains (and is anchored) as it plays
    model.mix = reference.mix = s.learner_mix
    optimizer = torch.optim.Adam(model.parameters(), lr=s.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=s.critic_lr)
    version = fresh = 0
    rolled = None  # the last rollback applied (RUN/rollback.json's id)
    if (out / 'latest.pt').exists():
        latest = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(latest['model'])
        optimizer.load_state_dict(latest['optimizer'])
        critic.load_state_dict(latest['critic'])
        critic_optimizer.load_state_dict(latest['critic_optimizer'])
        if 'reference' in latest:  # the anchor it had
            reference.load_state_dict(latest['reference'])
        version, rolled = latest['version'], latest.get('rollback')
        fresh = version - ADAM_WARMUP  # its Adam moments are back: no warm-up
    elif args.critic_weights:  # a new run with a trained critic (critic_offline.py's latest.pt: same network as the trunk)
        critic.load_state_dict(torch.load(args.critic_weights, map_location='cpu', weights_only=False)['critic'])
    model.train()
    if cuda and s.channels_last:
        for trunk in (model, reference, critic.trunk):
            policy.convolution_layout(trunk)
    objective, critic_objective = Evaluate(model), critic
    if world > 1:
        objective, critic_objective = (
            nn.parallel.DistributedDataParallel(m, device_ids=[device.index] if cuda else None,
                                                find_unused_parameters=True) for m in (objective, critic))

    def weights():
        return {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}

    def control(name):
        """RUN/<name>.json, written by loop.py: rank 0 reads it, every rank gets the same dict (or None)."""
        found = [None]
        if rank == 0:
            try:
                found = [json.loads((out / f'{name}.json').read_text())]
            except (OSError, ValueError):  # absent, or caught mid-write: the next update sees it
                pass
        if world > 1:
            dist.broadcast_object_list(found)
        return found[0]

    def follow_champion(current):
        """The KL reference is loop.py's champion (RUN/champion.json, replaced on promotion) -> its name."""
        spec = control('champion')
        if spec and spec['name'] != current:
            reference.load_state_dict(policy.load(spec['path'], device)[0].state_dict())
            return spec['name']
        return current

    def full():
        return dict(config=saved['config'], model=model.state_dict(), optimizer=optimizer.state_dict(),
                    critic=critic.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                    reference=reference.state_dict(), vocab=saved['vocab'], version=version, step=version,
                    rollback=rolled)

    champion = follow_champion(None)
    floor = version  # games played before this start (old settings) or a rollback are not learned from
    results = defaultdict(lambda: deque(maxlen=200))  # opponent -> the learning seat's recent outcomes

    def opponents():
        """{name: (checkpoint, PFSP weight)} over opponent_paths."""
        return matchmaking.pfsp(matchmaking.opponent_paths(out, init, s), results)

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
        oldest, warned = max(version - s.max_lag, floor), t0
        while True:  # until the ranks hold a batch between them, and every rank some of it (schedule() needs them all)
            total = torch.tensor(inbox.count(oldest), device=device)
            least = total.clone()
            if world > 1:
                dist.all_reduce(total)
                dist.all_reduce(least, op=dist.ReduceOp.MIN)
            if int(total) >= s.batch and int(least) > 0:
                break
            if int(total) >= s.batch and time.monotonic() - warned > 300:
                print(f'rank {rank}: a batch is waiting, but a rank has no fresh games ({inbox.count(oldest)} here, '
                      f'{len(inbox.conns)} actors)', flush=True)
                warned = time.monotonic()
            time.sleep(1)
        games, done, timing, stale = inbox.take(oldest)
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
        t_targets = time.monotonic()
        stats.update(critic_targets(batches, critic, s, device))
        stats['targets_s'] = time.monotonic() - t_targets
        t_policy = time.monotonic()
        if version >= s.value_warmup:  # before that, the critic alone learns (the policy's advantages need it)
            stats.update(update(model, reference, optimizer, batches, s, device, objective))
        stats['policy_s'] = time.monotonic() - t_policy
        t_critic = time.monotonic()
        stats.update(train_critic(critic, critic_optimizer, batches, s, device, critic_objective),
                     critic_s=time.monotonic() - t_critic)
        del batches
        version += 1
        champion = follow_champion(champion)
        spec = control('rollback')
        if spec and spec['id'] != rolled:  # loop.py: back to a champion's full checkpoint, keeping the version count
            back = torch.load(out / spec['path'], map_location='cpu', weights_only=False)
            model.load_state_dict(back['model'])
            optimizer.load_state_dict(back['optimizer'])
            critic.load_state_dict(back['critic'])
            critic_optimizer.load_state_dict(back['critic_optimizer'])
            rolled, floor = spec['id'], version
            stats['rolled_back'] = 1.0
        t2 = time.monotonic()
        judged = [row for row in done if 'eval' in row]
        mine = dict(stats=stats, results=[row for row in done if 'eval' not in row], judged=judged, timing=timing,
                    stale=stale, lag=lag, games=len(games), decisions=sum(len(g['action']) for g in games),
                    trained=sum(0.5 if g['opponent'] == 'current' else 1.0 for g in games), actors=len(inbox.conns))
        everyone = [None] * world
        if world > 1:
            dist.all_gather_object(everyone, mine)
        else:
            everyone = [mine]
        for r in everyone:
            for row in r['results']:
                if row['opponent'] != 'self':
                    results[row['opponent']].append(row['outcome'])
        if rank == 0 and version % s.snapshot_every == 0:  # a policy snapshot (opponent, gate candidate) and its full
            torch.save(dict(config=saved['config'], model=model.state_dict(), vocab=saved['vocab'], version=version),
                       out / 'pool' / 'snapshot.tmp')  # checkpoint (what a rollback returns to)
            (out / 'pool' / 'snapshot.tmp').replace(out / 'pool' / f'v{version:06d}.pt')
            torch.save(full(), out / 'ckpt' / 'ckpt.tmp')
            (out / 'ckpt' / 'ckpt.tmp').replace(out / 'ckpt' / f'v{version:06d}.pt')
        if rank == 0 and version % s.checkpoint_every == 0:
            torch.save(full(), out / 'latest.tmp')
            (out / 'latest.tmp').replace(out / 'latest.pt')
        if rank == 0:
            with (out / 'evals.jsonl').open('a') as f:  # evaluation games, for loop.py
                for row in (row for r in everyone for row in r['judged']):
                    f.write(json.dumps(dict(row, version=version)) + '\n')
        if world > 1:
            dist.barrier()  # the new snapshot is on disk before any rank offers it
        with inbox.lock:
            inbox.reply = (version, weights(), opponents())
        if rank == 0:
            now = time.monotonic()
            finished = [row for r in everyone for row in r['results']]
            judged = sum(len(r['judged']) for r in everyone)
            total_games += len(finished) + judged
            times = [t for r in everyone for t in r['timing']]
            steps = max(1.0, sum(t.get('steps', 0) for t in times))
            keys = everyone[0]['stats'].keys() | set().union(*(r['stats'].keys() for r in everyone))
            row = dict(
                version=version, hours=round((now - started) / 3600, 4), games=total_games,
                games_per_hour=round((len(finished) + judged) / max(now - last, 1e-6) * 3600),
                trained_games_per_hour=round(sum(r['trained'] for r in everyone) / max(now - last, 1e-6) * 3600),
                eval_games=judged, champion=champion,
                trajectories=sum(r['games'] for r in everyone), decisions=sum(r['decisions'] for r in everyone),
                learning_decisions_per_second=round(sum(r['decisions'] for r in everyone) / max(now - last, 1e-6), 1),
                stale=sum(r['stale'] for r in everyone), lag=float(np.mean([x for r in everyone for x in r['lag']] or [0])),
                actors=sum(r['actors'] for r in everyone), wait_s=round(t1 - t0, 1), update_s=round(t2 - t1, 1), lr=lr,
                value_only=version - 1 < s.value_warmup,
                actor_ms={k: round(1000 * sum(t.get(k, 0) for t in times) / steps, 2)
                          for k in ('features', 'wait', 'engine')},
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
    p.add_argument('--critic-init', type=Path, help="the critic's network instead of --init's (e.g. a bigger IL model)")
    p.add_argument('--critic-weights', type=Path, help="a new run's critic weights ({'critic': ...}, e.g. critic_offline.py's)")
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True, help='the clapha-engine checkout')
    p.add_argument('--pool', type=Path, required=True, help='deck pairs as played (arena.py --pool)')
    for k, v in DEFAULTS.items():
        p.add_argument('--' + k.replace('_', '-'), dest=k, default=v,
                       **({'action': argparse.BooleanOptionalAction} if isinstance(v, bool) else {'type': type(v)}))
    p = sub.add_parser('act')
    p.add_argument('run', type=Path)
    p.add_argument('--collectors', type=int, default=16, help='one native game each')
    p.add_argument('--servers', type=int, help='decision servers (default one per GPU)')
    p.add_argument('--gpus', help='indices of the visible GPUs to use (default all; none: CPU)')
    p.add_argument('--mps', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fast-inference', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--opponent-affinity', action=argparse.BooleanOptionalAction, default=True,
                   help='route already-sampled opponents to balanced server cohorts')
    args = parser.parse_args()
    if args.command == 'act':
        return actors.act(args)
    args.out.mkdir(parents=True, exist_ok=True)
    for d in ('pool', 'ckpt', 'league'):
        (args.out / d).mkdir(exist_ok=True)
    saved = args.out / 'settings.json'
    settings = {k: getattr(args, k) for k in DEFAULTS}
    if saved.exists():
        settings = {**settings, **json.loads(saved.read_text())}  # a resumed run keeps its settings
    elif int(os.environ.get('RANK', 0)) == 0:
        saved.write_text(json.dumps(settings, indent=1))
    learn(args, settings)


if __name__ == '__main__':
    main()
