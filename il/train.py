"""Imitation learning on the rebuilt replay cache.

    python -m il.train data/cache runs/il-001                      # GPU box, all shards
    python -m il.train data/cache-smoke runs/il-smoke --epochs 1 --val-replays 20

Each sample is one player's whole game (full backprop through time; games are ~700
decisions). The last shard (sorted by name) is held out for validation unless --val-shards
says otherwise. Writes metrics.jsonl, last.pt and best.pt (lowest validation NLL).
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .model import INPUT_KEYS, Policy


class Games(torch.utils.data.IterableDataset):
    """Yields per-player sequences, shard by shard; each worker reads its own shards."""

    def __init__(self, paths, vocab, seed, faithful_only=True, limit=None, shuffle=True):
        self.paths, self.vocab, self.seed = list(paths), vocab, seed
        self.faithful_only, self.limit, self.shuffle = faithful_only, limit, shuffle

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        paths = self.paths[info.id::info.num_workers] if info else self.paths
        if self.shuffle:
            paths = rng.sample(paths, len(paths))
        for path in paths:
            shard = D.Shard(path)
            order = list(range(len(shard)))[:self.limit]
            if self.shuffle:
                rng.shuffle(order)
            for i in order:
                replay = shard[i]
                if self.faithful_only and not D.faithful(replay):
                    continue
                for owner in (0, 1):
                    yield D.sequence(replay, owner, self.vocab)


def collate(sequences):
    """Pad to the longest game; `time_mask` marks real steps."""
    T = max(len(s['action']) for s in sequences)
    out = {}
    for key in sequences[0]:
        first = sequences[0][key]
        batch = np.zeros((len(sequences), T, *first.shape[1:]), first.dtype)
        for b, s in enumerate(sequences):
            batch[b, :len(s[key])] = s[key]
        out[key] = torch.from_numpy(batch)
    out['time_mask'] = torch.from_numpy(np.array([[t < len(s['action']) for t in range(T)] for s in sequences]))
    out['slot_ok'][~out['time_mask']] = False  # padded steps: only WAIT/ABILITY stay finite
    return out


def losses(model, batch, device, value_coef):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    logits, value, _ = model({k: batch[k] for k in INPUT_KEYS})
    labeled = batch['label_mask'] & batch['time_mask']
    nll = F.cross_entropy(logits[labeled].float(), batch['action'][labeled], reduction='none')
    value_loss = F.mse_loss(value[batch['time_mask']].float(), batch['value'][batch['time_mask']])
    action = batch['action'][labeled]
    pred = logits[labeled].argmax(-1)
    acts = action > D.ABILITY
    top5 = logits[labeled][acts].topk(5, -1).indices
    stats = {
        'nll': nll.mean().item(), 'value_mse': value_loss.item(),
        'wait_acc': (pred[action == D.WAIT] == D.WAIT).float().mean().item(),
        'act_rate_true': (action != D.WAIT).float().mean().item(),
        'act_rate_pred': (1 - logits[labeled].float().softmax(-1)[:, D.WAIT]).mean().item(),
        # On human card plays: chosen card right / exact (card, cell) in top-1 and top-5.
        'card_acc': ((logits[labeled][acts][:, 2:].reshape(-1, 4, D.CELLS).logsumexp(-1).argmax(-1))
                     == (action[acts] - 2) // D.CELLS).float().mean().item(),
        'cell_top1': (logits[labeled][acts][:, 2:].argmax(-1) + 2 == action[acts]).float().mean().item(),
        'cell_top5': (top5 == action[acts, None]).any(-1).float().mean().item(),
    }
    return nll.mean() + value_coef * value_loss, stats, len(action)


def evaluate(model, loader, device, value_coef):
    model.eval()
    totals, rows = {}, 0
    with torch.no_grad():
        for batch in loader:
            _, stats, n = losses(model, batch, device, value_coef)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v * n
            rows += n
    model.train()
    return {k: v / max(rows, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=16, help='player-games per step')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--val-shards', nargs='*', help='shard file names held out (default: last)')
    parser.add_argument('--val-replays', type=int, default=500, help='replays per val shard')
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--keep-diverged', action='store_true', help='also train on games whose rebuild ended differently')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', help='default: cuda if available, else cpu')
    parser.add_argument('--threads', type=int, help='torch CPU threads (keep small on a laptop)')
    parser.add_argument('--max-steps', type=int, help='stop after this many updates (smoke tests)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    # ponytail: no MPS default; a laptop GPU at full load freezes the desktop.
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    shards = sorted(args.cache.glob('part-*.npz'))
    val_names = set(args.val_shards or [shards[-1].name])
    train_paths = [p for p in shards if p.name not in val_names] or shards  # tiny smoke caches: overlap is fine
    val_paths = [p for p in shards if p.name in val_names]
    vocab_path = args.cache / 'vocab.json'
    if not vocab_path.exists():
        vocab_path.write_text(json.dumps(D.build_vocab(shards)))
    vocab = D.load_vocab(vocab_path)

    model = Policy(vocab_size=max(vocab.values()) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, best = checkpoint['step'], checkpoint['best']
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'{device}; {sum(p.numel() for p in model.parameters()):,} params; train shards {len(train_paths)}, '
          f'val {sorted(val_names)}', flush=True)

    loader_args = dict(batch_size=args.batch, collate_fn=collate, num_workers=args.workers,
                       pin_memory=device.type == 'cuda')
    val_loader = torch.utils.data.DataLoader(
        Games(val_paths, vocab, 0, not args.keep_diverged, args.val_replays, shuffle=False), **loader_args)
    autocast = torch.autocast('cuda', torch.bfloat16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)
    log = open(args.out / 'metrics.jsonl', 'a')

    def save(name):
        torch.save({'model': model.state_dict(), 'config': model.config, 'vocab': vocab,
                    'optimizer': optimizer.state_dict(), 'step': step, 'best': best}, args.out / name)

    def validate():
        nonlocal best
        with autocast:
            stats = evaluate(model, val_loader, device, args.value_coef)
        record = {'step': step, 'split': 'val', **stats}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()
        if stats['nll'] < best:
            best = stats['nll']
            save('best.pt')
        save('last.pt')

    started, running = time.perf_counter(), {}
    for epoch in range(args.epochs):
        loader = torch.utils.data.DataLoader(
            Games(train_paths, vocab, args.seed * 1000 + epoch, not args.keep_diverged), **loader_args)
        for batch in loader:
            with autocast:
                loss, stats, _ = losses(model, batch, device, args.value_coef)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            for k, v in stats.items():
                running[k] = running.get(k, 0) + v
            if step % 100 == 0:
                record = {'step': step, 'epoch': epoch, 'split': 'train',
                          'steps_per_s': round(100 / (time.perf_counter() - started), 2),
                          **{k: v / 100 for k, v in running.items()}}
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + '\n')
                started, running = time.perf_counter(), {}
            if step % args.eval_every == 0:
                validate()
            if step == args.max_steps:
                break
        if step == args.max_steps:
            break
    validate()


if __name__ == '__main__':
    main()
