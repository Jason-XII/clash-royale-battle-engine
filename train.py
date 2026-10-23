"""Imitation learning on a converted dataset: the training loop (the network is cr/policy.py, the loss and data
cr/imitation.py).

    python train.py --data DATASET --out RUN                                  # one GPU (or CPU: --device cpu)
    torchrun --standalone --nproc_per_node=3 train.py --data DATASET --out RUN   # three GPUs, --batch is global

Rerunning the same command resumes from RUN/latest.pt with the settings saved there.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.distributed as dist

from cr import observation as C
from cr.imitation import Games, Objective, collate, loss_and_metrics, summarize
from cr.policy import Config, Policy


# ---- training -----------------------------------------------------------------------------------------------------------
DEFAULTS = dict(epochs=4.0, batch=32, chunk=128, workers=4, lr=3e-4, final_lr=1.0, weight_decay=0.01, warmup=500, value_coef=0.5,
                next_play=0.5, card_dim=32, object_dim=32, attention_layers=1, conv_channels=48, conv_layers=4, core=256, board=0,
                res_blocks=0, flight_grid=0, tile_blocks=0, board_rows=8, board_cols=4, res_norm=1, dist_bias=0,
                card_stats=0, ability_labels=0, max_train=0, max_val=0,
                seed=0, log_every=50, checkpoint_every=1000, device='cuda')
# final_lr: the learning rate falls linearly to lr x final_lr by the last step (1.0 = constant)
# medium model (3.4M, 2026-10-16): --card_dim 64 --object_dim 96 --attention_layers 3 --conv_channels 96 --conv_layers 6
# --core 384
# large model (8.0M, 2026-10-18): --card_dim 64 --object_dim 160 --attention_layers 6 --conv_channels 160 --conv_layers 12
# --core 448 --board 32


def lockstep(batches, device, world):
    """`batches` while every rank still has one: the ranks step together, and a pass ends for all when one rank's share
    runs out (the rest of the others' shares is skipped this pass). One tiny all-reduce a step. (DDP's Join let a rank
    that ran out skip optimizer steps: only the weights are re-synced when Join ends, its Adam moments stayed behind.)"""
    for item in batches:
        if world > 1:
            more = torch.ones((), device=device)
            dist.all_reduce(more, op=dist.ReduceOp.MIN)
            if not more:
                return
        yield item
    if world > 1:  # out of batches: tell the others, at the same step
        dist.all_reduce(torch.zeros((), device=device), op=dist.ReduceOp.MIN)


def train(data, out, settings):
    out = Path(out)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    cuda = settings['device'] == 'cuda'
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if cuda else torch.device('cpu')
    torch.set_num_threads(2)
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(minutes=15))
    manifest = json.loads((Path(data) / 'manifest.json').read_text())
    vocab = C.Vocab(manifest['schema']['vocab'])
    train_entries = [e for e in manifest['entries'] if not e['validation']]
    val_entries = [e for e in manifest['entries'] if e['validation']]
    if settings.get('max_train'):  # ablations: the same random subset for every run with the same seed
        train_entries = random.Random(settings['seed']).sample(train_entries, min(settings['max_train'], len(train_entries)))
    if settings.get('max_val'):
        val_entries = random.Random(settings['seed'] + 1).sample(val_entries, min(settings['max_val'], len(val_entries)))
    batch = max(1, settings['batch'] // world)
    # ponytail: an epoch is an estimate (random chunk offsets vary the count a little); exact epoch ends are not kept
    total = int(settings['epochs'] * sum(math.ceil(e['decisions'] / settings['chunk']) for e in train_entries) // settings['batch'])

    torch.manual_seed(settings['seed'])
    stats = C.card_matrix(vocab) if settings.get('card_stats') else None  # --card_stats 1: the table's fields
    config = Config(vocab_size=len(vocab), next_play=settings['next_play'] > 0, **{k: settings[k] for k in
                    ('card_dim', 'object_dim', 'attention_layers', 'conv_channels', 'conv_layers', 'core')},
                    board=settings.get('board', 0), card_stats=0 if stats is None else stats.shape[1],
                    **{k: settings.get(k, v) for k, v in dict(res_blocks=0, flight_grid=0, tile_blocks=0, board_rows=8,
                                                              board_cols=4, res_norm=1, dist_bias=0).items()})
    policy = Policy(config, C.effect_matrix(vocab), stats).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'], fused=cuda)
    step = 0
    if (out / 'latest.pt').exists():
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        policy.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']
    if cuda:  # measured faster on A100s (2026-10-14)
        for conv in policy.convs:
            conv.to(memory_format=torch.channels_last)
        policy.convs[0].register_forward_pre_hook(lambda m, a: (a[0].contiguous(memory_format=torch.channels_last),))
        policy.forward = torch.compile(policy.forward, dynamic=True)
    objective = Objective(policy, settings['value_coef'], settings['next_play'])
    if world > 1:
        objective = nn.parallel.DistributedDataParallel(objective, device_ids=[device.index] if cuda else None,
                                                        find_unused_parameters=True)
    autocast = (lambda: torch.autocast('cuda', torch.bfloat16)) if cuda else nullcontext
    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None

    def write(row):
        log.write(json.dumps(row) + '\n')
        log.flush()
        print(json.dumps(row), flush=True)

    def checkpoint():
        if rank == 0:
            tmp = out / 'latest.tmp'
            torch.save(dict(config=asdict(config), model=policy.state_dict(), optimizer=optimizer.state_dict(),
                            step=step, settings=settings, vocab=sorted(vocab.index)), tmp)
            tmp.replace(out / 'latest.pt')

    def loader(entries, seed, validation=False, rank=0, world=1):
        return torch.utils.data.DataLoader(Games(data, entries, settings['chunk'], seed, validation, rank, world,
                                                 bool(settings.get('ability_labels'))),
                                           batch_size=batch, num_workers=settings['workers'], collate_fn=collate,
                                           pin_memory=cuda)

    def validate():
        if rank == 0:
            policy.eval()
            with torch.no_grad(), autocast():
                rows = [loss_and_metrics(policy, *({k: v.to(device) for k, v in g.items()} for g in b),
                                         settings['value_coef'], settings['next_play'])[1]
                        for b in loader(val_entries, 0, validation=True)]
            policy.train()
            write(dict(step=step, split='val', **summarize(rows)))
        if world > 1:
            dist.barrier()

    started, window = time.monotonic(), []
    while step < total:
        # a pass over the data; a resumed run starts a fresh order (seeded by the step it resumes at)
        begun = step
        for xs, ts in lockstep(loader(train_entries, settings['seed'] * 1_000_003 + step, rank=rank, world=world),
                               device, world):
            xs, ts = ({k: v.to(device, non_blocking=True) for k, v in g.items()} for g in (xs, ts))
            for group in optimizer.param_groups:
                group['lr'] = settings['lr'] * min(1.0, (step + 1) / max(1, settings['warmup'])) * \
                    (1 - (1 - settings.get('final_lr', 1.0)) * step / total)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss, metrics = objective(xs, ts)
            if not math.isfinite(metrics['loss']):
                raise FloatingPointError(f'nonfinite loss at step {step}; latest.pt is unchanged')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            step += 1
            window.append(metrics)
            if rank == 0 and step % settings['log_every'] == 0:
                write(dict(step=step, split='train', seconds=round(time.monotonic() - started), **summarize(window)))
                window = []
            if step % settings['checkpoint_every'] == 0:
                checkpoint()
            if step >= total:
                break
        if step == begun:  # else this loop would never end
            raise RuntimeError('a pass made no step: some rank has no training data')
        checkpoint()
        validate()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='a converted dataset (with manifest.json)')
    parser.add_argument('--out', type=Path, required=True, help='the run directory')
    for key, value in DEFAULTS.items():
        parser.add_argument('--' + key, type=type(value), default=value)
    args = parser.parse_args()
    settings = {k: getattr(args, k) for k in DEFAULTS}
    args.out.mkdir(parents=True, exist_ok=True)
    saved = args.out / 'settings.json'
    if saved.exists():
        settings = json.loads(saved.read_text())  # a resumed run keeps its settings
    elif int(os.environ.get('RANK', 0)) == 0:
        saved.write_text(json.dumps(settings, indent=1))
    train(args.data, args.out, settings)



if __name__ == '__main__':
    main()
