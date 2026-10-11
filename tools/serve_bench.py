"""Where a decision server's time goes on this machine: one Server.act call split into input preparation,
GPU forward and sampling + copy back, at several batch sizes. Safe to run next to training (a few seconds of GPU).

    PYTHONPATH=. python tools/serve_bench.py ~/cr-engine/runs/il-002/best.pt ~/cr-engine/data/cache/part-000000-0.npz
"""
import sys
import time

import numpy as np
import torch

from il import data as D
from il.model import INPUT_KEYS
from il.rl import Server, distribution, load_policy, log_prob, sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)
model, saved = load_policy(sys.argv[1], device)
model.eval()
seq = D.sequence(D.Shard(sys.argv[2])[0], 0, saved['vocab'])
sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
print('device', device, torch.cuda.get_device_name() if device.type == 'cuda' else '')
for batch in (1, 8, 32, 64):
    rows = [{k: seq[k][i % 100] for k in INPUT_KEYS} for i in range(batch)]
    server = Server({'current': model}, device, model.config['core'])
    requests = [((i, 0), 'current', rows[i]) for i in range(batch)]
    for _ in range(10):
        server.act(requests)
    sync()
    parts = np.zeros(3)
    calls = 50
    for _ in range(calls):
        t0 = time.perf_counter()
        x = {k: torch.from_numpy(np.stack([r[k] for r in rows]))[:, None].to(device, non_blocking=True) for k in INPUT_KEYS}
        state = model.initial_state(batch, device)
        sync()
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
            logits, value, _ = model(x, state)
        sync()
        t2 = time.perf_counter()
        with torch.no_grad():
            dist = distribution(logits[:, 0])
            action = sample(dist)
            torch.stack([action.float(), log_prob(dist, action), value[:, 0].float()], 1).cpu().tolist()
        t3 = time.perf_counter()
        parts += (t1 - t0, t2 - t1, t3 - t2)
    t = time.perf_counter()
    for _ in range(calls):
        server.act(requests)
    sync()
    whole = (time.perf_counter() - t) / calls * 1000
    prep, forward, post = parts / calls * 1000
    print(f'batch {batch:2d}: whole Server.act {whole:6.2f} ms | inputs {prep:5.2f}  forward {forward:5.2f}  '
          f'sample+copy {post:5.2f} ms | {whole / batch:5.2f} ms per decision')
