"""Engine throughput across cores: each process boots its own clapha engine and plays games back to back
with a cheap random policy (an affordable card on its own half now and then), for a fixed time.

    PYTHONPATH=. python tools/clapha_throughput.py ~/clapha-engine --procs 1,8,16 --seconds 30

--work state     step 5 ticks per decision and capture the full State (what an agent sees)
       features  + the model inputs of both players (a collector's whole job when the model runs on the GPU)
Stepping alone was measured by tools/clapha_bench.py (~59k ticks/s per core).
"""
import argparse
import multiprocessing as mp
import os
import random
import time


def worker(index, clapha, work, seconds, vocab_path, start, out):
    log = os.open(f'runs/throughput-logs/worker-{index}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)  # the engine's own chatter goes to the worker's log
    os.dup2(log, 2)
    from native_engine import Play
    from native_engine.clapha import ClaphaEngine
    time.sleep(index // 4 * 8)  # boot 4 at a time: 16 at once crashed some (seen on a busy node)
    try:
        engine = ClaphaEngine(clapha)
    except Exception as error:
        out.put(('failed', index, repr(error)))
        out.close()
        out.join_thread()
        os._exit(1)
    out.put(('ready', index, None))
    if work == 'features':
        from il import data as D
        from il.agent import Agent
        vocab = D.load_vocab(vocab_path)
    rng = random.Random(os.getpid())
    start.wait()  # every process starts timing together, after all have booted
    deadline = time.perf_counter() + seconds
    games = decisions = ticks = 0
    while time.perf_counter() < deadline:
        state = engine.reset(seed=rng.randrange(1 << 30))
        if work == 'features':
            agents = [Agent(None, vocab, owner) for owner in (0, 1)]
        first = state.tick
        while not state.finalized and state.tick < 7200 and time.perf_counter() < deadline:
            plays = []
            for p in state.players:
                ok = [h for h in p.hand if h.cost * 10000 <= p.elixir]
                if ok and rng.random() < 0.15:
                    y = rng.randrange(1000, 14000) if p.owner == 0 else rng.randrange(18000, 31000)
                    plays.append(Play(p.owner, rng.choice(ok).slot, rng.randrange(500, 17500), y))
            state = engine.step(plays, 5).state
            if work == 'features':
                for agent in agents:
                    agent.observe(state)
            decisions += 1
        ticks += state.tick - first
        games += state.finalized
    out.put(('done', index, (games, decisions, ticks)))
    out.close()
    out.join_thread()
    os._exit(0)  # the engine segfaults in its exit-time destructors


def collect(out, workers, kind, timeout):
    """Messages of `kind` from {index: process}; a worker that dies or stays silent counts as failed."""
    import queue
    got, failed, pending = {}, [], set(workers)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        try:
            tag, index, value = out.get(timeout=1)
        except queue.Empty:
            for i in [i for i in pending if workers[i].exitcode is not None]:
                pending.discard(i)
                failed.append((i, f'exit {workers[i].exitcode}'))
            continue
        pending.discard(index)
        if tag == kind:
            got[index] = value
        else:
            failed.append((index, value))
    failed += [(i, 'timed out') for i in pending]
    return got, failed


def run(args, procs):
    ctx = mp.get_context('fork')
    start, out = ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=worker, args=(i, args.clapha, args.work, args.seconds, args.vocab, start, out))
               for i in range(procs)]
    for w in workers:
        w.start()
    ready, failed = collect(out, dict(enumerate(workers)), 'ready', timeout=60 + procs // 4 * 8)
    print(f'    {len(ready)} of {procs} engines booted', flush=True)
    start.set()
    results, lost = collect(out, {i: workers[i] for i in ready}, 'done', timeout=args.seconds + 60)
    failed += lost
    for w in workers:
        w.join(timeout=5)
    games, decisions, ticks = map(sum, zip(*results.values())) if results else (0, 0, 0)
    s = args.seconds
    if failed:
        print(f'    {len(failed)} workers failed or died (logs in runs/throughput-logs/): {failed[:3]}')
    procs = max(len(results), 1)
    print(f'{procs:3d} procs [{args.work}]: {games / s:6.2f} games/s ({games / s * 3600:8,.0f}/h)  '
          f'{decisions / s:9,.0f} decisions/s  {ticks / s:11,.0f} ticks/s  '
          f'= {ticks / s / procs:8,.0f} ticks/s per process', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('clapha')
    parser.add_argument('--procs', default='1,8,16')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--work', choices=('state', 'features'), default='state')
    parser.add_argument('--vocab', default='data/cache/vocab.json')
    args = parser.parse_args()
    print(f'cpus visible to this process: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}', flush=True)
    os.makedirs('runs/throughput-logs', exist_ok=True)
    for procs in map(int, args.procs.split(',')):
        run(args, procs)
