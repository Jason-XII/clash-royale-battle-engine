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


def worker(index, clapha, work, seconds, vocab_path, booting, start, out):
    log = os.open(f'runs/throughput-logs/worker-{index}.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)  # the engine's own chatter goes to the worker's log
    os.dup2(log, 2)
    from native_engine import Play
    from native_engine.clapha import ClaphaEngine
    try:
        with booting:  # many engines booting at once can crash one (seen at 16 on a busy node)
            engine = ClaphaEngine(clapha)
    except Exception as error:
        out.put(('failed', repr(error)))
        out.close()
        out.join_thread()
        os._exit(1)
    out.put(('ready', None))
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
    out.put(('done', (games, decisions, ticks)))
    out.close()
    out.join_thread()
    os._exit(0)  # the engine segfaults in its exit-time destructors


def run(args, procs):
    ctx = mp.get_context('fork')
    booting, start, out = ctx.Semaphore(4), ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=worker, args=(i, args.clapha, args.work, args.seconds, args.vocab, booting, start, out))
               for i in range(procs)]
    for w in workers:
        w.start()
    booted = [out.get() for _ in workers]
    failed = [why for kind, why in booted if kind == 'failed']
    running = procs - len(failed)
    start.set()
    results = [out.get()[1] for _ in range(running)]
    for w in workers:
        w.join()
    games, decisions, ticks = map(sum, zip(*results))
    s = args.seconds
    if failed:
        print(f'    {len(failed)} of {procs} engines failed to boot (logs in runs/throughput-logs/): {failed[0]}')
    procs = running
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
    print(f'cpus visible to this process: {len(os.sched_getaffinity(0))}', flush=True)
    os.makedirs('runs/throughput-logs', exist_ok=True)
    for procs in map(int, args.procs.split(',')):
        run(args, procs)
