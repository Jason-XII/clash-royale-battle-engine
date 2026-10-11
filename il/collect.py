"""Collectors in another job: they find the learner through <run>/endpoint.json (written by il.rl --remote on a
shared file system), connect over the cluster network and play its games. When the learner stops or restarts they
wait and reconnect, re-reading the file; the learner simply loses them when this job ends.

    python -m il.collect ~/cr-engine/runs/ppo-general --collectors 30        # or: sbatch tools/collectors.sbatch
"""
import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import Client
import os
from pathlib import Path
import socket
import time

from .rl import collector_engine, play_games


def endpoint(run):
    """The learner's current address and key, waiting until a learner has written them."""
    path = run / 'endpoint.json'
    while True:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(10)


def remote(slot, run):
    from native_engine.clapha import raise_thread_limit
    raise_thread_limit()
    first = endpoint(run)
    name = f'{socket.gethostname()}-{os.getppid()}-{slot}'
    engine, vocab = collector_engine(name, first['out'], first['clapha'], slot // 4 * 8)
    while True:
        learner = server = None
        try:
            ep = endpoint(run)
            key = bytes.fromhex(ep['authkey'])
            learner = Client(tuple(ep['learner']), authkey=key)
            learner.send(('hello', None, None))
            hello = learner.recv()
            server = Client(tuple(ep['servers'][hello['server']]), authkey=key)
            print(f'connected to {ep["host"]} as collector {hello["index"]}', flush=True)
            play_games(hello['index'], ep['delay'], engine, vocab, learner, server)
            print('the learner stopped; waiting for the next one', flush=True)
        except (OSError, EOFError) as error:  # learner gone, restarting, or not up yet
            print(f'not connected ({error!r}); retrying', flush=True)
        finally:
            for c in (learner, server):
                if c is not None:
                    c.close()
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help='the il.rl --remote output directory (on the shared file system)')
    parser.add_argument('--collectors', type=int, default=30)
    args = parser.parse_args()
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


if __name__ == '__main__':
    main()
