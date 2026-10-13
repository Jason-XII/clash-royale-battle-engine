"""Where a self-play decision's CPU time goes (run it in a job, not on the login node): one game of random plays in
ClaphaEngine, timing engine.step (5 ticks and its State), record() and both players' clapha-il inputs (il.policy.Inputs).

    PYTHONPATH=. python tools/time_decision.py ~/clapha-engine ~/il-runs/pilot-base/latest.pt [--play-rate 0.05]
"""
import argparse
import os
import random
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clapha')
    ap.add_argument('checkpoint', help='a clapha-il checkpoint (its vocabulary)')
    ap.add_argument('--play-rate', type=float, default=0.05, help='chance of a random play per player and decision')
    ap.add_argument('--deck', default='26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026')
    args = ap.parse_args()
    from native_engine import Play
    from native_engine.clapha import ClaphaEngine
    from il import policy
    from il.arena import make_match, mirror
    _, vocab, _ = policy.load(args.checkpoint)
    engine = ClaphaEngine(args.clapha)
    match = make_match(mirror(args.deck), 5)
    state = engine.reset(match=match)
    inputs = policy.Inputs(engine, match, {0: vocab, 1: vocab}, policy.legal_fn())
    inputs(state)  # builds the legal-tile tables a long-lived collector keeps
    rng, seconds, n, objects = random.Random(1), {'step': 0.0, 'record': 0.0, 'inputs': 0.0}, 0, 0
    while not state.finalized and state.tick < 7200:
        t0 = time.perf_counter()
        record, issued = engine.record(), inputs.commands.update(state, engine)
        t1 = time.perf_counter()
        for view in inputs.views.values():  # what Inputs does with the record, timed apart from reading it
            policy.pack(view.step(record, issued))
        t2 = time.perf_counter()
        plays = [Play(o, rng.randrange(4), rng.randrange(1, 18) * 1000, rng.randrange(1, 15) * 1000 + (16000 if o else 0))
                 for o in (0, 1) if rng.random() < args.play_rate]
        state = engine.step(plays, 5, delay=policy.PLAY_DELAY).state
        t3 = time.perf_counter()
        seconds['record'] += t1 - t0
        seconds['inputs'] += t2 - t1
        seconds['step'] += t3 - t2
        n += 1
        objects += len(record['objects'])
    print(f'{n} decisions, {objects / n:.1f} objects a decision; per game: '
          + ', '.join(f'{k} {v:.2f} s' for k, v in seconds.items()) + f'; total {sum(seconds.values()):.2f} CPU-s')
    sys.stdout.flush()
    os._exit(0)  # the engine segfaults in its exit-time destructors


if __name__ == '__main__':
    main()
