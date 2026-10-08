"""Measure resident-batch throughput over complete, validated episodes.

Run: python -m native_engine.batch_benchmark
"""
import random
import sys
import time
from dataclasses import dataclass
from statistics import median

from . import Batch
from .benchmark import random_play

BATTLES = 16
TICKS = 10
REPEATS = 5


@dataclass(frozen=True, slots=True)
class Sample:
    ticks_per_second: float
    round_trip_seconds: float
    episodes_per_second: float
    attempts: int
    queued: int
    consumed: int


def actions_for(randomizer, state):
    plays = (random_play(randomizer, player) for player in state.players)
    return tuple(play for play in plays if play)


def measure(batch, seed, ticks=TICKS):
    randomizer = random.Random(seed)
    states = batch.reset(seed=seed)
    advanced = attempts = queued = consumed = rounds = 0
    started = time.perf_counter()
    while not all(state.finalized for state in states):
        actions = [actions_for(randomizer, state) if not state.finalized else () for state in states]
        previous = states
        states = batch.step(actions, ticks=ticks)
        deltas = tuple(after.tick - before.tick for before, after in zip(previous, states))
        stalled = [index for index, (before, after, delta) in
                   enumerate(zip(previous, states, deltas))
                   if not before.finalized and not after.finalized and delta <= 0]
        if stalled:
            raise RuntimeError(f'active battles did not advance: {stalled}')
        advanced += sum(deltas)
        attempts += sum(map(len, actions))
        queued += sum(receipt.status == 0 for receipts in batch.last_receipts for receipt in receipts)
        consumed += sum(len(state.plays) for state in states)
        rounds += 1
    seconds = time.perf_counter() - started
    return Sample(advanced / seconds, seconds / rounds, batch.battles / seconds,
                  attempts, queued, consumed)


def benchmark(battles=BATTLES, repeats=REPEATS, port=26789):
    batch = Batch(battles=battles, port=port)
    try:
        measure(batch, seed=0)  # discard one complete warm-up episode
        return [measure(batch, seed=1 + sample * battles) for sample in range(repeats)]
    finally:
        batch.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 26789
    samples = benchmark(port=port)
    rates = [sample.ticks_per_second for sample in samples]
    attempts = sum(sample.attempts for sample in samples)
    queued = sum(sample.queued for sample in samples)
    consumed = sum(sample.consumed for sample in samples)
    print(f'battles={BATTLES} ticks={TICKS} repeats={REPEATS} complete_episodes=true')
    print(f'median={median(rates):,.0f} ticks/s  samples={",".join(f"{rate:.0f}" for rate in rates)}')
    print(f'median_episodes={median(sample.episodes_per_second for sample in samples):.2f}/s  '
          f'median_round_trip={median(sample.round_trip_seconds for sample in samples) * 1000:.2f} ms')
    print(f'actions attempted={attempts} queued={queued} consumed={consumed}')


if __name__ == '__main__':
    main()
