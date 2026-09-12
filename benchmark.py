"""Benchmark one battle with random actions every ten simulation ticks."""
import random
from time import perf_counter

from . import Engine, Play


def random_play(randomizer, player):
    affordable = [card for card in player.hand if card.cost * 10000 <= player.elixir]
    if not affordable:
        return None
    card = randomizer.choice(affordable)
    x = randomizer.randint(1000, 17000)
    if card.card >= 28000000:
        y = randomizer.randint(3000, 29000)
    else:
        y = randomizer.randint(7000, 14500) if player.owner == 0 else randomizer.randint(17500, 25000)
    return Play(player.owner, card.slot, x, y)


def main():
    randomizer = random.Random(1)
    engine = Engine()
    started = perf_counter()
    state = engine.reset(seed=1)
    reset_seconds = perf_counter() - started
    attempts = executions = 0
    rollout_started = perf_counter()
    while not state.finalized:
        actions = tuple(filter(None, (random_play(randomizer, player) for player in state.players)))
        transition = engine.step(actions, ticks=min(10, 7200 - state.tick))
        attempts += len(actions)
        executions += sum(transition.executed)
        state = transition.state
    rollout_seconds = perf_counter() - rollout_started
    simulated_seconds = (state.tick - 90) * 0.05
    print(f"result={state.result} crowns={state.crowns} final_tick={state.tick}")
    print(f"attempts={attempts} executions={executions} wire_bytes={engine.wire_bytes}")
    print(f"reset={reset_seconds:.3f}s rollout={rollout_seconds:.3f}s total={perf_counter() - started:.3f}s")
    print(f"simulated={simulated_seconds:.1f}s speed={simulated_seconds / rollout_seconds:.1f}x real-time")
    print(f"ticks_per_second={(state.tick - 90) / rollout_seconds:.1f}")


if __name__ == '__main__':
    main()
