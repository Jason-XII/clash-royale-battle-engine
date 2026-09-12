"""Actor views for the non-evolved standard-deck prototype only."""
from dataclasses import dataclass

from .protocol import CardPlay, Entity, State


@dataclass(frozen=True)
class Observation:
    tick: int
    owner: int
    elixir: float
    hand: tuple[tuple[int, int, int], ...]  # slot, card, cost
    cycle: tuple[int, ...]
    crowns: tuple[int, int]
    entities: tuple[Entity, ...]
    opponent_revealed: tuple[int, ...]
    plays: tuple[CardPlay, ...]
    finalized: bool


class Observer:
    def __init__(self, owner):
        self.owner = owner
        self.generation = None
        self.revealed = set()

    def observe(self, state: State):
        if self.generation != state.generation:
            self.generation = state.generation
            self.revealed.clear()
        self.revealed.update(e.card for e in state.plays if e.owner != self.owner)
        own = state.players[self.owner]
        return Observation(state.tick, self.owner, own.elixir / 10000,
            tuple((c.slot, c.card, c.cost) for c in own.hand), own.cycle, state.crowns,
            state.entities, tuple(sorted(self.revealed)), state.plays, state.finalized)
