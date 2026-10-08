from dataclasses import dataclass


@dataclass(frozen=True)
class HandCard:
    slot: int
    card: int
    cost: int
    parameter: int  # Native command encoding; never a policy feature.
    form_card: int  # Card id of the form that would be played (evolution/hero), else == card.


@dataclass(frozen=True)
class Player:
    owner: int
    elixir: int  # Fixed-point native units: 10,000 = one elixir.
    hand: tuple[HandCard, ...]
    cycle: tuple[int, ...]


@dataclass(frozen=True)
class Entity:
    id: int
    owner: int
    card: int  # -1 for towers.
    x: int
    y: int
    hp: int | None
    max_hp: int | None
    target_id: int | None  # Stable entity ID held by the native attack component.
    attack_cooldown_ms: int | None  # Zero means ready; rounded up to a 50 ms tick.
    entity_type: str  # troop, building, projectile or area
    deployment_remaining_ms: int | None


@dataclass(frozen=True)
class CardPlay:
    sequence: int
    tick: int
    owner: int
    card: int
    cost: int


@dataclass(frozen=True)
class State:
    generation: int
    tick: int
    finalized: bool
    result: int | None
    crowns: tuple[int, int]
    players: tuple[Player, ...]
    entities: tuple[Entity, ...]
    plays: tuple[CardPlay, ...]
    next_sequence: int

    @classmethod
    def decode(cls, data):
        players = tuple(Player(p['owner'], p['elixir'], tuple(HandCard(*c) for c in p['hand']),
                               tuple(p['cycle'])) for p in data['players'])
        entities = tuple(Entity(*e) for e in data['entities'])
        return cls(data['generation'], data['tick'], data['finalized'], data['result'],
                   tuple(data['crowns']), players, entities,
                   tuple(CardPlay(*e) for e in data['plays']), data['next'])
