from dataclasses import dataclass


_BASIC_TRAITS = {
    -1: ("projectile", None, "air_and_ground"),
    26000000: ("melee", "ground", "ground"),
    26000001: ("projectile", "ground", "air_and_ground"),
    26000003: ("melee", "ground", "buildings"),
    26000005: ("melee", "air", "air_and_ground"),
    26000014: ("projectile", "ground", "air_and_ground"),
    26000018: ("melee", "ground", "ground"),
}

_BASIC_SPELLS = {28000000, 28000001}


@dataclass(frozen=True)
class HandCard:
    slot: int
    card: int
    cost: int
    parameter: int  # Native command encoding; never a policy feature.


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
    card: int
    x: int
    y: int
    hp: int | None
    max_hp: int | None
    target_id: int | None  # Stable entity ID held by the native attack component.
    attack_cooldown_ms: int | None  # Zero means ready; rounded up to a 50 ms tick.
    entity_type: str
    deployment_remaining_ms: int | None
    attack_type: str | None
    movement_type: str | None
    target_type: str | None
    is_spell: bool | None

    @classmethod
    def decode(cls, record):
        values = tuple(record)
        entity_type = values[9]
        if entity_type == "projectile":
            traits = (None, None, None, values[2] in _BASIC_SPELLS)
        else:
            traits = (*_BASIC_TRAITS[values[2]], None)
        return cls(*values, *traits)


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
        entities = tuple(Entity.decode(e) for e in data['entities'])
        return cls(data['generation'], data['tick'], data['finalized'], data['result'],
                   tuple(data['crowns']), players, entities,
                   tuple(CardPlay(*e) for e in data['plays']), data['next'])
