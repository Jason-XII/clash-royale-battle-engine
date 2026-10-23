"""Host-side live reservations; no new checkpoint features or changes to training defaults.

Costs are in elixir, ticks at 20 Hz. A reservation is an unpaid action, not an
extra debit on a balance which already includes its execution. Reconcile a
coherent reader frame before asking for the next action.
"""
from dataclasses import dataclass
import math

from cr import observation as C


@dataclass
class Pending:
    kind: str
    cost: float
    tap: int
    due: int
    slot: int
    card: int | None = None
    x: int = 0
    y: int = 0
    charges: int = -1
    acknowledged: bool = False


def balance_at(elixir, tick, events, until=None):
    """Walk payment times, with regeneration counted once and capped at 10.

    Return None if an obligation cannot be paid. Events at the same tick are
    deliberately charged together; iteration order cannot manufacture elixir.
    """
    at, balance = tick, float(elixir)
    for due, cost in sorted(events):
        due = max(tick, due)
        if until is not None and due > until:
            break
        balance = min(10.0, balance + C.regen(at, due) / 10000) - cost
        if balance < -1e-6:
            return None
        at = due
    if until is not None:
        balance = min(10.0, balance + C.regen(at, until) / 10000)
    return balance


def payment_tick(elixir, tick, pending, cost, *, ability=False):
    """Predict a new tap's payment, or None if it would overspend.

    First find when the CLIENT can issue it (cards may be held for HOLD_TICKS;
    no such hold is assumed for abilities). Only then add deployment/command
    delay and check all outstanding payments. In particular, do not invent a
    longer client hold just because some of the visible bar is reserved.
    """
    if not math.isfinite(cost) or not 0 <= cost <= 10:
        raise ValueError('action cost must be finite and between 0 and 10')
    events = [(max(tick + 1, p.due), p.cost) for p in pending]
    for wait in range(1 if ability else C.HOLD_TICKS + 1):
        available = balance_at(elixir, tick, events, tick + wait)
        if available is None:
            return None
        if available + 1e-6 < cost:
            continue
        due = tick + wait + C.DEPLOY_DELAY
        # This is the first time the client issues the tap. If its subsequent
        # payment conflicts, defer the TAP; it will not hold an affordable tap.
        return due if balance_at(elixir, tick, events + [(due, cost)]) is not None else None
    return None


class LiveBudget:
    """At most two own cards, plus an ability; local taps survive reader ACK lag."""

    def __init__(self, owner, *, overlap=False, ability_costs=(None, None)):
        if len(ability_costs) != 2 or any(c is not None and (not math.isfinite(c) or not 0 <= c <= 10)
                                          for c in ability_costs):
            raise ValueError('provide two ability costs, each between 0 and 10 (or None if unknown)')
        self.owner, self.overlap, self.ability_costs = owner, overlap, ability_costs
        self.pending = []

    def update(self, frame):
        if frame['failure'] != 'none' or not frame.get('coherent', True) or not frame.get('queue_ok'):
            return
        tick = frame['game_tick']
        me = next(p for p in frame['players'] if p['side'] == self.owner)
        hand = {i: me['deck_card_ids'][s] for i, s in enumerate(me['hand_deck_indices']) if s >= 0}
        commands = [c for c in frame.get('commands', ()) if c['owner'] == self.owner]
        alive = []
        for p in self.pending:
            if p.kind == 'card':
                ack = next((c for c in commands if c['card_id'] == p.card and c['issue_tick'] >= p.tap
                            and c['execute_tick'] > tick), None)
                if ack:
                    p.due = ack['execute_tick']
                    p.acknowledged = True
                # An empty hand slot is not confirmation; the card can still
                # be queued. A replacement and no future ACK confirms spending
                # in this same coherent frame, whose elixir now includes it.
                paid = not ack and p.slot in hand and hand[p.slot] != p.card
            else:
                button = next((a for a in me.get('abilities', ()) if a['controller_slot'] - 1 == p.slot), None)
                paid = button is not None and (button['button'] not in (2, 4)
                                               or button.get('charges', -1) != p.charges)
                ack = None
            if paid:
                continue
            if not ack and tick > p.due + C.HOLD_TICKS + 10:
                # A missing receipt is not proof the action was free. Stop
                # rather than silently spend an uncertain balance twice.
                raise RuntimeError(f'unconfirmed {p.kind} tap at tick {p.tap}; stop and inspect the reader log')
            alive.append(p)
        self.pending = alive
        # Include already-visible own commands (for example, attaching after a
        # human tap). Local tap + reader ACK is one obligation, never two.
        for c in commands:
            card = c.get('card_id', 0)
            if c['execute_tick'] <= tick or card <= 0 or any(p.card == card for p in self.pending):
                continue
            cost = C.COSTS.get(card)
            if cost is None or card == C.MIRROR:
                raise RuntimeError('untracked variable/unknown-cost card in queue; restart before the next battle')
            slot = next((s for s, cid in hand.items() if cid == card), -1)
            self.pending.append(Pending('card', cost, c['issue_tick'], c['execute_tick'], slot, card,
                                        c.get('x', 0), c.get('y', 0), acknowledged=True))

    def card_due(self, elixir, tick, slot, card, cost):
        cards = [p for p in self.pending if p.kind == 'card']
        if len(cards) >= (2 if self.overlap else 1) or any(p.slot == slot or p.card == card for p in cards):
            return None
        if any(not p.acknowledged and tick < p.due - C.DEPLOY_DELAY for p in cards):
            return None  # do not overwrite a card the client is still holding for elixir
        return payment_tick(elixir, tick, self.pending, cost)

    def ability_due(self, elixir, tick, controller):
        cost = self.ability_costs[controller]
        # Preserve the policy's existing card -> ability block. A pending
        # ability, however, need not block a sufficiently funded card.
        if cost is None or self.pending:
            return None
        return payment_tick(elixir, tick, self.pending, cost, ability=True)

    def commands(self):
        return [(p.due, dict(owner=self.owner, card_id=p.card, x=p.x, y=p.y, tap=p.tap))
                for p in self.pending if p.kind == 'card']
