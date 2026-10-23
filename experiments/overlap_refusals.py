"""Why the simulator refuses an overlapping player's card taps (23 Oct): arena.py with every refused card tap of a
player with Engine.overlap written to REFUSALS (one JSON line: tick, owner, the reason, the pending plays). Reasons, in
this order: no card in the slot; the slot already in flight; a pending card still held (short of elixir); two or more
cards in flight; elixir short once the pending costs are reserved.

    python experiments/overlap_refusals.py REFUSALS.jsonl A.pt B.pt --a-overlap ...arena.py's arguments...
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arena  # noqa: E402
from cr import engine  # noqa: E402

out = open(sys.argv.pop(1), 'a')
step = engine.Engine.step


def logged(self, actions=(), ticks=5, delay=0):
    pending, before = [dict(p) for p in self.pending], self.state
    result = step(self, actions, ticks, delay)
    for a, ok in zip(actions, result.executed):
        if ok or a.slot == engine.ABILITY or not self.overlap[a.owner]:
            continue
        mine = [p for p in pending if p['owner'] == a.owner and p['slot'] != engine.ABILITY]
        card = next((h for h in before.players[a.owner].hand if h.slot == a.slot), None)
        why = ('no card' if card is None else 'slot in flight' if any(p['slot'] == a.slot for p in mine) else
               'pending held' if any(p['held'] for p in mine) else 'two in flight' if len(mine) >= 2 else 'elixir')
        out.write(json.dumps(dict(tick=before.tick, owner=a.owner, why=why, pending=len(mine), elixir=before.players[a.owner].elixir,
                                  cost=card and card.cost, reserved=sum(p['cost'] for p in mine))) + '\n')
        out.flush()
    return result


if __name__ == '__main__':
    engine.Engine.step = logged
    arena.main()
