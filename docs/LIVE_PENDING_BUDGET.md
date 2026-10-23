# Live pending-action budget

## Problem and scope

`play.py` previously tracked only one pending card. An ability tap created no
reservation, so the next decision could choose a card against a balance that
had not yet paid for the ability. Removing the single-card guard alone also
allows two cards to commit the same elixir. The policy's overlap calculation
accounts for one card, not a list of cards and abilities.

This change fixes the host-side live player's bookkeeping and adds opt-in
two-card overlap. It does **not** change PPO rewards, model weights, input
dimensions, replay conversion, or the simulator's acceptance/release rules.
In particular, `cr/engine.py` still has a separate issue: its pending abilities
have cost zero and card reservations are computed only from pending cards.
Training/simulator parity is follow-up work, not a claim of this change.

## Annotated changes

| File | Change and purpose |
| --- | --- |
| `cr/live_budget.py` | Tracks unpaid card/ability taps, reconciles reader acknowledgements, forecasts payments chronologically, and limits concurrent cards. |
| `play.py` | Computes masks from the complete ledger before sampling, rechecks cards before tapping, and records a reservation immediately after a successful host tap. Merges local taps with queue acknowledgements without counting them twice. |
| `cr/policy.py` | Accepts optional live masks **after** the network forward pass. A live timeline decision bypasses the older one-flight affordability estimate, avoiding a second subtraction. Calls without the optional masks retain the previous sampler path. |
| `tests/test_live_budget.py` | Exercises spending, regeneration, cap, hold boundaries, acknowledgements, reconciliation, and concurrency rules. |
| `tests/test_live_masks.py` | Exercises the real sampler and fake-device live loop, including ability then unaffordable Zap, regeneration-funded Zap, and two taps 250 ms apart. |
| `.github/workflows/tests.yml` | Runs the complete existing and new test suite on Linux, where the simulator's `resource` import is supported. |

## Timing and affordability

Use the observed balance from one coherent frame. Pending entries represent
payments **not yet reflected in that balance**. Between payments, add the
existing `observation.regen()` result and cap the balance at 10; subtract each
payment once. Reject a plan if any payment would overdraw the balance.

A card must first be issuable within the measured `HOLD_TICKS`. The prediction
then includes its nominal `DEPLOY_DELAY`. The first issuable time is used:
the client will not hold an already-affordable tap longer to accommodate our
reservations. Abilities are required to be affordable immediately because an
ability hold has not been measured.

For example, at 2x regeneration, with 5.25 observed elixir, a 3-cost card due in
10 ticks, a 1-cost ability due in 5 ticks, and Zap costing 2 due in 22 ticks:

```
5.25 + 22 / 28 - 3 - 1 - 2 = 0.0357...
```

The Zap tap is allowed now. Starting with 5.0 instead is rejected: it would be
short at its predicted payment. Regeneration before an earlier payment at
the 10-elixir cap is leaked, not stored for later. This is not the overly
conservative rule "current elixir minus all reservations must cover Zap now."

## Running and deliberate behavior changes

```
python play.py CHECKPOINT --tables TABLES --ability-costs COST_1 COST_2
python play.py CHECKPOINT --tables TABLES --ability-costs COST_1 COST_2 --overlap
```

Replace `COST_1` and `COST_2` with **verified costs for the current deck/build**,
in controller order. Zero is accepted only when explicitly configured. The
reader/exported tables used here do not expose an authoritative ability-cost
field. Without costs, ability actions are disabled and an `ability_disabled`
event explains why. This is an intentional behavior change: unknown is not free.
Do not reuse these arguments for a different deck without checking them.

Controller 2 remains disabled in the live sampler because this player has no
verified screen coordinate for its button. Previously its choice could tap
controller 1's button. A second cost argument does not enable the second button.

Card overlap is off by default. With `--overlap`, at most two different actual
hand slots may be outstanding. A card still held for elixir blocks another card
tap until issued. Cards in the cycle are not promoted into an empty hand slot.
The network still explicitly encodes only its first own pending card; the host
ledger checks all pending costs. This is experimental, not a trained multi-card
queue representation or a claim of improved strength.

An empty hand slot or missing ACK does not release a reservation. A coherent
replacement-card frame with no future ACK confirms the card payment; ability
button/charge changes confirm its resolution. These confirmations share the
frame's updated elixir, avoiding a second deduction. An unexplained timeout
stops the live player rather than silently freeing uncertain elixir. An already
visible own queue entry is adopted once; an untracked variable/unknown-cost
card stops with an explanation instead of guessing a cost.

## Verification boundaries

Unit and fake-device loop tests cannot establish actual reader/payment ordering
or touch acceptance on a device. The forecast uses the existing nominal 22-tick
delay until a card's queue ACK supplies its actual due tick. Live jitter and
the ability button/cost-deduction ordering require a recorded native test.
Do not describe this patch as universally regression-free or live-validated.

Before enabling overlap in normal play, record:

1. Ability then Zap with insufficient funds: no Zap tap; after the deduction,
   the updated balance is not charged for the ability twice.
2. The same sequence when regeneration funds Zap: the tap is allowed before
   payment, within the client's real hold/command timing.
3. Two funded cards 250 ms apart, an unfunded pair, an empty slot, and a held card.
4. Queue ACK delay, cancellation/missed touch, and uncertain timeout handling.
5. A serial-mode battle and existing checkpoint warm-up for comparison.
