# Model inputs: what the model should see, and what it does not see yet

Written with Arron (2026-10-11) for this repo's model (il/data.py features, il/model.py). It lists what a player
uses that the model's input lacks, why it matters, and where each piece can come from. It is a checklist, not a
design: how to encode each item is open.

Where each missing piece comes from:
- **(A) available now**: clapha-engine's reader (`crx`) already reads it; only il/data.py / clapha.py need to use it.
- **(B) engine work**: needs new reads in clapha-engine (private). FirstLight's probe, vendored here under
  vendor/firstlight/probe, documents most of these states; they are being ported and checked card by card. This file
  says when one becomes available.
- **(C) decision**: a choice about what the model may know (e.g. what a human player cannot see).

Status words: **missing** (not in the features), **partial**, **has**. "Checked" means verified on the engine; anything
not marked checked is from reading the code.

## 1. Unit state

**2026-10-11:** clapha-engine now reads every (B) item below on both builds (`Reader.unit_states()`, documented and
checked card by card in its docs/engine/UNIT_STATES.md), so they are (A): available to add to the features. Jason's
e8f483b added in-flight plays and presses, ability state, evolution per hand slot, the next three cycle cards and an
opponent elixir estimate (zero-initialised: old checkpoints play as before).

| What | Why | Now | Source |
|---|---|---|---|
| Attack phase (windup / hit / cooldown) and time into it | when the next hit lands; whether a stun resets it | partial (reload timer only) | (A) clapha-engine `Reader.unit_states()` `attack`: stage, timeline, hit count, cycle (time to the next hit), load time; checked per card |
| Which unit it targets (and where) | pulls, kiting, who dies next | partial (has a target: yes/no) | (A) exact target id and target position |
| Run-up charge (Prince, Dark Prince, Battle Ram, Ram Rider) | double damage; stun resets it | missing | (A) clapha-engine `Reader.unit_states()` `charge`; checked (Prince, ...) |
| Damage ramp on one target (Inferno Tower / Dragon, Mighty Miner) | when to reset it | missing | (A) clapha-engine `Reader.unit_states()` `attack.hitCount` / stage (Inferno: checked); see UNIT_STATES.md for the cards still open |
| Charge-up before one big shot (Sparky, …) | the stun / Zap window | missing | (A) clapha-engine `Reader.unit_states()` `attack.loadMs` (Sparky's charge-up) |
| Dashes, jumps, hooks (Bandit, Boss Bandit, Mega Knight, Fisherman, Golden Knight, …) | invulnerable during a dash, landing damage | missing | (A) clapha-engine `Reader.unit_states()` `immune` (Bandit / Boss Bandit dash), state `dashing` (Mega Knight jump: not immune), `attack.dashCooldownMs`; Golden Knight hop index open |
| Frozen / slowed / raged / stunned / poisoned, time left | value of every trade | missing | (A) clapha-engine `Reader.unit_states()` `buffs`: every effect on the unit with name, time left, time so far, speed / hit speed %, who applied it |
| **Can't act**: frozen, held by vines, netted, knocked into the air (by an ability), and for how long | whether it can attack / move now | missing | (A) clapha-engine `Reader.unit_states()` `buffs[].stopsMovement / stopsAttacks` with time left, `airborne` (knock-up), `pushed` + `pushRemainingMs` (knockback); E-Wiz / Lightning / E-Spirit stuns, Log / Snowball pushes: still open |
| **Invisible** (Royal Ghost, Archer Queen ability): can't be targeted, **can** take area damage | spells still hit it | missing | (A) clapha-engine `Reader.unit_states()` `invisible` (checked: Poison still hurts an invisible Royal Ghost) |
| **Underground** (Miner / Drill / Mighty Miner travelling, Tesla hidden): can't be targeted, **can't** be hurt | nothing hits it | missing | (A) clapha-engine `Reader.unit_states()` `hidden` (the engine's own check; Miner / Drill / Tesla checked: no damage) |
| Evolved or not, hero or not (per unit) | they behave differently | missing | (A) the unit's form id |
| Hero / champion ability (英雄技能资讯): **whether it is in its ability now, how long the ability has lasted so far, whether the ability has been used already, Boss Bandit's number of uses**; plus cooldown and charges | ability timing; what is left of a hero | **own: has (e8f483b)**: state of your hero / champion ability; per unit: missing | (A) clapha-engine `Reader.unit_states()` `ability`: in ability (phase), time into it, used, use count (Boss Bandit), cooldown, charges |
| Spell areas with a duration (Graveyard, Poison, Freeze, Tornado, Earthquake, …): time elapsed / left | how many more Skeletons, how long the freeze lasts | missing (position only) | (A) clapha-engine `Reader.unit_states()` areas: `remainingMs`, `durationMs` |
| "Just happened": died, spawned, got hit, spell landed | events are part of the moment's state | missing | (A) derivable from the change between two decisions |
| **Projectiles in flight**, including after their owner died (e.g. the Executioner's axe): direction, speed, where they will go | not placing a card into their path | partial (position + "projectile" only) | (A) clapha-engine `Reader.unit_states()` projectiles: shooter, aim point, homing target, launch point, movement per tick (the Executioner's axe return checked) |
| **What units leave behind** (e.g. Giant Skeleton's bomb): time until it goes off | placement and retreat | partial (a unit, no timer) | (A) clapha-engine `Reader.unit_states()` `leavesOnDeath` per unit; death bombs as kind `timed` with `fuseMs`; spawners `spawnTimerMs` |

Card by card: charging and the other special states differ per card (Bandit and Boss Bandit dash, Sparky, Inferno,
Mother Witch, Monk, …). The list per card, with how each is read and whether it is checked, is kept with the engine
work in clapha-engine; this file is updated as entries become available.

## 2. Plays in flight

What happens in the game, within about a second: **tap → 3 cards in hand, elixir spent, troop pending → troop lands →
the next card fills the empty slot (4 cards)**. Two plays close together: the refill waits for a timer
(docs/GAME_INTEGRATION.md).

| What | Now | Source |
|---|---|---|
| Own pending play: card, where, time until it lands | **has (e8f483b)**: plays and ability presses in flight |  |
| Hand as the screen shows it: card out and elixir spent at the tap; the refilled slot can't be played until dealt | missing (the model sees the game's hand, where the card stays until it lands) | (A) rules in docs/GAME_INTEGRATION.md |
| Enemy pending play | missing | (C) a human can't see it; the live client's command queue holds it 2-21 ticks before it lands (median 13) |

## 3. Own cards

| What | Now | Source |
|---|---|---|
| 4 cards in hand, next card | has | |
| **Whole cycle order**: next, next-next, next-next-next, next-next-next-next | **has (e8f483b)**: the next three cycle cards (with the next card: all four) |  |
| Which slots are the active **evo / hero / wild** (at most 1 evo, 1 hero, 1 wild) | partial (e8f483b: evolution per hand slot) | (A) per-slot form flags (the deck pool has them); use the active slots, not "this card can evolve" |
| Cycles until each evo is ready; ready | **has (e8f483b)**: evolution progress per hand slot |  |
| Hero card vs normal card | missing | (A) |

## 4. Opponent cards

| What | Now | Source |
|---|---|---|
| Cards seen | partial (last 8 different, with plays ago) | |
| **Card counting**: the order they were played, where each card is now (hand / cycle), plays until it is back | missing | (A) derived from their plays (the cycle is deterministic) |
| Opponent active evo / hero: which, cycles until evo | missing | (A) once seen; (C) at battle start the engine knows their slots. Live: heroes and champions are readable from the start, evolutions only once played (docs/GAME_INTEGRATION.md section 9) |
| A guessed deck (RoyaleAPI, live only; can be wrong, has no order) | missing | (C) e.g. "probably in their deck" flags with a confidence, kept apart from "seen", replaced by what the game shows |
| A given deck (friendly / war) | missing | (C) |

## 5. Memory

What can be **counted** (cycles, cards seen, elixir spent) should be an input, like a player's notes; the LSTM is then
left for what can't be counted (this opponent's habits).

## 6. Playing abilities

The model cannot play hero or champion abilities now: IL learned one "ability" action from human presses (when, not
which), PPO masks it out (il/rl.py), and il/live.py skips it. Needed: the engine taking ability commands (clapha-engine
crx has `queue_ability_action_at`, not verified yet: engine work) and an ability action per hero / champion in the
model's output.

## 7. Timing

- A decision every 5 ticks (0.25 s), like FirstLight. Deciding every 2-3 ticks would cost roughly 40-55% of games per
  hour (estimate). Cheaper precision: choose a small delay with each play (FirstLight: 0 / 50 / 100 / 150 / 200 ms).
- One play per decision (FirstLight allows two; Arron prefers one).
