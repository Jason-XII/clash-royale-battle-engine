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

**Update 2026-10-12** (commit e8f483b, Jason's AI): added as inputs (`extras`, `cycle_card`, `pending_card`, `ability_ok`;
il/data.py), zero-initialised so older checkpoints play as before, learned by PPO (the IL cache lacks them): own play or
ability press in flight, own hero / champion controller (ready, casting, charges, cooldown) and the ability as an
action, evolution per hand slot (next play evolved, progress), the next three cycle cards, an opponent elixir estimate
(regeneration measured on the cache: 1 per 2.8 s, x2 from tick 2400, x3 from 4800). The opponent's whole deck feeds the
value head only (training-time information, never a decision). Rows below say "has (2026-10-12)" where this applies.

**Update 2026-10-12** (Arron's PR): clapha-engine reads the (B) states below on both builds (`Reader.unit_states()`, checked
card by card in its docs/engine/UNIT_STATES.md). `--unit-states` puts 19 of them per unit into the `ent_state` input
(names in il/data.py `UNIT_STATE_NAMES`; a zero-initialised layer, so older checkpoints play as before). It costs about
0.4 ms of engine time per decision (measured in x86 emulation: 0.41 -> 0.80 ms per `ClaphaEngine.step`), so it is off
by default. Rows below say "has with --unit-states" where this applies.

Status words: **missing** (not in the features), **partial**, **has**. "Checked" means verified on the engine; anything
not marked checked is from reading the code.

## 1. Unit state

| What | Why | Now | Source |
|---|---|---|---|
| Attack phase (windup / hit / cooldown) and time into it | when the next hit lands; whether a stun resets it | **has with `--unit-states` (`ent_state`)**: fraction of the attack cycle, load time | clapha-engine `Reader.unit_states()` `attack` (checked per card) |
| Which unit it targets (and where) | pulls, kiting, who dies next | partial (has a target: yes/no) | (A) exact target id and target position |
| Run-up charge (Prince, Dark Prince, Battle Ram, Ram Rider) | double damage; stun resets it | **has with `--unit-states` (`ent_state`)** | clapha-engine `Reader.unit_states()` `charge` |
| Damage ramp on one target (Inferno Tower / Dragon, Mighty Miner) | when to reset it | missing | (A) clapha-engine `Reader.unit_states()` `attack.hitCount` / stage: available, not yet an input |
| Charge-up before one big shot (Sparky, …) | the stun / Zap window | **has with `--unit-states` (`ent_state`)** (load time) | clapha-engine `Reader.unit_states()` `attack.loadMs` (Sparky's charge-up) |
| Dashes, jumps, hooks (Bandit, Boss Bandit, Mega Knight, Fisherman, Golden Knight, …) | invulnerable during a dash, landing damage | **has with `--unit-states` (`ent_state`)**: immune, dash cooldown | clapha-engine `Reader.unit_states()` `immune`, `attack.dashCooldownMs`; dash state names not an input yet; Golden Knight hop index open |
| Frozen / slowed / raged / stunned / poisoned, time left | value of every trade | **has with `--unit-states` (`ent_state`)**: speed and hit-speed effects (summed) | clapha-engine `Reader.unit_states()` `buffs` (every effect with time left) |
| **Can't act**: frozen, held by vines, netted, knocked into the air (by an ability), and for how long | whether it can attack / move now | **has with `--unit-states` (`ent_state`)**: can't move / can't attack and the longest such time left, airborne, push time left | clapha-engine `Reader.unit_states()` `buffs[].stopsMovement / stopsAttacks`, `airborne`, `pushed`; E-Wiz / Lightning / E-Spirit stuns, Log / Snowball pushes: open in clapha-engine |
| **Invisible** (Royal Ghost, Archer Queen ability): can't be targeted, **can** take area damage | spells still hit it | **has with `--unit-states` (`ent_state`)** | clapha-engine `Reader.unit_states()` `invisible` (Poison still hurts an invisible Royal Ghost: checked) |
| **Underground** (Miner / Drill / Mighty Miner travelling, Tesla hidden): can't be targeted, **can't** be hurt | nothing hits it | **has with `--unit-states` (`ent_state`)** (hidden) | clapha-engine `Reader.unit_states()` `hidden` (the engine's own check; Miner / Drill / Tesla: checked) |
| Evolved or not, hero or not (per unit) | they behave differently | missing | (A) the unit's form id |
| Hero / champion ability (英雄技能资讯): **whether it is in its ability now, how long the ability has lasted so far, whether the ability has been used already, Boss Bandit's number of uses**; plus cooldown and charges | ability timing; what is left of a hero | partial (2026-10-12): own controller's ready / casting / charges / cooldown | (A) has; (B) per unit: in ability, time into it, used yet, use count |
| Spell areas with a duration (Graveyard, Poison, Freeze, Tornado, Earthquake, …): time elapsed / left | how many more Skeletons, how long the freeze lasts | **has with `--unit-states` (`ent_state`)** (time left) | clapha-engine `Reader.unit_states()` areas `remainingMs` |
| "Just happened": died, spawned, got hit, spell landed | events are part of the moment's state | missing | (A) derivable from the change between two decisions |
| **Projectiles in flight**, including after their owner died (e.g. the Executioner's axe): direction, speed, where they will go | not placing a card into their path | **has with `--unit-states` (`ent_state`)**: where it flies, relative to it, per seat | clapha-engine `Reader.unit_states()` projectiles: aim point, homing target, shooter |
| **What units leave behind** (e.g. Giant Skeleton's bomb): time until it goes off | placement and retreat | **has with `--unit-states` (`ent_state`)**: spawn timer / bomb fuse | clapha-engine `Reader.unit_states()` `spawnTimerMs`, kind `timed` `fuseMs`, `leavesOnDeath` |

Card by card: charging and the other special states differ per card (Bandit and Boss Bandit dash, Sparky, Inferno,
Mother Witch, Monk, …). The list per card, with how each is read and whether it is checked, is kept with the engine
work in clapha-engine; this file is updated as entries become available.

## 2. Plays in flight

What happens in the game, within about a second: **tap → 3 cards in hand, elixir spent, troop pending → troop lands →
the next card fills the empty slot (4 cards)**. Two plays close together: the refill waits for a timer
(docs/GAME_INTEGRATION.md).

| What | Now | Source |
|---|---|---|
| Own pending play: card, where, time until it lands | has (2026-10-12): card, cost, time until it lands, elixir after it; not where | (A) ClaphaEngine.pending; live: il.live's tap |
| Hand as the screen shows it: card out and elixir spent at the tap; the refilled slot can't be played until dealt | missing (the model sees the game's hand, where the card stays until it lands) | (A) rules in docs/GAME_INTEGRATION.md |
| Enemy pending play | missing | (C) a human can't see it; the live client's command queue holds it 2-21 ticks before it lands (median 13) |

## 3. Own cards

| What | Now | Source |
|---|---|---|
| 4 cards in hand, next card | has | |
| **Whole cycle order**: next, next-next, next-next-next, next-next-next-next | has (2026-10-12) | (A) the reader gives the full cycle order (ClaphaEngine._players already reads it) |
| Which slots are the active **evo / hero / wild** (at most 1 evo, 1 hero, 1 wild) | missing | (A) per-slot form flags (the deck pool has them); use the active slots, not "this card can evolve" |
| Cycles until each evo is ready; ready | has (2026-10-12), per hand slot | (A) evolution progress per slot |
| Hero card vs normal card | missing | (A) |

## 4. Opponent cards

| What | Now | Source |
|---|---|---|
| Cards seen | partial (last 8 different, with plays ago) | |
| Elixir (estimate from their plays and the clock) | has (2026-10-12) | (A) |
| **Card counting**: the order they were played, where each card is now (hand / cycle), plays until it is back | missing | (A) derived from their plays (the cycle is deterministic) |
| Opponent active evo / hero: which, cycles until evo | missing | (A) once seen; (C) at battle start the engine knows their slots. Live: heroes and champions are readable from the start, evolutions only once played (docs/GAME_INTEGRATION.md section 9) |
| A guessed deck (RoyaleAPI, live only; can be wrong, has no order) | missing | (C) e.g. "probably in their deck" flags with a confidence, kept apart from "seen", replaced by what the game shows |
| A given deck (friendly / war) | missing | (C) |

## 5. Memory

What can be **counted** (cycles, cards seen, elixir spent) should be an input, like a player's notes; the LSTM is then
left for what can't be counted (this opponent's habits).

## 6. Playing abilities

Has (2026-10-12): one ability action, a play option only while a controller is ready (`ability_ok`); ClaphaEngine
presses it through `queue_ability_action_at` with the live delay; il.live taps the button at (973, 1954). Checked on
clapha: hero Wizard ready -> pressed -> charge spent. Open: decks with two controllers (one action presses the first
ready one), and which unit's ability when several carry one.

## 7. Timing

- A decision every 5 ticks (0.25 s), like FirstLight. Deciding every 2-3 ticks would cost roughly 40-55% of games per
  hour (estimate). Cheaper precision: choose a small delay with each play (FirstLight: 0 / 50 / 100 / 150 / 200 ms).
- One play per decision (FirstLight allows two; Arron prefers one).
