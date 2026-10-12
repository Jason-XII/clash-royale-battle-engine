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

| What | Why | Now | Source |
|---|---|---|---|
| Attack phase (windup / hit / cooldown) and time into it | when the next hit lands; whether a stun resets it | partial (reload timer only) | (A) the reader gives attack stage, timeline and load time per unit; their meaning per card to be checked |
| Which unit it targets (and where) | pulls, kiting, who dies next | partial (has a target: yes/no) | (A) exact target id and target position |
| Run-up charge (Prince, Dark Prince, Battle Ram, Ram Rider) | double damage; stun resets it | missing | (A) movement charge progress; to be checked per card |
| Damage ramp on one target (Inferno Tower / Dragon, Mighty Miner) | when to reset it | missing | (B) |
| Charge-up before one big shot (Sparky, …) | the stun / Zap window | missing | (A?) probably the attack timeline; to be checked |
| Dashes, jumps, hooks (Bandit, Boss Bandit, Mega Knight, Fisherman, Golden Knight, …) | invulnerable during a dash, landing damage | missing | (B) |
| Frozen / slowed / raged / stunned / poisoned, time left | value of every trade | missing | (B) |
| **Can't act**: frozen, held by vines, netted, knocked into the air (by an ability), and for how long | whether it can attack / move now | missing | (B) all of these should be one list of effects per unit in the engine (the buff component FirstLight's probe documents); read the list, not each effect separately |
| **Invisible** (Royal Ghost, Archer Queen ability): can't be targeted, **can** take area damage | spells still hit it | missing | (B) |
| **Underground** (Miner / Drill / Mighty Miner travelling, Tesla hidden): can't be targeted, **can't** be hurt | nothing hits it | missing | (B) |
| Evolved or not, hero or not (per unit) | they behave differently | missing | (A) the unit's form id |
| Hero / champion ability (英雄技能资讯): **whether it is in its ability now, how long the ability has lasted so far, whether the ability has been used already, Boss Bandit's number of uses**; plus cooldown and charges | ability timing; what is left of a hero | missing (abilities masked) | (A) own: cooldown, charges, button state; (B) per unit: in ability, time into it, used yet, use count |
| Spell areas with a duration (Graveyard, Poison, Freeze, Tornado, Earthquake, …): time elapsed / left | how many more Skeletons, how long the freeze lasts | missing (position only) | (B) |
| "Just happened": died, spawned, got hit, spell landed | events are part of the moment's state | missing | (A) derivable from the change between two decisions |
| **Projectiles in flight**, including after their owner died (e.g. the Executioner's axe): direction, speed, where they will go | not placing a card into their path | partial (position + "projectile" only) | (B) |
| **What units leave behind** (e.g. Giant Skeleton's bomb): time until it goes off | placement and retreat | partial (a unit, no timer) | (B) |

Card by card: charging and the other special states differ per card (Bandit and Boss Bandit dash, Sparky, Inferno,
Mother Witch, Monk, …). The list per card, with how each is read and whether it is checked, is kept with the engine
work in clapha-engine; this file is updated as entries become available.

## 2. Plays in flight

What happens in the game, within about a second: **tap → 3 cards in hand, elixir spent, troop pending → troop lands →
the next card fills the empty slot (4 cards)**. Two plays close together: the refill waits for a timer
(docs/GAME_INTEGRATION.md).

| What | Now | Source |
|---|---|---|
| Own pending play: card, where, time until it lands | missing (since PR #1 the model cannot play while one is pending, but it cannot see it) | (A) ClaphaEngine.pending |
| Hand as the screen shows it: card out and elixir spent at the tap; the refilled slot can't be played until dealt | missing (the model sees the game's hand, where the card stays until it lands) | (A) rules in docs/GAME_INTEGRATION.md |
| Enemy pending play | missing | (C) a human can't see it; the live client's command queue holds it 2-21 ticks before it lands (median 13) |

## 3. Own cards

| What | Now | Source |
|---|---|---|
| 4 cards in hand, next card | has | |
| **Whole cycle order**: next, next-next, next-next-next, next-next-next-next | missing | (A) the reader gives the full cycle order (ClaphaEngine._players already reads it) |
| Which slots are the active **evo / hero / wild** (at most 1 evo, 1 hero, 1 wild) | missing | (A) per-slot form flags (the deck pool has them); use the active slots, not "this card can evolve" |
| Cycles until each evo is ready; ready | missing | (A) evolution progress per slot |
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
