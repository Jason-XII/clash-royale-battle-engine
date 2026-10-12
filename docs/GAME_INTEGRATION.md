# How the game takes our input: cards, touches, timing

> This file is kept in two repos with the same text: clapha (where the measuring tools are) and
> Jason-XII/clash-royale-battle-engine. Paths like `mac012/...` and `il/...` are clapha's. Change both in the same
> change, with the evidence; old numbers go to the changelog at the end.

What the Clash Royale client does with a card play, measured on the device, and what follows for the
console and for training. Everything the bot does to the game goes through these rules; a rule we have
wrong here is a play the model made that never happened, or happened late.

Measured on build 160402012 (MuMu, 1440x2560), 2026-10-03, with `mac012/tap_probe.py`: six friendly
battles between the user's two accounts with both bots off, about 330 timed gestures, each touch placed
against the game's memory on the device's own clock (20 ms samples). Records and the full report:
`docs/measurements/tap_probe_2026-10-03/`. Sections say where a number comes from; "7x" marks what was
only measured in the 7x-elixir friendly mode. To measure again, see the last section.

## 1. The rules, in short

1. A play is two touches: the card's hand slot, then a tile. The command gets an **issue tick** 0-4 ticks
   after the tile touch goes up (usually +1 or +2) and **executes 21 ticks after its issue tick**.
2. At the touch the *screen* takes the card out of the hand and its cost off the elixir bar. The *game*
   does both when the command executes, 1.05 s later.
3. The slot of a played card is **empty on the screen until the game deals the next card into it**, which
   it does at execution: exactly 21 ticks after the played card's issue tick. The next card is shown only
   under "Next:".
4. **A touch on that slot before the deal does nothing.** It does not play the card, select it, or queue
   anything. **From 100 ms after the deal a touch is always taken**; in the 100 ms before that, sometimes.
5. Two cards that are both in the hand need no delay between them beyond ~5 ms from the end of one
   gesture to the start of the next.
6. How long each touch lasts and the gap between the two touches of a play do not matter (2-48 ms holds,
   0-33 ms gaps).
7. The elixir the client checks at the touch is the **screen's**: every earlier touch's cost already off.
8. The client takes no play in the first seconds of a battle: first touch taken at tick 94, first issue
   tick 101.
9. A card asked onto a tile it may not take is **moved to the nearest tile it may**, not refused. A
   building asked onto a standing building is moved three tiles away.
10. Our own command first shows in the queue we read **1-12 ticks after its issue tick**, 4-17 ticks after
    the touch. Nothing can be called lost before that.

## 2. Clocks

- **Tick.** The game runs 20 ticks a second (`game_tick`, the simulated tick at battle+0x60: 19.99
  ticks/s measured over 20 s). Ticks are not perfectly even against real time: over a second the tick a
  given moment falls in can be one off a straight 50 ms grid.
- **Device clock.** The reader stamps every sample with the device's monotonic clock
  (`sample_monotonic_us`). `src/fast_tap.c` (version 3) reads the same clock: it reports when each touch
  went down and up, and `at US <gesture>` starts a gesture at a given time on it. That is what lets a
  touch be placed against the game's memory to the sample interval (20 ms at best: the reader's and the
  queue tool's minimum).
- **What the client shows is behind what it has received.** battle+0x64 (received tick) runs 6-19 ticks
  ahead of battle+0x60 (the tick we read): the client's buffer against network jitter. We read the
  simulated state.

## 3. A play from touch to board

Times from one play with the console's gesture (two 16 ms touches, 8 ms apart):

| when | what |
|---|---|
| 0 ms | card touch goes down on the hand slot; the card lifts (it is selected) |
| 16 ms | card touch up |
| 24 ms | tile touch down |
| 40 ms | tile touch up: the play is made. On the screen the slot is now empty and the cost is off the bar |
| issue tick | the tick the tile touch went up in, plus 0 to 4: +0 9, +1 38, +2 21, +3 8, +4 2, -1 1 (79 plays) |
| issue + 1..12 ticks | the command first shows in the queue we read (423 commands: mostly 3-11, most often 10) |
| issue + 21 ticks | it executes: the card leaves the game's hand, the next card is dealt into that slot, the game's elixir drops |
| issue + 22 ticks | its unit is on the board |
| deal + 100 ms | the dealt card can be touched |

- The console sees "tap -> issued" as 1-6 ticks (2 in 166 of 262 plays), counted from the frame it sent
  the tap on. From the tap being sent to the console first seeing the command in the queue: 4-17 ticks
  (267 plays; more than 12 in 9%).
- The issue tick cannot be steered to better than about a tick from our side: the same touch timing gives
  issue ticks spread over three ticks.
- A command's execution is fixed once issued: nothing we do later changes when it lands.

## 4. The hand

- Four slots; the deck's other four cards wait in the cycle, the first of them shown as "Next:".
- **The deal.** The game deals the cycle's first card into a slot when the card played from it executes:
  21 ticks after that play's issue tick, 139 of 139 single plays. In memory the slot goes from one card
  to the next within one sample, or stands empty for a tick or two (387 / 34 / 3 of 424 deals, 7x).
- **Two plays close together: the refill timer.** One timer per player. When it reads 0, a played card's
  slot is refilled as the play executes and the timer restarts at 1000 ms (500 ms in double elixir, 350 ms in
  triple); a play that executes while it is running leaves its slot empty until it runs out; several waiting
  slots fill one per period, the lowest slot first. This is RoyaleSim's measurement on client 15.535.29
  (github.com/RoyaleGym/RoyaleSim, `tests/hand_refill.rs`), and our five normal-elixir games of 2026-10-03
  fit it: of 20 plays that executed less than 20 ticks after the previous deal, 14 were dealt 19-20 ticks
  after that deal, 2 ten ticks after it and 4 at once (the period being 10 or 7 ticks in double and triple
  elixir). So a second card played right after another can leave its slot empty for up to a second longer.
  7x: no wait at all -- each slot dealt at its own issue + 21, two commands with the same issue tick a tick
  apart (19 pairs). Our reader's `refill_timer` field reads 0 on this build (it is read at the older
  client's offset), so the console goes by the hand itself.
- **On the screen** (`hand_during_play.jpg`): the played card's slot is empty, a blank frame, from the
  touch until the deal; then the new card pops in. Cards the bar cannot pay for are grey. A selected card
  is lifted with a bright edge.
- **When the dealt card can be touched.** The second tap of 139 trials, by when its card touch went down,
  counted from the first 20 ms sample that had the new card in the game's hand:

  | card touch down | taken | refused |
  |---|---|---|
  | before the deal | 0 | 31 |
  | 0-49 ms after | 1 | 17 |
  | 50-99 ms | 19 | 33 |
  | 100 ms and later | 38 | 0 |

  It is the touch going *down* that counts. With 2 ms touches: refused at 13, 31, 53, 64 ms; taken at 70,
  77, 87, 100. With 48 ms touches, down at 54-87 ms and still down past 100 ms: all six refused; down at
  110: taken.
- **A touch before the deal leaves nothing behind.** The slot touched 50-400 ms before its deal and a
  tile touched 300 ms after it: the dealt card was put down 0 of 9 times. A refused tap is simply gone.

## 5. The touch

- **Shape.** One troop or spell long in the hand, each touch held 2-34 ms with 0-33 ms between the card
  touch and the tile touch (twelve combinations): 30 of 31 taken. Nothing to tune here.
- **Selection.** The card touched last is the one a tile touch puts down (card A, card B, tile: B). A
  selected card stays selected for at least 1.5 s. Touching it a second time does not deselect it. A tile
  touched with nothing selected does nothing.
- **Two gestures in a row** (both cards long in the hand), by the time from the first gesture's last
  touch going up to the second's first touch going down:

  | between | pairs | both taken | second lost | both lost |
  |---|---|---|---|---|
  | under 5 ms | 17 | 10 | 5 | 2 |
  | 5 ms and more | 19 | 18 | 0 | 0 (one first lost) |

  In the live games of 2026-10-03 the console wrote the two plays of a turn back to back: 5 of 9 second
  gestures lost.
- **What a lost tap costs.** The touch itself costs nothing in the game. The cost was in our bookkeeping:
  the console counted the card as sent, so the model's hand showed the card after it in that slot, and the
  next tap "for that card" went to the slot still holding the first one and put *it* down on the other
  card's tile. Six such in five games (a Cannon on the Hog Rider's tile at the bridge, a Log on the
  Cannon's tile).

## 6. Placement

- **Screen point of a tile.** `ScreenLayout.deployment_point` takes a cell of the local player's own view
  (row 0 at their back line), with the column mirrored for both sides (`mac012/layout_fix.py`); the
  console turns FirstLight's native tile into that cell (`console.screen_cell`). Checked on every play by
  the console's "placement ... asked ..., game got ..." line: 0.0 tiles off.
- **A tile the card may not take** (7x, cards in the game's own coordinates):

  | asked | what the client did |
  |---|---|
  | a troop on the other half | put on our front row of that column (row 14); with that lane's tower down, on the nearest tile of the opened pocket |
  | a troop on the river or a bridge | row 14 of that column (a bridge tile itself once the pocket is open) |
  | a troop on our King Tower's tiles | behind it, row 0 of that column |
  | The Log on the other half | row 14 of that column: it is placed like a troop, on our side only |
  | Fireball on the other half | there: a spell may go anywhere |
  | a building on a standing building's tiles | three tiles left, right or back of where asked (7 times) |
  | a building against a tower | one or more tiles away from it |

  None of 28 was refused. So a wrong tile never loses the play; it changes where it lands. The action
  mask is what keeps the model's tile and the game's the same.

## 7. Elixir

- The game's elixir (what we read) drops when a command executes. The bar on the screen drops at the
  touch.
- **The client checks the screen's.** The whole hand played from a full bar, cheapest first, 30 or 60 ms
  apart: Skeletons (1), Log (2), Musketeer (4) taken, Hog Rider (4) refused, both times -- the game still
  held ten elixir, the screen three.
- So the elixir a play may count on is the game's now, less the cost of every card touched and not yet
  executed, plus what regenerates until its own touch. That is what `firstlight_obs.screen_view` and the
  mask's elixir lead (`il/params.elixir_lead`) compute.

## 8. The start and the end of a battle

- **Start.** One play at each of four ticks: touches at ticks 70, 78, 86 refused, at 94 taken; in two
  other battles touches at 37, 56, 87.5, 88.7 refused and 98.6, 99.5 taken. All the taken ones were issued
  at tick 101. FirstLight's first decision tick is 90, whose earliest tap is tick 94. RoyaleSim carries the
  same lockout as 90 ticks (another user's reading of the real engine).
- **End.** After the result is decided the client's clock runs on and it still takes touches, though no
  command executes (NOTES, friendly 2026-09-25): the console stops tapping when `battle_result` says the
  battle is over.

## 9. What we see of the other side

- Their commands are in the same queue as ours, with card and tile, before they execute: 2-21 ticks
  before, median 13 (`il/params.OPPONENT_LEAD_TICKS`, 34 recorded sessions).
- The cards they have played (in first-play order) and their exact elixir are readable; their hand and
  cycle are not (the hand reads empty for the whole battle, the cycle pointer is null). A Royal Trainer's
  commands carry account -1.
- **Their evolution and hero slots** (measured 2026-10-11 on 515 recorded battles, 443 against real players):
  - In the battle's own player data, their deck and its slot flags are hidden until the battle ends, against every
    kind of opponent: readable before the first play in 0 of 515 battles, at the end in 97% of real-player battles.
    From that data an opponent evolution is known once played (the unit's form id), or guessed from a deck prior
    (RoyaleAPI, or the deck the other console publishes); a HUD must show the guess as a guess.
  - **Open:** a separate record outside the battle (found by scanning the game's memory, `deck_vector_scan`; it looks
    like what the result screen / chat card is drawn from) held the opponent's whole deck **before the first play** in
    a friendly and in Training Camp (2026-09-24). Not tested in ladder battles, and not checked for evolution / hero
    forms per slot. If it holds both in ladder battles, opponent evolutions are known at the start.
  - **Their hero and champion are known from the start**: their ability controller names the card, and every
    controller seen was there before the first play (132 of 132). Our table of controller characters lacked Hero
    Ice Wizard (character 1741956296, 14 battles) and Goblinstein (4047250230, 3 battles).
  - The slot flags mean the **active** form of the slot, not "this card can evolve": on our own side, flag 1
    slots were played as an evolution 291 times and never as a hero, flag 2 slots as a hero 123 times. No other
    value occurs; a wild slot shows as what is equipped in it.

## 10. Where the simulator differed from the game

The engine runs the game's own logic, so the deal, the command age and the board are the game's. What
differs is what stands between a decision and a command -- the client -- which our harness (`il/duel.py`)
plays the part of.

| | the client | the harness until 2026-10-03 | now |
|---|---|---|---|
| a card not yet dealt | cannot be touched | sent anyway: the command went in a tick before it executed, by when the card had been dealt | `deal='hold'`: the tap waits for the deal + 3 ticks; `deal='mask'`: and the card is not offered |
| the hand the model is shown | the played card's slot empty until the deal | the next card already in the slot (`screen_view`) | with `deal_rule` that card is in the view but illegal |
| elixir at the tap | the screen's | the screen's | same |
| two plays in one turn | both taken if 5 ms apart | both taken | same |

- Every model up to pilot3 was trained with the first row wrong. In pilot3's own recordings 160 of the
  learner's 2,543 plays (6%, about four a game) were issued 1-12 ticks before their deal, and landed. Live
  the same habit was six lost taps a game.
- `il/vs_firstlight.py` (FirstLight's own environment) dropped such plays, as the client does: part of
  why pilot3 did worse there than in training.
- How the current model reacts when the card is simply not offered (`deal='mask'`), decided again on 291
  such turns of 36 recorded games: it waits in 79%, and would play another card in 14% -- but for the
  Cannon in half (Ice Spirit, Musketeer, The Log in its place). So for a model trained without the rule
  the console holds the tap instead; the mask is for models trained with it (`il.rl collect --deal mask`,
  recorded in the checkpoint, read by the console).

## 11. What the console does with each rule

`mac012/console.py`, `mac012/tapper.py`:

- **The deal** (`_note_hand`, `_try_play`, `DEALT_MS`). The console keeps, per slot of the game's own
  hand, when it first saw the card there. A play is tapped in the slot where the game's hand holds its
  card, never by the model's hand, and its first touch goes down 120 ms after that first sight, on the
  device's clock (fast_tap `at`). Until then the play waits, up to 2.5 s; if the model picks the card
  again meanwhile, the newer tile is used.
- **A model trained with the rule** is not offered a card the game has not dealt
  (`firstlight_obs.screen_view(deal_rule=True)` -> `action_mask` reason `not_dealt`), so its plays need no
  wait.
- **Two plays in a turn** are written 20 ms apart (`Tapper.spacing`).
- **The start**: no tap before tick 95 (`FIRST_TAP_TICK`).
- **A lost tap**: a tap whose command is not in the queue 20 ticks later is sent again once
  (`LOST_AFTER_TICKS`).
- **Timing**: every play is held to land `TARGET_DELAY` (26) ticks after its moment; the timing record
  (`build/timing_<port>.jsonl`, `il/timing.py`) has each play's ticks.

## 12. Measuring again

After a game update, or to test a new idea about the client:

    ./start-consoles.sh                       # builds and pushes src/fast_tap.c (version 3)
    # a friendly between your two accounts (7x elixir gives the most trials), both bots off
    CR_MUMU_SERIAL=127.0.0.1:26656 ./py mac012/tap_probe.py deal pairs shape select early illegal build occupied burst
    ./py mac012/tap_probe_report.py build/tap_probe/<time>.jsonl

`tap_probe.py` waits for a battle, runs the named experiments in turn until it ends, and writes every
hand change, command, touch and trial. `--deltas`, `--spacings`, `--shapes`, `--deal-holds` choose the
values tried. It keeps the Hog Rider and Musketeer in the hand so the other side's towers stand longer.
The report decides "taken" from the whole queue record, not from what the probe saw at the time.

In ordinary games, with the bot playing: `il/timing.py` (lateness of every play), `il/live_games.py`
(the match in the app's replay, every failed click marked with its reason).

## 13. Not known yet

- The refill timer's periods on this build (1000 / 500 / 350 ms are the 15.535 client's), its period in 7x,
  and where it sits in memory here.
- Whether the 100 ms after the deal is the same on another device or frame rate (it looks like an
  animation's length).
- Building placement next to towers and other buildings: moved, but by what rule.
- The tick a touch must come after at the start of a battle is between 89 and 94 (90 by RoyaleSim's note).
- Nothing here was measured for hero and champion ability buttons.

## Changelog

- 2026-10-03: first measurement, build 160402012 (MuMu), tap probe (sections 1-12).
- 2026-10-11: shared with clash-royale-battle-engine; the refill timer (section 4) cites RoyaleSim's measurement on
  the August client; its double / triple elixir periods are not measured on our build yet (section 13).
- 2026-10-11: section 9, the opponent's evolution and hero slots: what is readable when (515 recorded battles);
  the out-of-battle deck record (friendly / Training Camp only so far) added as open.
