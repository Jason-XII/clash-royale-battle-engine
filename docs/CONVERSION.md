# Converting RoyaleAPI battles exactly

A converted battle is only usable when the engine rebuilds it as it was played: every tower ends at the HP RoyaleAPI
recorded. These rules took the 2,000-battle sample of the 700k dump from 54% (convert.py before them, October) to **95.6%
exact** in the 6-18 October window (1,076 of 1,126 on the native engine, 73 of them with a found seed), and 1-5 October
to 162 of 165 non-dice battles. Each rule was measured on real battles and made its battles exact without breaking any
exact one. The full record (every measurement, the engine side) is clapha-engine `docs/training/CONVERSION_RULES.md`.

Needs clapha-engine at ba78ae1 or later (`CrxEnv.selection_of(elixir_less=)`, `queue_card_at(selection=)`,
`ready_ability_sources`, `queue_ability_action_at(check_ready=False)`).

## The rules (all in convert.py)

| # | Rule | Was |
|---|---|---|
| 1 | RoyaleAPI's tick (`data-t`) is the command's **issue** tick; the game lands a play or press **21 ticks later**. Labels: the last decision before the issue tick | landed at tick + 1, 20 ticks early (a live friendly: 71 of 71 plays; the 300-replay sample 210 -> 249 of 266) |
| 2 | A press names its unit; RoyaleAPI does not. Where two different cards were ready, `play_exact` tries each and keeps the one that reproduces the battle | newest ready unit |
| 3 | A press is checked at its issue, not 21 ticks later (`check_ready=False`) | refused when the button was not ready at execution |
| 4 | A card's selection (form, cost) is made at the tap (`selection_of` at the issue tick) | taken at execution |
| 4b | That selection sees the **client's** bar: engine elixir less the player's own plays still in flight (the engine pays at landing) | Spirit Empress mounted where she was on foot |
| 4c | **Mirror** copies the card that has landed when the Mirror lands: no selection at the tap | copied the card before |
| 5 | The King's level is not published: tower troop's level, else the side's commonest card level, whichever reproduces | one level for both Kings |
| 6 | Run until the game's own rules end the battle (7,200 ticks only as a safety stop) | the recorded length (cut tiebreakers short) |
| 7 | Each battle on the game data live at its time (below; `tools/assign_by_date.py`) | one engine per month |
| 9 | Cards kept out of the starting hand (Elixir Collector, Mirror) wait at a queue place set by the **seed alone**: `calibrate_deal` searches seeds | moved the card between deck slots and refused (51 of 1,791) |
| 10 | Score only the six tower objects (native ids 5,000,000-5,000,005) | a troop on a fallen tower's spot was read as the tower |
| 11 | A hot update reaches players **during** its first day: such a battle runs on both data versions (`alt`), the one that reproduces counts | data by date only |

Dice: Skeleton King, Hunter, Elixir Golem (and Graveyard on older data) draw from the battle's seed, which RoyaleAPI does not
publish. `--seeds N` tries up to N seeds per battle that misses (stops after 5 seeds give the same result).

## Game data by battle time

| From (UTC) | Data (`--engine` name) | Engine code |
|---|---|---|
| 10-19 ~09:40 | d22: 16.402.22 (real) | 160402020 |
| 10-06 ~06:30 | d19: 16.402.19 (real) | 160402020 |
| 28 Sep - 1 Oct | oct1: rebuilt -- Hero Ice Wizard rework + three `V16_*_BUG` flags (in no official post) + the Electro Wizard hero (5 Oct; its two files must be listed in `data_manifest.toml`) | 160402012 |
| 09-17 | sep17: rebuilt, the 16 Sep post (4 changes) | 160402012 |
| 09-07 ~05:30 | d2b: 16.402.2 from app 160402012 (8 Sep post) | 160402012 |
| 09-01 | d2a: 16.402.2 from app 160402002 | 160402002 |

The asset server serves only the current data (older versions 403), so sep17, oct1 (and jul6, aug26 for the summer) were
rebuilt from real later files and accepted only when their battles turned exact; ask Arron for the folders. For future
battles: save each data version when it goes live (its fingerprint changes) and none has to be rebuilt.

## Running it on the schema-5 dump

1. `python clapha-engine/crx/replay_audit/schema5_json.py DUMP OUT` -> one JSON-lines file per window (keeps Ranked,
   Ladder and Grand Challenge; drops River Race, friendlies, tournaments, showdowns, Classic Challenge).
2. `python tools/assign_by_date.py assignment.jsonl OUT` -> each battle's data version (+ `alt` on update days).
3. `python convert.py --assignment assignment.jsonl --replays OUT --engine d19 ENGINES/160402020 DATA/d19 --engine d22 ...
   --clapha-engine ~/clapha-engine --out ~/il-data/oct --seeds 64`

Also left out: battles before July 2026 (no data of their time) and the Tombstone hero on 3-4 September 2026 (a bug made it
free).

## Finding why a battle misses

1. Group the misses by date (a cluster is a data window), by card against the exact battles, by mode.
2. Per battle: towers, each side's elixir spent and leaked against RoyaleAPI's, failed plays and presses, plays held for
   elixir (the first held play: the rebuild already differs before it).
3. Every hit on each tower and what dealt it: one hit too many = a unit died a few ticks late.
4. Move single plays a few ticks as a lead finder, compare the two runs object by object, then find the mechanism. A
   match is a lead, never proof: one battle matched with either of two Miners delayed; its cause was rule 4b.
5. Build candidate data from the real later file (never an invented value) and date it by the battles that need it.
6. Run every fix on all battles with that card: it must fix misses and break none.

Open: the dice seed (perhaps computable from public battle fields), a play's landing tick after a bad network (RoyaleAPI
shows only the tap), 8 misses in 6-18 October without a confirmed cause.
