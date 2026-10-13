# The imitation-learning pilot: steps, checks, what to send back

For Jason and his AI. The IL model and pipeline live in the private repo ClashAlpha/clapha-il; the engine side (oracle,
readers) in ClashAlpha/clapha-engine. This page is the order of work on the cluster and what must be true after each
step. The full checks, and what a wrong number means: clapha-il `docs/MONITOR.md`. Stop at any failed check and tell
Arron; do not work around it. Policies, what to do when something goes wrong, and how speed-ups are done here:
clapha-il `docs/HANDOFF.md` (private repo) — read it first.

Before converting: `git pull` in ~/clapha-engine (795e83a or later: the Dagger Duchess read failed in every battle after
an engine's first, so earlier conversions have no daggers) and in ~/clapha-il (0155582 or later: conversion 2.7x faster with identical records; idle games left out; and from 99ebee7: the training input is built ~4x faster per core and batches carry only the
object slots they use; every value byte-for-byte the same as before, checked on 168 player-games with
`tools/compare_features.py`). Game files: Arron's Drive
folder (`shasum -a 256 -c SHA256SUMS`); the collected battles are the **-v2** tar.

| step | job | what must be true |
|---|---|---|
| 1. Which IL_Replay games replay exactly | `tools/oracle.sbatch` | report: about 70% exact; "error" rows only for replays the converter refuses |
| 2. Conversion and label check | `tools/convert.sbatch`, then `check_converted.py ~/il-raw` (clapha-il) | every accepted game kept; **PASS** |
| 3. Data audit | `inspect_il.py audit --games ~/il-raw/games --convert-out ~/il-raw --sample 300 --firstlight ~/firstlight-cr --out audit-il.html` (clapha-il) | **no flags** (it checks that Dagger Duchess games have daggers: games converted before clapha-engine 795e83a do not; on the pilot's games that one flag is acceptable, a few % of games; on the full conversion it is not) |
| 4. Look at three games | `inspect_il.py game` on a Dagger Duchess game, a hero / champion game, an evolution game | the hand, elixir, opponent's shown cards, plays in flight, legal tiles and unit states look right decision by decision |
| 5. Inputs played = inputs trained | `tools/check_clapha_inputs.py` | no differing decisions besides the hand's unknown flag |
| 6. The pilot: five runs | `tools/il.sbatch` (RUN=pilot-...) | see "Starting the full run" step 1 |
| 7. Report | `inspect_il.py run ~/il-runs/pilot-* --out pilot-report.html` | sent to Arron (below) |
| 8. The full run | `tools/il.sbatch` (below) | the rules in "Starting the full run" all hold |
| (later) Arron's collected battles | `tools/oracle_collected.sbatch` | not needed for the full run; for the matchup specialists |

**One clapha-il.** `il/clapha_agent.py` uses `clapha_il.features.PlayerView`, which is not in ClashAlpha/clapha-il yet: push it
there, merged with clapha-il's 2026-10-12 changes to `features.py` (time since an ability was used, the Royal Chef
cooking clock, channel names) and `train.py` (the card loss only where the whole hand is known; --resume, --patience,
best.pt, --val-games; several --games; `cycle.own_views`, the fast hand reconstruction). The pilot, the full run and the
arena must all run the same clapha-il commit,
or the arena plays a model with inputs it was not trained on. Re-run `check_clapha_inputs.py` after the merge.

Arron's mac checks of the same code (2026-10-12): 60 IL_Replay + 24 collected games converted, all kept, PASS, audit
with no flags; a 300-game oracle sample: 211 exact, all of them pass the elixir check.

## What to send back

Training logs stay private, as agreed: nothing of it goes to this public repo. After steps 1-3 and after the pilot, put
the small outputs in the private clapha-il repo, branch `reports`, folder `reports/<date>/`: the oracle report lines, the
`check_converted.py` lines, `audit-*.html` / `.json`, and for the pilot `pilot-report.html` / `.json` (`runs.json` holds
every `metrics.jsonl`). No checkpoints, no game data. Arron's AI reads them and reports to Arron.

## Starting the full run (no need to wait for Arron when every rule below holds)

Use each run's last validation line in `metrics.jsonl` (`best_score` = the lowest what + which + tile loss on held-out
games; lower is better; a run started before clapha-il 5c7aa25 has no `best_score`: take the lowest what + which + tile
over its validation lines) and the IL audit's `baselines` (`audit-il.json`). Pull clapha-il before the pilot.

**1. Every run must be healthy.** If any rule fails for `pilot-base`, stop and tell Arron; if it fails for another run,
leave that run out of step 2 and say so in the report.
- `nonfinite` adds up to 0 over the run (no skipped batches);
- the last validation: `card_acc` above the baseline "card: most played card in hand", `tile_top1` and `tile_top5`
  above the two tile baselines;
- the last validation: `p_play_mean` within 25% of `play_rate` (the model plays about as often as people);
- no NaN anywhere in the validation lines.

**2. Choose the size and attention.** Among `pilot-base`, `pilot-no-attention`, `pilot-large`: the lowest `best_score`;
but when a cheaper run is within 1% of it, the cheaper one (no-attention, then base, then large).

**3. Keep unit states and effect vectors on.** If `pilot-no-unit-states` or `pilot-effect-ids` has a `best_score` more
than 1% *better* than the run with the same size, do not start: that points at a bug in those inputs; tell Arron.

**4. Start it** with the chosen run's flags (none for `pilot-base`; `--attention-layers 0`; or `--core 1024
--conv-channels 160 --object-dim 192`) after the full conversion:

    RUN=il-full GAMES=~/il-raw/games VOCAB=~/il-runs/vocab-full.json LIMIT=0 EPOCHS=20 \
      FLAGS="<chosen flags> --patience 5 --val-every 5000 --val-games 1000 --resume" sbatch tools/il.sbatch

It trains on every exact IL_Replay game until the score stops improving (5 validations without a better one; `best.pt`
is the model). The same line again after the job's time limit: it resumes from `latest.pt`.
Arron's collected battles are left out of this run on purpose: they pass a weaker check than exact tower HP; they come in
later (matchup specialists) after a comparison of their own.

**5. Report:** in clapha-il's `reports` branch, `reports/<date>/full-run.md`: the five runs' last validation lines,
which rules held, the choice and why, the job id. Then the run report every day or two
(`inspect_il.py run ~/il-runs/il-full`).

**While it runs**, the same health rules hold: `nonfinite` stays 0, `data_wait` under ~0.2, the validation score goes
down. If the validation score rises for 3 validations in a row from the start, or anything is NaN, stop the job
(`scancel` the job) and tell Arron.

## After the full run

`best.pt` is the model. Before it replaces anything, arena games (`il/arena.py`, which plays clapha-il checkpoints): against
the current best PPO model and the best pilot model. The console's live check comes after that.
