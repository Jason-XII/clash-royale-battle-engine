# The imitation-learning pilot: steps, checks, what to send back

For Jason and his AI. The IL model and pipeline live in the private repo ClashAlpha/clapha-il; the engine side (oracle,
readers) in ClashAlpha/clapha-engine. This page is the order of work on the cluster and what must be true after each
step. The full checks, and what a wrong number means: clapha-il `docs/MONITOR.md`. Stop at any failed check and tell
Arron; do not work around it.

Before starting: `git pull` in ~/clapha-engine and ~/clapha-il (both changed on 2026-10-12: a Dagger Duchess read that
failed after an engine's first battle, the elixir check, tower-troop trials, monitoring). Game files: Arron's Drive
folder (`shasum -a 256 -c SHA256SUMS`); the collected battles are the **-v2** tar.

| step | job | what must be true |
|---|---|---|
| 1. Which IL_Replay games replay exactly | `tools/oracle.sbatch` (running) | report: about 70% exact; "error" rows only for replays the converter refuses |
| 2. Which of Arron's battles replay | `tools/oracle_collected.sbatch` (a `LIMIT=60` trial first) | report: about 65% accepted; ambiguous ones (two tower-troop pairs, or both sides) are listed as rejected, never accepted |
| 3. Conversion, label check, audit | `tools/convert.sbatch` (a `LIMIT=40` trial first) | every accepted game kept; `check_converted.py`: **PASS** for both; both audits: **no flags** |
| 4. Look at three games | `inspect_il.py game` (clapha-il) on a Dagger Duchess game, a hero / champion game, an evolution game | the hand, elixir, opponent's shown cards, plays in flight, legal tiles and unit states look right decision by decision |
| 5. The pilot: five runs | `tools/pilot.sbatch`, one per setting | no skipped steps (`nonfinite` 0); `data_wait` under ~0.2; card accuracy above the audit's baselines (39% card, 15% / 43% tile top-1 / top-5 on the Mac sample) |
| 6. Report | `inspect_il.py run ~/cr-engine/runs/pilot-*` | sent to Arron (below) |
| 7. The full run | `tools/il_full.sbatch` | the rules in "Starting the full run" all hold |

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

**4. Start it:** `FLAGS="..." sbatch tools/il_full.sbatch` with the flags of the chosen run (none for `pilot-base`;
`--attention-layers 0`; or `--core 1024 --conv-channels 160 --object-dim 192`). It trains on every exact IL_Replay game
until the score stops improving (5 validations without a better one), resubmitted after each 4-day limit (it resumes).
Arron's collected battles are left out of this run on purpose: they pass a weaker check than exact tower HP; they come in
later (matchup specialists) after a comparison of their own.

**5. Report:** in clapha-il's `reports` branch, `reports/<date>/full-run.md`: the five runs' last validation lines,
which rules held, the choice and why, the job id. Then the run report every day or two
(`inspect_il.py run ~/cr-engine/runs/il-full`).

**While it runs**, the same health rules hold: `nonfinite` stays 0, `data_wait` under ~0.2, the validation score goes
down. If the validation score rises for 3 validations in a row from the start, or anything is NaN, stop the job
(`scancel --name=cr-il-full`) and tell Arron.

## After the full run

`best.pt` is the model. Before it replaces anything, arena games: it plays whole games in the engine against the current
best PPO model (the tool is being built in clapha-il and will be added to this page). The console's live check comes
after that.
