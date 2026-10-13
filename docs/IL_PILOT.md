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

Arron's mac checks of the same code (2026-10-12): 60 IL_Replay + 24 collected games converted, all kept, PASS, audit
with no flags; a 300-game oracle sample: 211 exact, all of them pass the elixir check.

## What to send back

Training logs stay private, as agreed: nothing of it goes to this public repo. After steps 1-3 and after the pilot, put
the small outputs in the private clapha-il repo, branch `reports`, folder `reports/<date>/`: the oracle report lines, the
`check_converted.py` lines, `audit-*.html` / `.json`, and for the pilot `pilot-report.html` / `.json` (`runs.json` holds
every `metrics.jsonl`). No checkpoints, no game data. Arron's AI reads them and reports to Arron.

## After the pilot

The full run waits for Arron's sign-off on the pilot numbers and for arena games: the pilot models playing whole games in
the engine against each other and against the current best PPO model (being built in clapha-il). The console's live
check comes after the full run.
