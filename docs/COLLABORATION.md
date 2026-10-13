# Working with Arron (clapha)

For Jason and his AI. Arron builds clapha (a Clash Royale bot console and clapha-engine, the engine this repo trains
on) and sends fixes here as pull requests. This file says where things live, how training metrics reach Arron, and how
lasting findings are recorded.

## Where things live

| What | Where | Public? |
|---|---|---|
| Self-play (PPO), arena, defense stats, the engine wrapper | this repo | yes |
| Model, inputs, replay conversion, imitation learning | github.com/ClashAlpha/clapha-il (Jason's work on branch `jason`) | yes |
| Pull requests from Arron (fixes, speed-ups) | this repo, from arronwang01/clash-royale-battle-engine | yes |
| What the model's input is and lacks | clapha-il's docs/INPUTS.md | yes |
| How the live client takes a play (deal, refill, elixir, timing) | [docs/GAME_INTEGRATION.md](GAME_INTEGRATION.md) | yes |
| The engine: loader, per-build addresses, porting a new game build, engine reads per card | github.com/arronwang01/clapha-engine | **no** (Jason has access) |
| Training metrics for Arron | sent by Jason on request (below) | no |
| Arron's console: Mac app that plays on the MuMu emulator with a model (FirstLight's or this repo's), HUD, Wi-Fi phone | github.com/arronwang01/clapha-console (its README says what it is and what it needs) | **no** (Jason has access) |

Rule of thumb: anything about the engine's insides (addresses, memory layouts, how a new game build is ported) goes to
clapha-engine; everything else goes here.

## Metrics for Arron

Not pushed automatically: training logs stay private (Jason, 2026-10-12; PR #6's `tools/report_metrics.py` was not
taken). Jason sends lines of `runs/<run>/metrics.jsonl` when they are needed. The fields that show where time goes:
`collected_per_hour` vs `games_per_hour` (collectors or learner the limit?), `backlog` / `held_s`, `collector_ms`,
`server_ms_per_call`, `mean_batch`, `update_s`, `wait_s`.

## Lasting findings

A finding is worth keeping when the next model, the next game build or the next person needs it: a game rule, a
measured timing, a card's special state, an engine read. Each one is written down with **what it is, how it was
measured or checked, on which build and date**, and **how to check it again**:

- game and client rules → docs/GAME_INTEGRATION.md (here);
- model input requirements → clapha-il's docs/INPUTS.md;
- engine reads, per-card engine states, porting notes → clapha-engine (private).

Nothing is written as fact before it is checked; open questions are marked as open.

## How docs/GAME_INTEGRATION.md is kept

- It is measured on the live client with Arron's tap probe (in clapha) and each section says where its numbers come
  from. The current numbers are from build 160402012 (October); Null's Royale runs the August build, where some
  timings may differ (marked where known).
- After a game update, or to test a new idea about the client: run the probe again, compare, and change the section
  with its new date and evidence; old numbers move to the changelog at the end.
- Arron edits it in clapha, next to the probe, and sends the same text here in the same change, so the two copies
  never differ.
