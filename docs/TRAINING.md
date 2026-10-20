# Training notes: imitation (IL) and PPO

What we decided, measured and discussed while training: IL from human replays, then PPO from the IL model. It's
written in order. Sections 1–5 cover the small model's PPO runs (`~/cr-ppo/runs/ppo-4`, 15–16 October), section 6
covers scaling up to the medium IL model, and section 7 covers PPO from it (`~/cr-ppo/runs/ppo-m1`). Sections 8 and
9 cover the current run, gen-large, under the improvement loop, and the deployment-temperature finding.
How to run each script is in `README.md`, and the live client's integration is in `GAME_INTEGRATION.md`.

## 1. The run

- **Learner**: 3 A100s (DDP), plus 13 collectors on the 4th A100.
- **Actors**: RTX 5090 job (34 collectors, 4 decision servers) and Students job (50 collectors, 3 servers). A Students
  job is limited to 1 day, so three more are chained behind it (each starts when the previous ends).
- **Speed**: ~36–48k games/h. Collectors are CPU-bound; GPU decisions use CUDA graphs (2.3k → 46k games/h).
- **Opponents**: half self-play; the rest the IL model or recent snapshots (`pool/v000100.pt`, every 100 updates),
  picked more often the less often the policy beats them.
- **Reward**: +1 win / −1 loss; −0.002 per decision at full elixir; −0.01 per play the engine refuses;
  tower-damage shaping (0.1 × change in "crowns + tower HP lost, theirs minus ours").
- **Critic** (since ppo-4): a separate network, used only in training (never in live play):
  - a copy of the policy's network from the IL weights, plus the opponent's hand, next cards and ability;
  - two heads: P(win) trained on the final outcome, and the short-term (shaping/penalty) rewards.
- **Stop everything**: `scancel --name=cr-ppo`.

## 2. Monitoring

```bash
ssh cluster 'tail -f ~/cr-ppo/runs/ppo-4/metrics.jsonl' | jq -c '{v:.version, gph:.games_per_hour, score, win_ev, win_ev_late, kl_ref, entropy, refused:.refused_per_game}'
```

**Strength**
- `score.<opponent>`: mean result against that opponent over its last ≤200 games (+1 = always wins).
  - `il` = the IL model; `v000100` etc. = snapshots.
  - A new snapshot starts near 0. If the current policy pulls above ~+0.1 within ~100 updates, it is still
    improving.
  - Scores swing ±0.1 (few games, random decks, opponent mix). Judge trends over 20+ updates.
- Training scores are only a rough gauge. The reliable measure is an **arena panel**: fixed decks and seeds, paired
  seats, 200 games, `arena.py`.

**Critic**
- `win_ev`: share of the variance in outcomes explained by P(win), over every decision of fresh games before
  training on them.
  - 0 = no better than 50/50. ≈ 1 − 4 × Brier score.
  - It cannot reach 1: early-game outcomes are genuinely uncertain.
  - It is not a strength meter. Evenly matched self-play keeps it moderate however strong both sides are.
  - Guesses: now ~0.3; pro level ~0.4–0.55; far superhuman maybe ~0.5–0.65. AlphaGo's value network was ~0.75 on
    Go, which is perfect-information and decided earlier.
- `win_ev_late`: the same over the last quarter of each game. ~0.6 now. A sudden drop means the critic broke.

**Policy update**
- `kl_ref`: distance from the KL reference: the IL model in the early runs, the current champion in gen-large.
- `approx_kl` / `clip_frac`: step size per update; ~0.002–0.01 / 0.01–0.1 is normal.
- `entropy`: falling fast = becoming too deterministic.

**Pipeline**
- `games_per_hour`.
- `trajectories`: games per update; capped at 2× batch.
- `stale`: games dropped as too old; should be 0.
- `wait_s` / `update_s`: learner waiting for data / time per update.
- `actors`: should be 105.

## 3. Timeline and findings

| Updates | Change | Why / result |
|---|---|---|
| ppo-3 (≈0–80) | Shared value head | Value fit stuck ≈0.35; stopped to free the GPUs |
| ppo-4 0–20 | Separate critic, critic-only warm-up | — |
| 20–30 | Minibatches built once per update, batch capped at 2× | Learner was rebuilding minibatches 8× per update and fell behind |
| 100–180 | lr 2e-5, KL weight 0.05 | **Stalled**: even with its update-100 self, KL to IL flat at 0.027 |
| 180 | lr → 5e-5, KL weight → 0.01 | Moved again: beat v000100 by ~+0.18, IL score highest so far |
| ~275 | −0.01 per refused play | Refused plays had grown 4 → 12 per game (see below); now ~8–9 |
| 280–410 | — | **Stalled again**: ≈0 vs its update-300 self, KL flat ≈0.08 |
| 410 | λ 0.95 → 0.99, batch 512 → 1024 | Noisy learning signal (see below); first ~50 updates: slight upward tilt |

**Refused plays** (10 self-play games each)

| Per game, both seats | IL model | PPO (update ~235) |
|---|---|---|
| Card taps | 47 | 78 |
| Refused | 2.7 | 15.1 |

- About half were taps without enough elixir, even after the client's ~1 s hold.
- The other half were taps while a previous tap was still held waiting for elixir. The "my card in flight" input
  misses held taps whose landing time moved.
- In the simulator these are free no-ops. On the live client a second tap during a hold may behave differently.

**Arena vs the IL model** (seed 7, ~185 games each)

| Checkpoint | Win rate |
|---|---|
| v100 | 0.73 ± 0.06 |
| v180 | 0.76 ± 0.06 |
| v200 | 0.82 ± 0.05 |
| v275 | 0.84 ± 0.05 |

**The "sine wave" in `score.il`**
- The IL model gets only ~7% of games, so its 200-game window covers ~6 updates of random decks. That alone is noise.
- On top of that, a feedback loop: the policy wins → fewer IL games → it forgets that style → loses more → more IL
  games again.
- The arena showed real strength flat at ~0.82 while the training score swung 0.55–0.75.

**Throughput decline** (47k → 36k games/h)
- Mostly the opponent pool growing: each decision server makes one GPU call per distinct model in a batch.
- A little from longer games.
- The fix (all collectors on a server playing the same opponent, rotated) was declined for now.

## 4. Why PPO alone plateaus here

- **Credit assignment**: ~760 decisions per seat per game, most of them waits. One better placement moves the win
  probability by a fraction of a percent, while the noise in each decision's advantage is ~±1. Learning precise
  placement and timing from the outcome alone takes enormous numbers of games.
- **Exploration** is random noise around the IL policy. It rarely finds coordinated sequences (timed spells,
  counter-pushes).
- **Self-play drifts sideways**: it gets better against itself, not in general (non-transitivity).
- Each fix (lr, λ, batch) gives a step, then a plateau. The plateau is the signal to add a new mechanism, not to
  keep tuning PPO.

## 5. Long-term plan (discussed; goal: strong with any deck, i.e. real game understanding)

**Measure** by arena Elo against fixed milestones, and in the end by live trophies.

**Search during training, distilled into the network** (expert iteration, as in AlphaZero/KataGo) — the most
important item:
- At a card-play decision, copy the game, try ~8 candidate placements or timings, and play each ~5 s forward (2
  rollouts each) with the policy for both sides.
- Score the result with a local score (tower HP, elixir difference, value of units on the board), plus the critic
  as it improves.
- The best candidate becomes a training target. Credit becomes local: "from this exact state, tile A beat tile B".
- Live play stays the plain network: no search cost at play time. Search only runs in training data generation.

**Search cost estimate** (from measured numbers)
- One step (5 ticks) with features and both players ≈ 3.5 ms of CPU.
- A 5 s rollout ≈ 70 ms. A search (8 × 2 rollouts) ≈ 1.1 s.
- Searching 20% of the ~60 card plays per game adds ~13 s per game, against ~9 s today. That's ~2.5× slower: ~16k
  games/h, but ~200k searched decisions per hour (~5M a day; the whole IL dataset was 6M).
- Engine-only rollouts (no further plays) are ~4× cheaper, for screening candidates.

**Search risks**
- Copying engine state: a second engine in one process crashes. The plan is `os.fork()` of a collector, if the
  engine's threads survive it and the child reproduces the parent exactly.
- Scoring a 5 s future with a weak critic: hence the local score.
- Hidden information: rollouts see the opponent's true hand. Randomise their unseen cards consistently with what has
  been seen.
- Rollouts assume the opponent plays like our policy.

**De-risking steps**

| Step | What | Cost |
|---|---|---|
| 1 | Fork test: fork mid-game, step both copies with the same plays, compare states, time the fork | ~1 day |
| 2 | Search agent vs plain policy, ~100 arena games. Only worth distilling if it wins clearly (≥65–70%) | ~1–2 days |
| 3 | Searched collectors: train the policy on search choices + PPO | ~1 week |

**Step 1 result: fork works** (`~/cr-ppo/diag/fork_test.py`, run inside the learner job)

| Check | Result |
|---|---|
| A forked child steps normally | Yes, no hang (the game library's threads are idle between steps) |
| Two children, same plays | Identical for 200 steps (50 s of game) |
| A child with different plays | Diverges |
| Child vs parent replaying the same plays | Exact match |
| Parent after 90+ forks | Unchanged |
| Fork | ~12 ms |
| One engine-only 5 s rollout | ~37 ms including the fork |
| 16 rollouts at once | Fine |

**Step 2: placement search vs the plain policy** (`~/cr-ppo/diag/search_arena.py`)
- Checkpoint: update 470. 200 paired games.
- The searcher plays like the policy (same card, same timing). When it plays a card, it tries the policy's top 8
  tiles plus the sampled one, each forked and run 8 s forward with no further plays from either side.
- Score: tower damage (the shaping potential) + 0.3 × (our units' HP − theirs, in princess towers).
- It never uses the opponent's hand. Opponent plays still in flight but not yet visible are removed from the copies.

- **Result: search wins 68% ± 7%** (108–50 over 158 games) against the plain policy, with about twice the crowns
  (1.11 vs 0.51 per game). It changed the policy's tile in ~52% of its card plays. No failures.
- Read again on 20 October: the plain policy here sampled at temperature 1. Sharpening alone (temperature 0.1, no
  search) wins ~68% against the same policy at temperature 1 (section 9), so this result doesn't show that search
  adds anything beyond picking the policy's preferred tiles. Search vs the sharpened policy is the test that would.

**Step 3 attempt (16 October): distilling search into the network — collapsed, rolled back**
- Collectors searched 20% of the learning seats' card plays (~8 searches and ~6.4 labels per game). The tile head
  learned toward softmax(score / 0.02) over the candidates, with weight 0.5 next to the PPO loss.
- Speed: ~19–20k games/h, versus ~36k without search.
- **Within ~30 updates the policy collapsed:**
  - score vs the IL model +0.62 → −0.25;
  - vs the update-100 to update-500 snapshots, −0.75;
  - agreement with the search's best tile only 0.29 → 0.32;
  - distance from the IL model 0.09 → 0.13.
- Likely cause: the search loss (~2.4) dominated the gradients. It pulled the network shared by the tile, card, timing
  and memory parts, while barely teaching the tile head. Its targets are also noisy (near-ties, 8-second view).
- A bug in the label-to-tile mapping is not ruled out yet.
- Rolled back to update 510 with search off. The collapsed run is kept in `runs/ppo-4/search-collapse/`.
- **Before any retry:**
  - check offline that the labels match the rollouts;
  - test distillation with only the tile head trainable, then an arena of that network alone vs update 510;
  - retry only with search gradients limited to the tile head, weight ~0.05, and decisive labels only.

**Other levers**
- **A league** (AlphaStar-style): a main agent plus 1–2 exploiters that only learn to beat it, reset to the IL model
  periodically. This stops sideways drift.
- **Better signal**: pretrain the critic offline on the 176k human games; UPGO (reinforce only plays that turned out
  better than expected); bigger batches.
- **Humans in the loop**: IL variants fine-tuned on top-player replays as league opponents. ~20 live games every few
  days; fix what doesn't carry over to live play (refused taps, delays, held taps).
- **Deck specialisation** is the fastest way to ladder strength, but conflicts with the any-deck goal. Keep it only as
  a fallback.
- **Search dropped (16 October).** Passive rollouts ignore the opponent's reply, and overriding the policy with their
  result hurt it. Labels are expensive: ~9k per hour on 4 CPUs, against ~10M PPO decisions per hour. The offline
  label generators were stopped at ~5k labels.
- **UPGO (16 October, from update 550)**: `upgo` = 0.5 adds AlphaStar's UPGO advantage to the usual one. It follows
  the actual game while it went better than the critic expected, and uses the critic's estimate after that. Rollback:
  `latest-before-upgo.pt`. Judge by `score.v000100`–`v000500` after ~50–80 updates, then an arena vs update 510.

## 6. Scaling up (16 October)

- PPO stopped at update ~580 (UPGO: flat to slightly worse, entropy falling). `latest.pt` kept.
- The 1.26M model's perception (units and board) is only ~0.1M parameters. The IL run showed no overfitting (train 5.17,
  validation 5.13), and the user's 20k-game pilot favoured the larger network.
- **Medium IL run**: 3.4M parameters (perception 0.77M). card 64, object 96, 3 attention layers, convs 96 × 6, core 384.
  2.7 epochs, learning rate 3e-4 falling linearly to 3e-5 (`final_lr` 0.1), 3 A100s, ~4.8 h. Run `~/il-runs/medium-1016`.
- Baseline to beat (small IL, validation): loss 5.13, tile top-1 0.263, card accuracy 0.688.

**The tile "collapse" was bf16 rounding (found 16 October)**
- From ~66k steps the medium run's tile scores got worse while card choice improved; at ~123k the tile loss jumped
  2.9 → 5.0.
- Cause: softmax ignores a constant added to every tile, so the tile scores drifted together, to ~2,000 by 138k. At that
  size bf16 (~3 significant digits) rounds neighbouring tiles to the same score: 85% of plays had a tie for the top tile.
- The network itself was fine. Scored in fp32, the 138k checkpoint had the best tile top-1 yet (0.29), only
  overconfident.
- The small model was hit too (49% ties). Its true tile top-1 is ~0.275, not the 0.263 measured. Every PPO run so far
  sampled placements from these rounded scores.
- Fix: the tile head always runs in fp32 (`Policy.tiles`, with a test). The run resumed from its 105k checkpoint.

| Validation / training | Loss | Tile loss | Tile top-1 | Tile top-5 | Card acc |
|---|---|---|---|---|---|
| Small, final (validation, bf16 tiles) | 5.130 | 2.85 | 0.263 | 0.611 | 0.688 |
| Medium @78.6k (validation) | 5.225 | 2.867 | 0.249 | 0.602 | 0.669 |
| Medium @130k, fixed (training rows) | 4.967 | 2.704 | 0.290 | 0.644 | 0.696 |
| **Medium, final @207.5k (validation)** | **4.864** | **2.643** | **0.299** | **0.657** | **0.702** |

**Live (16 October, Nulls ladder, `--queue`)**: the 171k checkpoint went 9–4 over 13 games (3–0 four times; every loss
by 1–2 crowns) before the emulator crashed. The user, watching: "way, way better" than the small model, it defends and
attacks. Final weights: `~/cr-live/models/medium-1016-final/medium-1016-final.pt` (no optimizer state).

**Lessons**
- Watch every head separately: the total loss kept falling while placement degraded.
- When one head goes bad and the others don't, suspect numerics first: dtype and magnitudes.

## 7. PPO from the medium model: ppo-m1 (16–17 October)

**Setup**
- Init and KL reference: the medium IL model.
- The elixir mask is reworked: a card the client can't pay for gives its share of PLAY to WAIT.
- Batch 1024, λ 0.99, KL weight 0.01, a critic-only warm-up of 60 updates, lr 2e-5. lr was raised to 4e-5 at v430
  with no instability: approx_kl ~0.002, entropy ~0.195.
- ~24–30k games/h on 87 collectors.

**Arenas** (paired seats, 192 games)

| Match | Win rate |
|---|---|
| v100 vs IL | 78.6% |
| v300 vs IL | 93.8% |
| v300 vs v100 | 76% |
| v400 vs v350 | 56.2 ± 7.0% |
| v500 vs v400 | 44.5 ± 3.6% (191 games) |

**Live (Nulls ladder)**
- v350: 42–14 over several sessions.
- v400: 12–12. It stalls in league 4, where games are mostly decided 1–0.

**Diagnosis of the plateau** (17 October)
- **Tiny updates spread over a huge task.** approx_kl is ~0.002. At ~1k games per update, each of the 31k deck
  matchups has been played about once.
- **The KL pull to IL may cap it.** kl_ref has sat flat at ~0.06 for hundreds of updates: the reward pushing away
  and the KL pulling back balance out. This is the same signature as ppo-4's v100–180 stall, which a lower KL weight
  broke.
- **A noisy critic** (win_ev ~0.32) over ~1000 decisions per game: credit assignment is weakest in exactly the
  close games it now loses.
- **Self-play blind spots**: the opponents are itself and its last 6 snapshots.
- **The simulator-to-live gap** is unmeasured.

**Search, re-read from the logs**
- As a player, the 8-second placement search beat the plain policy 68 ± 7%. So short-sightedness is not shown.
- Distilling it collapsed because of how it was trained: its loss swamped the shared network, and agreement with the
  search's choices barely rose (0.29 → 0.32).
- The planned careful retry never ran: gradients to the tile head only, weight ~0.05, decisive labels only, after an
  offline check of the labels.

**Order chosen**
1. Anchor the KL to a recent snapshot (v400) instead of IL, using `ppo.py learn --reference`. Judge it by an arena
   against v400 after ~100 updates.
2. Exploiters (AlphaStar-style league) for robustness across decks and opponents.
3. Bigger batches with more CPUs, and the careful search retry, later.

## 8. The improvement loop: gen-large (from 19 October)

**Why.** Every earlier run needed someone watching it: a plateau or a drift was only seen by reading metrics. The
paired-fork test (19 October) also showed that full-game fork labels are about as noisy as the critic (mean
|G(a) - G(a')| ~0.55, against ~1 for unrelated games), so forks and search were set aside (kept outside git in
`attic/`) and plain PPO runs inside a loop that only keeps a model once it has proved better.

**How it works** (`loop.py`, on the login node; the heavy work stays in the jobs)
- A frozen **champion** (at first the large IL model) is also the KL reference; the learner is the **challenger**.
- Every 20 updates the snapshot goes through a **gate**: evaluation games played by the collectors themselves,
  frozen against frozen (400 against the champion; 200 against each held-out benchmark: medium IL, ppo-m1 v400 and
  the E-Barbs specialist v1100). Promotion needs >= 55% against the champion and a benchmark mean no more than
  3 points below the champion's.
- A promoted champion becomes the new KL reference and a permanent league opponent.
- **Rollback** to the champion's full checkpoint after two gates below 45% or a collapsed policy (entropy < 0.05);
  **stop** after `stall_after` gates without a promotion (10 at first; raised to 14, then 17, to give each
  setting change a fair trial), or 72 h. A minibatch whose gradient is not finite is skipped.
- Jobs that end or hang are resubmitted; `RUN/report.md` is rewritten every minute.

**Setup.** Large IL policy (8.0M) and critic (its own copy); batch 1024, lr 3e-5, one epoch, minibatch 4, critic
2 epochs, KL weight 0.01, λ 0.99, margin 0.2, critic warm-up 10 updates. Learner 4 A100, actors 50 collectors and 4
decision servers on 4 RTX 5090 (the QoS allows 16 CPUs per partition).

**Throughput.** ~17k trained games/h at first (critic warm-up: ~19–20k), ~14k by 20 October (games got longer:
~760 → ~890 decisions per seat). The learner and actors are balanced outside gates; during a gate the evaluation
games take actor time and the learner waits.

**Promotions (19 October).** v20 (66.2% vs IL), v40 (67.4%), v60 (62.9%), v120 (58.0%) and v200 (57.2%, 22:57).
Benchmark mean: IL 36.3% → v60 71.3% → v200 72.4%. v80, v100 and v140–v180 were kept (v100 beat v60 58.6% but fell
3.6 points on the benchmarks).

**First results (19 October).**
- The first gate found a serving bug: the decision servers kept one LSTM-state tensor sized for the large model, so
  the medium benchmarks crashed them. Fixed (one state tensor per LSTM size) and covered by a test.
- **v20 promoted** after 10 policy updates: 66.2% against the large IL model over 447 games; benchmark mean 51.8%
  against IL's 36.3% (medium IL 79.9%, ppo-m1 v400 34.4%, E-Barbs specialist v1100 41.2%).
- Large IL itself scores only ~20% against ppo-m1 v400, so beating v400 is the next milestone.

## 9. Deployment temperature, and the plateau after v200 (19–20 October)

**The finding.** The policy picks a card and then a tile from softmax distributions. Played at a lower temperature
for those two choices (0.1–0.25: almost always its preferred card and tile), every model is far stronger. When to
play stays sampled as trained. The user found this live:

| Model | Temperature | Live result |
|---|---|---|
| v120 | 1 | 2–3 (league 5) |
| v120 | 0.25 | 14–4–1, reaching league 6 |
| v200 | 0.25 | 20–0 on Trophy Road; 9–4–1 in league 6 |
| v200 | 0.1 | 4–2 in league 6 (6 games) |

The simulator agrees (paired arenas, 60–200 games each):
- v120 against the large IL model: 87–88% with IL at temperature 1, 62–69% with IL at 0.1. Sharpening alone makes
  IL much stronger, so much of PPO's apparent gain over IL was sharpening.
- v120 at 0.5 / 0.25 / 0.1 against itself at 1: 66.3% / 55.7% / 68.5%.
- The user's bar, ≥ 75% against ppo-m1 v400: v120 at 0.1 scores 69.0 ± 3.3% against v400 at 1, and 55.0 ± 3.5%
  against v400 also at 0.1.

Why: the IL policy imitates many players, and its preferred choice is better than a random draw from its spread.
PPO's small steps (approx_kl ~0.002 per update) barely narrow that spread, and an entropy bonus keeps it. Systems
like AlphaZero likewise play their best move at test time, not a sample.

**What changed in training.**
- 19 October 23:32: `frozen_temperature` 0.1. Opponents, gates and benchmarks play sharpened, so a gate measures
  deployed strength. The earlier gates (at temperature 1) are kept in `evals-T1.jsonl`.
- 20 October 07:30: `learner_temperature` 0.5. The learner samples at 0.5 and PPO scores log-probabilities at 0.5.
  Three gates, all ~50% against v200: no effect.
- 20 October 13:49: `learner_temperature` 0.1 with `learner_mix` 0.1. The learner plays as deployed, except that
  10% of its untempered policy is mixed in so it still tries alternatives, and PPO scores that mixture exactly. The
  critic then also values the policy that is deployed.

**The plateau.** From v220 to v380, nine gates were all within 47–53% of v200, with benchmark means of 67–74%
(v200: 70.3%). PPO kept changing the sampled policy (kl_ref to v200 grew to 0.035), but not the top choices that a
sharpened policy plays. Training the deployed policy directly (13:49) is the response; the next gates decide it.

**Exploiter.** `gen-exploit` trains only against gen-large's current champion (`against`). Over ~400 updates its
score against v200 rose from −0.47 to about −0.22, while it samples at 0.5 against a champion at 0.1. It has found
no holes yet. Its snapshots would join gen-large's league only if it did.

**Hero ability.** The model presses its ability ~0.04% of the times one is ready, while humans press ~1%. The press
works in the simulator and live, but neither pressing as soon as ready (36% against the plain model) nor at the IL
model's timing (48.7%) did better, so this was set aside. Open: the 1% exploration floor on presses
(`ABILITY_EXPLORE`) adds random presses to training and gate games.

**Next, if the gates stay flat.** Test whether search beats the sharpened policy, before building search
distillation (section 5); a stronger critic (`win_ev` has stayed ~0.3); exploiter snapshots in the league once the
exploiter finds holes.
