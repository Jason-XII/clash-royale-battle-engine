# Training notes: imitation (IL) and PPO

What we decided, measured and discussed while training: IL from human replays, then PPO from the IL model. It's
written in order. Sections 1–5 cover the small model's PPO runs (`~/cr-ppo/runs/ppo-4`, 15–16 October), section 6
covers scaling up to the medium IL model, and section 7 covers PPO from it (`~/cr-ppo/runs/ppo-m1`). Sections 8 and
9 cover gen-large under the improvement loop, and the deployment-temperature finding. Section 10 diagnoses the
plateau after v200 (search, distillation, critics), section 11 redesigns the network to be cheaper and better, and
section 12 covers PPO on the new network (gen-r3), the search experiments on it and the first live games.
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

## 10. Diagnosing the plateau after v200 (20–21 October)

**Stopped.** gen-large stopped on 20 October at 17:14, at the user's call ("drop models after v200 and focus on what
works"). Its gates stayed level with v200 after the 13:49 change too. Every file is kept, and v200 stays the
champion. gen-exploit also stopped, at about v445, scoring −0.20 against v200 with no holes found. It had started
from gen-large's v120 checkpoint, the same lineage as v200, which may be why it found nothing. A future exploiter
should start from the IL model, as AlphaStar's do.

**Step 1: an honest baseline.** All models play sharpened (card and tile T 0.1), paired seats, 400 games each.

| Match | Score |
|---|---|
| v200 vs large IL | 83.3 ± 3.7% |
| v400 vs large IL | 81.2 ± 3.8% |
| v400 vs v200 | 47.2 ± 4.9% |
| v200 vs ppo-m1 v400 (the user's bar is 75%) | 54.2 ± 4.9% |

PPO learned real decisions, not just sharpening. But nothing after v200 improved, and the 75% bar is far away.

**Step 2: what did PPO change?**
- Head transplants: v200's heads put into the large IL model, played against plain IL, ~450 games each.
  - v200's tile head: 51.1%.
  - v200's card head: 56.0%.
  - v200's timing head: 74.1%.
  - All of v200: 83.3%.

  **PPO's gain is mostly when to play.**
- The same 400 human validation games, read by both models:
  - v200's preferred card differs from IL's 21% of the time; its preferred tile differs 36% of the time, 4.3 tiles
    away on average.
  - Mean P(play): v200 0.054, IL 0.035, humans 0.039.

  PPO moved many tiles without making them better.

**Step 3: is there a stronger teacher? Search.** Placement search runs at every card play. It takes the policy's
card, then tries the policy's own tile and its top 8 tiles in forked engine copies. Each copy plays 8 seconds with no
further plays. Each end state is scored by tower potential + 0.3 × material, and the best tile is played. The
rollouts are passive (no one plays), so the opponent's hidden hand never enters.
- Placement search vs plain v200 (both T 0.1): **65.5 ± 4.6%** over 412 games. It changed 55% of tile choices.
- Variants (~200 games each, vs plain v200):
  - Searching the card as well: 52.5%. Card switches judged on an 8-second passive view hurt.
  - Horizon 240 ticks: 66.5%. Horizon 100: 65.0%.
  - Material weight 0.6: 61.5%.
- With the end state scored by the large IL value head (own view only) instead of the formula:
  - Tile search: 69.6 ± 6.3%.
  - Card + tile search: 65.2 ± 6.5%. A learned value fixes card search.
  - Value head + formula: 70.1 ± 6.3%.
- Responsive search (true state, the opponent answers, 2 rollouts): 52 ± 15% after 44 games. Stopped: it slowed
  the learner.
- A bug was found and fixed in the search scripts. The rollout delay was `sum(PLAY_DELAY) // 2` = 1113 ticks, a
  leftover from when PLAY_DELAY was a (min, max) pair. No candidate ever landed, so every score tied. The
  production code was fine.

**Teaching the search to the network failed.** The labels came from v200 self-play at T 0.1 with every card play
searched: ~3,200 games, 268k labels, 52k of them decisive. Only `film` and `tile` were trained; training the shared
network had caused the 16 October collapse.

| Round | Loss | Arena vs v200 (both T 0.1) |
|---|---|---|
| 1 (d1a / d1b) | cross-entropy to the search's best tile, KL 0.1 | 45.7 ± 3.1% / 42.9 ± 2.2%: worse |
| 2 (d2a / d2b) | pairwise: the better tile of each candidate pair scores higher, KL 1.0 / 0.3 | 48.8 ± 2.2% / 48.7 ± 2.2%: no gain |

- Round 1 moved 40% of top tiles, but only 22% of the moves went to the search's best. It taught "move away from
  your tile" rather than the search's choice.
- The learning curve: 400 / 800 / 1600 / 2864 training games give held-out pair accuracy 60.2 / 60.7 / 61.1 /
  61.3%. v200's own head scores 56.5%.
- Tile adapters (extra residual convs between the card conditioning and the tile scorer): 61.3% / 61.4%.

More labels and a bigger head barely help, so the frozen network's features are the limit. The network can't
represent what the search finds. This led to the redesign in section 11.

**Critics.** Explained variance of the game result. Human games are 300 held-out IL validation games; bot games are
600 seat-games from v200's label games.

| Predictor | Human | Bot |
|---|---|---|
| PPO critic, v430 / v200 (sees the opponent's hand) | 0.186 / 0.182 | 0.278 / 0.289 |
| Large IL value head (own view only) | 0.274 | 0.321 |
| v200's value head | 0.244 | 0.294 |
| Offline critic (below) | 0.260 | 0.327 |

The value learned from human outcomes beats the PPO critic even on bot games, so PPO's advantages were poor. The
offline critic is `ppo.Critic` with the large IL network plus the opponent's private inputs. It was trained with
BCE on the result at every decision of the 176k human games; each archive has both players' views, aligned decision
by decision. It beats the PPO critic but only matches the IL value head.

**gen-critic.** PPO restarted from v200 with the offline critic as the only change, from 20 October 23:59.
- The first night was lost to setup problems:
  - With no 4-GPU node free, the learner ran on 2 A100s. Minibatch 8 per rank ran out of memory 5 times, and each
    restart fell back to v200. Fixed with minibatch 4 and lr 1.5e-5.
  - A decision server hung, which starved the learner. Killing it let the supervisor restart it.
- The first gate, v220: 49.6% against v200 over 538 games. Stopped by the user on 21 October.

**Deployment settings** (paired games against v200 itself, ~400 each):

| Setting | Score vs v200 |
|---|---|
| Card / tile T 0.05 vs T 0.1 | 55.4 ± 4.8% |
| Greedy (T 0) vs T 0.05 | 48.0 ± 4.9% |
| When-to-play temperature 0.7 / 1.3 (instead of 1) | 48.7% / 44.4 ± 4.8% |
| v200 vs v120 | 51.2 ± 4.9% |

Deploy at T 0.05; timing stays sampled at 1. `sharpen()` now supports greedy T 0.

**Which decks suit the model** (`diag/deck_analysis.py`: 9,335 simulator games between near-v200 players, each game
credited to both decks):
- Pressure and beatdown decks suit it; precise siege, chip and cycle decks don't. Placement is its weak spot.
- By win condition: Mortar 59.6 ± 3.3%, Lava Hound 57.0%, Giant Skeleton 56.7%; Hog 48.0%, X-Bow 41.2%, Goblin Drill
  38.6%.
- Best well-measured deck: Minions, Rascals, Cannon Cart, Skeleton Balloon (evolved), Berserker (hero), Mortar
  (evolved), Fireball, Barbarian Barrel: 64.9 ± 6.0%.
- This measures the model against itself, not against humans, so it's a proxy.

## 11. A cheaper, better network (21 October)

**v200's network** (8.06M parameters, about 1.64 G multiply-adds per decision):
- Objects go through a 6-layer transformer (160 wide) and are summed onto a 32 × 18 grid.
- The grid goes through 12 plain 3 × 3 convs at 160 channels, with no residuals and no normalization.
- The core (grid mean and max, an 8 × 4 map, hand, flights, scalars → MLP → LSTM 448) feeds the heads.
- The tile head is the card-conditioned grid (FiLM) plus a 1 × 1 conv.
- The convs are about 98% of the compute; the transformer is about 2%.

**How the options were judged.** Each option is one IL run on the same 30k random human games, 1 epoch, scored by
the loss on the same 1,000 held-out games. Lower is better, and there is one seed each. Short runs can favour smaller
networks, so the winner still has to prove itself on the full data.

**Ablations A–F** (the medium model: 96-wide transformer with 3 layers, a 96-channel board network, core 384):

| Run | Change | Loss | Tile top-1 | Card acc |
|---|---|---|---|---|
| A | none | 6.019 | 21.2% | 56.6% |
| B | residual board network (`--res_blocks 1`) | 5.886 | 22.2% | 57.5% |
| C | plays in flight drawn on the board | 6.038 | 21.0% | 55.7% |
| D | 2 residual blocks in the tile head | 5.850 | 22.6% | 58.1% |
| E | 16 × 9 coarse map instead of 8 × 4 | 6.006 | 21.5% | 57.1% |
| F | all of the above | **5.795** | 23.7% | 57.8% |

B and D help. C and E add nothing.

**Compute ablations G–J** (the large trunk; only the board network varies):

| Run | Board network | Loss | Multiply-adds per decision |
|---|---|---|---|
| G | v200's: 160 ch × 12 plain convs | 5.923 | 1.64 G |
| H | 160 ch, 3 residual blocks | 5.876 | 0.98 G |
| I | 128 ch, 4 residual blocks | **5.832** | 0.84 G |
| J | 96 ch, 3 residual blocks | 5.903 | 0.42 G |

Residual blocks with less compute beat v200's network. Depth beats width: I beats H with fewer channels.

**Measured speed** (`diag/bench-1021/bench_arch.py`, real inputs, bf16, decisions per second):

| Network | Serving, A100 / RTX 5090 | IL training, A100 / RTX 5090 | Training memory |
|---|---|---|---|
| v200 | 24.7k / 25.3k | 8.3k / 8.2k | 9.1 GB |
| I | 25.7k / 30.0k | 8.8k / 10.1k | 17.7 GB |
| J | 37.1k / 45.3k | 12.1k / 15.4k | 11.6 GB |
| 64 ch, 3 blocks, 1 tile block, transformer 128 × 4 | 48.2k / 57.8k | 16.6k / 23.4k | 7.7 GB |

The speed-ups lag the compute cuts. A GPU profile of J's serving step (`prof_arch.py`):
- Convs take only 25% of GPU time.
- GroupNorm, with the fp32 casts autocast adds around it, takes ~38%.
- Elementwise operations take ~20%.
- v200 loses 15% to NCHW↔NHWC layout permutes.

GroupNorm is also why the residual networks use more training memory.

**FirstLight as a reference** (`firstlight-cr/native_runner/training/v4/model.py`). Its network has 12.7M
parameters, but only 0.29M of them are in the board network (64 channels, a stem plus 3 residual blocks with
GroupNorm), for an estimated ~0.4 G multiply-adds per decision. It puts its capacity into input semantics instead:
- a transformer with attention biases from relation types and unit-to-unit distances;
- cards described by stats and mechanics;
- units grouped into up to 48 tokens.

Its tile head is the same as ours. It confirmed residual blocks and a small board network. We tested its distance
bias (M below). Card stats need a per-card stats table we don't have yet (only costs and rarities).

**Round 3** (large trunk; K is the base and each other run adds one change; time = seconds for the same steps on one
A100):

| Run | Change | Loss | Time |
|---|---|---|---|
| G | v200's network, for reference | 5.923 | 2,923 |
| K | 96 ch, 4 residual blocks | 5.870 | 2,295 |
| L | K without GroupNorm (each branch gated, from 0) | 6.044 | 1,864 |
| M | K + distance bias in attention | 5.832 | 2,874 |
| N | K + 1 residual block in the tile head | **5.805** | 2,375 |

- Removing GroupNorm is fast but clearly worse, so it's dropped.
- The distance bias helps a little but takes back the whole speed gain, so it's left out for now.
- The tile block is the best change and nearly free. It also helped on the medium trunk (D).

**Decision: N.** The large trunk (transformer 160 × 6, LSTM 448, same heads) with a 96-channel board network of 4
residual blocks and 1 residual block in the tile head.
- About 0.5 G multiply-adds per decision, roughly 3× less than v200.
- 0.12 lower loss than v200's network.
- The PPO learner scores tiles only on decisions that play a card, so there the tile block is nearly free. Serving
  still scores tiles on every decision: `sample()` runs inside the decision servers' CUDA graphs, which need fixed
  shapes.
- Measured (A100, eager, batch 2048): serving 27.2k decisions/s vs v200's 24.5k, IL training 10.7k vs 8.3k, training
  memory 13.8 vs 9.1 GB. GroupNorm and elementwise overhead hide most of the compute cut. The extra memory forced the
  PPO minibatch from 4 to 2 games per GPU.

**Full-data run.** `~/il-runs/r3-1021`, 4 A100, code `src-r3-1021`, with the large IL model's recipe (4.5 epochs,
lr 3e-4 decaying to a tenth, 2000 warm-up steps). The first job (106537) died at step ~71.7k when Inductor filled the
node's /tmp; it resumed from step 71k as job 106920 with `TORCHINDUCTOR_CACHE_DIR` in the home directory.

| Held-out (same games) | Loss | Tile loss | Card acc | Tile top-1 | Value loss |
|---|---|---|---|---|---|
| Large IL (`large-1018`) | 4.6438 | 2.498 | 71.8% | 32.2% | 0.726 |
| N (`r3-final.pt`) | **4.6155** | **2.476** | **72.1%** | **32.5%** | 0.733 |

Arena, 400 paired games each (`~/cr-ppo/runs/gen-r3/arena-il/`):

| Match | Score |
|---|---|
| N IL vs large IL, both T 0.1 | **55.2 ± 4.9%** |
| N IL vs v200, both T 0.1 (large IL vs v200: 16.7%) | 24.0 ± 4.2% |
| N IL at T 0.05 vs T 0.1 | 51.7 ± 4.9% |

The offline critic on N (`critic-offline/run-r3`) matches the large one: explained variance of the result 0.252 on
human games and 0.378 on bot games (large: 0.254 / 0.378). The IL run, the check, the arena, the critic and the
start of PPO (section 12) ran unattended overnight (`runs/gen-r3/auto.sh` on the login node).

## 12. gen-r3, search on the new network, and live play (22 October)

**gen-r3** (`~/cr-ppo/runs/gen-r3`, from 22 October 03:31). PPO under loop.py from N IL, with the offline critic
(`--critic-weights`), the user's reward changes (king tower weight 3 → 1; the elixir-overflow penalty only while
enemy units are on the board), learner 4 A100 with minibatch 2, lr 2e-5, learning at T 1 and frozen copies at T 0.05.

| Gate | vs champion | Games | Benchmark mean | Medium IL | ppo-m1 v400 | spec v1100 | v200 | Result |
|---|---|---|---|---|---|---|---|---|
| (N IL) | | | 39.9% | | | | | |
| v20 | 64.5% vs IL | 485 | 51.3% | 76.3% | 40.5% | 50.2% | 38.1% | promoted |
| v40 | 56.5% vs v20 | 441 | 61.2% | 83.7% | 51.1% | 64.8% | 45.0% | promoted |
| v60 | 57.5% vs v40 | 588 | 66.4% | 82.0% | 62.3% | 73.3% | 48.1% | promoted |
| v80 | 48.0% vs v60 | 500 | 65.4% | 84.9% | 56.9% | 68.4% | 51.5% | kept v60 |
| v100 | 54.6% vs v60 | 551 | 68.5% | 87.0% | 63.3% | 72.2% | 51.6% | kept v60 |
| v120 | 54.8% vs v60 | 588 | 67.3% | 90.4% | 58.0% | 67.1% | 53.6% | kept v60 |

- v60 reached v200's level about 4 hours in; gen-large needed its whole run for that.
- v100 and v120 together score ~54.7% against v60 over ~1,100 games (about 3 standard errors above 50%), but each
  misses the 55% promotion bar. The champion is also the KL reference, so a stale champion pulls the learner back.

**Why the cheaper network did not train faster.** gen-r3 trains about 16k games/h, the same as gen-large.
- Learner: 5,690 decisions/s against gen-large's 5,600; per update, critic 144 s and policy 102 s (gen-large 142 /
  101). The isolated benchmark gives the new network only 1.29× in training, and the halved minibatch (the residual
  blocks need ~1.5× the activation memory) uses that up. The critic, 2 epochs over the same-sized network plus the
  opponent's inputs, is 53% of each update. The GPUs run at 99%.
- Actors: limited by the QoS's 16 CPUs (50 collectors at 25–30% of a core and 4 decision servers at ~75%), not by
  the GPUs (52–61% busy). A collector waits 9.3 ms of each ~12 ms decision for its server.
- The user decided not to change this.

**Search on the new champion.** Tile search judged by the large IL value head (`search_v.py --judge il`), v60 with
search against plain v60, both T 0.05: **70.3%** over 408 games (on v200 at T 0.1: 69.6%). It changes 77% of tile
choices, gaining 0.078 value on average when it does. The new network still leaves about 20 points in placement.

**Why PPO doesn't learn placement.** A better tile is worth about 0.08 value; the game result's leftover noise is
about 0.8 (the critic explains ~30% of results), and PPO sees one sampled tile out of ~576 with no alternative to
compare. When to play is a yes/no choice every 5 ticks, so it gets far more signal: in section 10, v200's timing head
alone was worth 74% and its tile head 51%.

**Distillation on the new network** (`~/cr-ppo/distill-r3/`; local copies in `attic/search-r3/`). Pair accuracy: on
held-out games, the share of the search's candidate pairs (scores more than 0.02 apart) that the head orders the same
way. Gain captured: of the value the search's best candidate adds over the policy's own top tile, the share the
trained head's top tile adds.

| Run | Labels | Anchor (KL) | Pair accuracy | Gain captured |
|---|---|---|---|---|
| v200 tile head (section 10) | formula, v200 games | 1.0 | 56.5 → 61.3% | |
| large IL tile head | formula | 1.0 | 55.6 → 60.2% | |
| N IL tile head | formula | 1.0 | 56.0 → 60.9% | |
| v60 tile head | formula | 1.0 | 57.0 → 61.5% | |
| N IL, whole network | formula | 1.0 | 56.0 → 61.7% | |
| N IL tile head (c-h) | formula | 0.1 | 66.0% | 8.5% |
| N IL, whole network (c-w) | formula | 0.1 | 67.4% | 22% → 3% (noisy) |
| v60 tile head, listwise (L-h) | value-judged, v60 games (399) | 0.1 | 58.3 → 62.7% | 10.2% |
| v60 whole network, listwise (L-w) | value-judged, v60 games (399) | 0.1 | 58.3 → 63.2% | 9.0% |

- The architecture doesn't move it, and training the whole network adds about a point.
- The anchor was holding pair accuracy at ~61%: at KL 0.1 it reaches 66–67%. But the head's own top choice still
  captures only ~10% of the search's gain (on the value-judged labels the search adds 0.060 value per card play; the
  head 0.006). It learns to order candidates rather than to pick the best one.
- The listwise loss trains the head's choice among the candidates towards softmax(score / 0.05), each label weighted
  by its score spread (`distill2.py --loss listwise --variant jmean`).
- Their arena against plain v60 (`arena-L/`, cut short at ~300 games each): L-h 46.2 ± 5.7%, L-w 43.6 ± 5.5%. No gain,
  as the ~10% gain captured predicted.

**Are the labels noisy?** `label.py` scores every candidate under several variants and `noise.py` measures how often
two variants agree on a pair (v60 self-play, 2,551 labels, pairs more than 0.02 apart):

| Variants compared | Formula labels | Value-judged labels |
|---|---|---|
| Card lands at tick 22 vs 25 | 89.5% | 91.5% |
| Horizon 160 vs 200 ticks | 86.7% | 86.5% |
| Judge: large IL vs N IL | | 87% |
| Formula vs judge | 70–75% | |

The labels are reproducible, so label noise doesn't explain the ceiling. The formula and the judge disagree on a
quarter of the pairs, and every distillation before 22 October used the formula's labels (the 65% player, not the
70% one).

**New labels.** `label.py` on v60's own games (every card play searched, landing at tick 22, horizons 160 and 200,
both judges; `labels-r3/`, Students job 108164): about 330 games and 33k labels per hour.

**Live** (`~/cr-live/models/gen-r3-v100/`, its own source copy: older copies can't load the new network). gen-r3
v100 at T 0.05: 6–1 in battles the user started, then 3–6 on the ladder in league 6 with auto-queue (v200 at T 0.25
went 9–4–1 there on 20 October). The user stopped it.

**Assessment (22 October).**
1. Simulator gains aren't clearly reaching the ladder. Training opponents and benchmarks are all our own models; IL
   (human-like) opponents are ~1% of training games, so mistakes that humans punish are never punished in training.
2. Placement is the largest measured gain (search, +20 points), and no loss, label set or network size has taught
   more than ~10% of it so far.
3. The 55% gate blocks real ~5% improvements, and the stale champion stays the KL reference.
4. The deployment temperature was tuned only in the simulator. Live: v120 at T 0.25 14–4–1, v100 at T 0.05 3–6
   (small samples).

Proposed: promote at 53% with ~800 games against the champion (applied the same afternoon, see Gates below); ~20% human-like opponents (IL at
several temperatures) and an exploiter started from IL; a fixed batch of ladder games per candidate, testing T 0.25
against T 0.05 live; and if distillation keeps failing, search during live play (building an engine state from the
live reader), the only route so far to the measured 20 points.

**A tile head that reasons about units** (stage 1; `distill3.py --interact`). Each tile's query (the card being played,
the core, the tile) attends over the trunk's unit tokens, with logits from content and from the unit's offset to the
tile, so the card can pick which distances and directions matter. Its score is added to the tile head's, from zero.
Trained on v60 with the listwise loss on the same 761 games of value-judged labels (57k training labels), 4 epochs:

| Run | Pair accuracy | Gain captured, held-out | Gain captured, training games |
|---|---|---|---|
| B0: the tile head | 63.5% | 13.7% | 20.5% |
| B1: + the attention branch | 63.3% | 10.9% (14.2% after epoch 3) | 26.3% |
| B2: + the card's and units' stats in the branch | 63.2% | 12.9% | 23.4% |

The branch fits the training games better and the held-out games no better: on this many labels it memorizes. 1,017
labelled games are kept (`labels-r3/`) for a retry with more labels or with the branch trained on human data first.

**What the model sees** (common.py `decision_inputs` and `PlayerView`). Up to 128 objects (units, buildings, towers,
spells, projectiles), each with its card and form id, side, kind, exact position and movement, hp, deploy time, its
current target, 31 unit-state values (stun, slow, attack cycle and load, charge, dash, ability, shield, height, ...) and
up to 6 buffs; its hand (cost, affordability, evolution), the next 4 cards, ability buttons; plays in flight (its own,
the opponent's from 13 ticks before landing); the opponent's seen cards and when each is back in hand, their hero
cards and exact elixir; the clock and crowns; the legal tiles. The objects go through the transformer and are summed
into their tiles; the tile head conditions the whole board on the card with one scale and shift per channel. Missing:
what each card does (only ids and costs), which led to the next test.

**Card stats.** `card_stats.py` reads the build's own logic tables (compressed CSV and TOML) into 28 fields per id
(range, sight range, hitpoints, damage, DPS, speed, deploy time, attacks air / ground, targets only buildings, flying,
splash, spell radius and damage, lifetime, spawner, death damage, ...) for every card and every unit data id in the
vocabulary (evolutions and heroes take their base unit's): `assets/card_stats.json`, `--card_stats 1`.

| 30k games, 1 epoch, held-out | N (round 3) | N rerun | N + card stats |
|---|---|---|---|
| Loss | 5.8047 | 5.804 | **5.767** |
| Card choice | 0.959 | 0.959 | 0.949 |
| Tile | 3.112 | 3.090 | 3.071 |
| Card accuracy | 58.1% | 57.9% | 58.7% |
| Tile top-1 | 23.4% | 23.7% | 23.7% |

About the distance bias's gain, nearly free; mostly card choice, not placement. **r4** (N + card stats, full data,
r3's recipe) started on 22 October 14:48 on 2 A100 (`~/il-runs/r4-1022`, ~17.5 h, with a resubmitting watcher).

**Overlapping plays.** Our simulator and live client refuse a tap while one of the player's plays is still in flight,
but humans play their next card ≤ 20 ticks after the previous in 5.7% of plays (~1.8 per game; 3,000 player-games),
so the real game allows it; it was most of PPO's ~2 refused plays per game. `arena.py` can allow it per player
(`Engine.overlap`, `--a-overlap`; the pending plays' cost is reserved). v60 with it vs v60 without: 50.5 ± 4.8% over
418 games (refused 0.18 vs 0.58 per game, 36.2 plays each). Correction (22 October, code review): this test never
overlapped two cards. The sampler (`train.what_logp`) only lets a model wait while one of its cards is in flight,
so the engine option freed nothing but card taps during a pending ability press; that was the drop in refusals.
Overlapping card plays remain untested; since 22 October `--a-overlap` also lifts the sampler's wait
(`model.overlap`: a card yes, an ability press no, not the slot in flight, its cost reserved).

**Ability presses (22 October).** The user saw the model almost never press some heroes' abilities live. Findings
(`~/cr-ppo/diag/ability-1022/`):
- The simulator refused every card tap while an ability press was on its way (about 22 ticks), which the model could
  not see; live, the client takes them. 96% of pool decks have a hero or champion. In v140 vs v140, 0.42 card taps per
  game were refused this way. PPO learned from these taps as if they had been played, and saw presses cost more than they do.
- Humans press on 0.96% of the decisions with a ready ability (176,732 games; from 0.34% for Knight to 2.4% for
  Golden Knight). Live, models press 3 to 20 times less. Large IL pressed 0.45% (4 games), its PPO descendants
  v60–v200 0.16–0.22%, and ppo-m1 0.08%: most of the loss comes in PPO.
- In the simulator, a press works for all 24 hero and champion forms in the pool (the charge is spent). Abilities cost 1 to
  5 elixir, which no input shows.
- IL labels were late: `convert.py` labels a press at its execution tick, a card 22 ticks before it lands, and a press
  takes the same 20–26 ticks live. In two-hero decks (8% of player-games) every press was labelled controller 1,
  and 42% of those were dropped as not ready.

Fixes, in `~/cr-ppo/src-abilityfix-1022` (the running jobs keep their code): `Engine.step` blocks a card only for
another card and a press only for another press (refused taps in v140 vs v140: 0.65 to 0.22 per game; the rest
are repeated presses, harmless). `train.py --ability_labels 1` moves each press 4 decisions earlier (to the first
ready decision if the button lit up later) and onto the ready controller: all labels usable (94.3% before), 0.16%
dropped. Not tested yet: how much more pressing is worth in the fixed simulator.

**Review fixes (22 October, four bugs the user reported; same snapshot).** (1) Multi-GPU training kept the ranks'
Adam states apart: under DDP's Join a rank that ran out of minibatches skipped optimizer steps, and only the weights
are re-synced when Join ends (reproduced with two gloo ranks: equal weights, different Adam moments). PPO now gives
every rank the same number of steps (`ppo.schedule`: the most any rank has, the others repeating some of their own);
IL steps the ranks together and ends a pass for all when one runs out (`train.lockstep`). (2) A specialist (`deck`)
collector crashed on evaluation games (no learning seat); the candidate now plays the deck there. (3) loop.py could
promote a gate that timed out before any benchmark game, and cached a champion baseline with gaps; such a gate is now
inconclusive, and a baseline is kept only when complete. (4) `--a-overlap`, above.

**Second review (23 October, six issues the user reported; local repo only, not in any running snapshot).** None
changed a past run as far as measured: every gen-r3 and gen-r3-place update took its policy and critic steps, no
gate of gen-r3, gen-large or gen-critic timed out (38 gates), all benchmark names start with `b-`, and all 176,732
games in the IL data pass the winner and crown checks. (1) `schedule` (above) made every rank skip an update when one
rank had no fresh games; the learner now waits until every rank has some, and collector i sends to rank i mod ranks.
(2) Frozen models (champion, candidate, benchmarks, league) played with the learner's 1% ability-press floor, which
live play does not have: only the learner explores now, so gates measure the deployed policy. (3) The decision server
cached models by name, so two runs' v000100 were one model; it caches by checkpoint path, and loop.py refuses
benchmark or initial-champion names like v000100 from another run. (4) ABILITY2 pressed the first ready controller;
the press now carries its controller to execution. Live play still taps one button (controller 1's). (5) convert.py
kept a game whose tower hit points matched but winner or crowns did not; it now needs all three. (6) A timed-out gate
counted one benchmark game as a complete benchmark; promotion and the cached champion baseline now need
`bench_games` games per benchmark.

**Gates.** v140 was promoted (55.9% vs v60, benchmark mean 70.4% vs 66.4%) after v80–v120 missed the 55% bar by
fractions of a point; v160 then scored 48.3% against v140. The user lowered the bar to 53% with 800 games against the
champion: at 400 games a no-better model would pass 53% about 12% of the time, at 800 about 4.5%. loop.py reads
loop.json only when it starts, so it was restarted between gates (15:00); the jobs kept running and the loop's state
(champion, gates, jobs) carried over.

**Left running on 22 October, 15:10.** gen-r3 under the loop (champion v140; it stops after 10 gates without a
promotion or at 72 hours); the full-data r4 IL (N + card stats, ~17.5 h on 2 A100), with a watcher that resubmits it
if it dies and then starts r4's arena against r3 IL and v200 (both T 0.1).

