# Precise timing: when a play lands, live

Measured by Arron's side (clapha) on 2026-10-20/21: the official client (app 160402012, data 16.402.22) against
Supercell's servers, friendlies between two of our own accounts, about 450 recorded plays. **Not measured on Null's
Royale** (the August client, a different server): check there before relying on any of it.

## What the server does

- **Turns.** The server sends the battle in turns of 10 ticks, one every 500 ms. The newest tick the client has received
  (`battle+0x64` on 160402012; R) ends in 9 in every battle we read. The screen runs 7-19 ticks behind R.
- **A play's issue tick is set by the server**, by when the command reaches it, not by the client. A play lands (executes)
  at **issue + 21**. Our own command shows in the queue only with a server turn, never locally.
- **The stamp over one turn**, by when the touch goes up after turn R arrived: R - 17 ... R - 11 rising about one tick per
  50 ms (the network decides the exact step: 60 / 35 / 5% of ramp plays got the shown tick +1 / +2 / +3), then
  **R - 8 flat** until the next turn arrives (a clamp: any command that misses the next turn gets R - 8).
- So **one landing tick in every 10 is certain**: issue R - 8, landing R + 13, a tick ending in 2. A touch about 400-440 ms
  after where turn R would have arrived (the 10th percentile of `arrival - R x 50 ms` over the last 60 turns, plus R x
  50 ms) got R - 8 in 109 of 110 plays. When turn R itself arrives 40 ms or more late, the slot is less sure (18/22, 21/23, 17/19 at 400 / 420 / 440 ms).

## What it means for training

The landing delay from a decision is not random: it depends on where in the server's turn the touch goes. Training
with an independent random delay per play (as `arena.PLAY_DELAY` does) teaches a model to be robust to timing, but it
cannot learn anything that needs an exact tick, because it never knows which tick it gets.

How much an exact tick matters, from the engine: 1,370 Ice Golem + Skeletons clutches against a lone Hog, each replayed with
the cards 1-2 ticks off:

| Ice Golem lands | 2nd card 1 early | on time | 1 late | 2 late |
|---|---|---|---|---|
| 1 tick early | 27% | 33% | 19% | 13% |
| **on time** | 37% | **100%** | 36% | 25% |
| 1 tick late | 24% | 30% | 24% | 18% |
| 2 ticks late | 15% | 24% | 17% | 17% |

(share of the 1,370 that still clutch). For ordinary plays (a Hog at the bridge, a building) a tick or two hardly
matters; for clutches and tight defence it decides the outcome.

What we do (clapha-train `il.duel --timing certain`): every play lands on the first tick ending in 2 at least 22 ticks
after the decision, and the model is told that exact delay as an input; live, the bot holds the touch for that turn's
R - 8 slot. Checked in the engine: 147 of 147 plays on their planned ticks. Live clutches with this timing (all attempts since
10-21 02:24): 182 attempts, 167 clutches; 156 of the 157 attempts whose cards all landed on their planned ticks clutched,
and 14 of the 15 misses were a card slipping a tick (38 of 364 cards slipped, mostly when a turn arrived late).

## Re-checking

Read R and the moment it changes (every few ms), touch a cheap card at known times after a turn arrives, and read each
play's issue tick from the command queue. The R - 8 window moves with the round trip (it moved from ~330-470 ms to
~400-490 ms after a turn in one evening), so measure it live rather than fixing it.
