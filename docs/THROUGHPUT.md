# PPO throughput, 17 October 2026

The target is accepted learning trajectories per hour, with the five-tick decision cadence,
recurrent policy, observations, opponent sampling, batch size and training passes preserved.
Raw simulated games alone do not establish training throughput or playing strength.

## Resources and baseline

The running specialist uses **four A100 learner GPUs**, **four RTX 5090 actor GPUs**,
and **two additional A100 actor GPUs**. The actors have 40 logical CPUs (20 physical cores).
The learner has 16 logical CPUs. Thus the results below include ten GPUs in total.
The specialist supplies one learning trajectory per game; the general model's self-play
configuration can supply two, so game counts across those runs are not directly comparable.

Versions 598–617 averaged 23,354 reported games/hour (23,230 from elapsed game counts),
about 813,000 accepted decisions per 1,028 trajectories. Updates averaged 70.25 seconds,
including 29.13 seconds for the critic, plus 88.31 seconds waiting for data.
The initial actor configuration was 50 collectors/four servers on RTX and 64/two on A100.

## Isolated actor measurements

Each node measurement used its full original CPU/GPU allocation, 50 seconds warmup,
then 75 measured seconds. Original actor workers were paused or replaced temporarily,
and restored afterward. Candidate games used frozen weights and were never submitted to
training. These short measurements establish capacity, not a sustained training rate.

| Configuration | RTX games/hour | A100 actor games/hour | Sum |
|---|---:|---:|---:|
| Original, 50 + 64 collectors | 12,576 | 10,800 | 23,376 |
| Packed/deferred inference and shared board features | 15,312 | 15,984 | 31,296 |
| Earlier greedy opponent routing | 16,752 | 20,160 | 36,912 |
| Probability-interval routing plus native helpers | 19,776 | 21,360 | 41,136 |
| Reused staging, 75 + 96 collectors | 19,632 | 30,672 | 50,304 |
| Reused staging, A100 increased to 128 collectors | — | 34,368 | — |

The selected live actor counts are **50 RTX collectors/four servers** and
**128 A100 collectors/two servers**. Combining their separate best measurements suggests
about 54k games/hour; it is not a simultaneous sustained measurement.
Doubling servers per GPU with MPS reduced RTX throughput (16,896/hour) and gave only
22,080/hour on A100 with 64 collectors; it was not selected.

## Implementation

- CUDA serving packs inputs into one pinned transfer per graph/model group, queues model
  groups, and aggregates output readback. Warm staging fills the pinned arrays directly.
  Independently owned graph memory prevents cold captures from overwriting pending output.
  Readbacks fence reused buffers, including requests larger than the maximum row bucket.
- Collectors first sample their opponent using the unchanged PFSP distribution, then route
  within that opponent's cumulative probability interval. This balances expected traffic
  while reducing frozen-model/server pairs from M×S to at most M+S−1, without a barrier.
- Both player views share owner-independent board extraction and unit features per decision.
  Optional CPython helpers replace ctypes scalar reads and the existing 31 unit features.
- PPO can use IL's channels-last convolution layout. A real A100 trajectory minibatch,
  including two policy and two critic epochs, improved from about 1.21 to 1.08 seconds.
  This excludes DDP and does not establish a full-update speedup.
- New learner metrics separate target, policy and critic time and count accepted learning
  decisions per second. Actor logs count model calls as well as batches and rows.

## Validation

CPU suite: 18 passed, two CUDA tests skipped with the native extension; Python fallback:
17 passed, three skipped. Both CUDA regression variants (search top tiles enabled/disabled)
passed on the cluster, covering recurrent state, cold captures, object buckets, weight
reload, reset, model removal and reused buffers across multiple chunks.
Native/Python comparison covered 1,000 real decisions and 12,973 unit rows, with exact
frame/unit dictionaries and both players' input arrays. A further 3,000 randomized unit
feature cases matched. Repeated same-state observation CPU time fell from 0.578 to 0.435 ms.

An object-encoder compilation experiment gave about 5% inference improvement but changed
log probabilities (maximum difference about 0.143); it was rejected.
No fixed-panel strength evaluation has yet established an improvement in playing quality.

## Live rollout and rollback

Candidate actor code is frozen in `~/cr-ppo/src-throughput-1017`; the original source is
unchanged in `~/cr-ppo/src-spec-1017`. Jobs 100562 and 100563 retain their allocations.
The reversible supervisor `~/cr-ppo/diag/throughput-live-actors-1017.py` stops the old actor
parent, removes its workers and launches the candidate in a separate process group/socket
namespace. On supervisor exit it terminates the candidate and resumes the old parent,
which automatically recreates its original workers. State is recorded in
`RUN/throughput-actors-JOB.json`; parent output is `RUN/throughput-live-JOB.out`.
To roll back one actor group, send SIGTERM to the supervisor PID in its state file on
that allocation's node. Do not resume the old parent while the candidate still runs.
The learner has not been replaced during this actor rollout.

Steady-state learner updates 637–640 accepted **4,234 trajectories in approximately
300.49 seconds: 50,725/hour and 11,007 learning decisions/second**, about 2.18× the
elapsed-count baseline. The window duration is reconstructed from the rounded per-update
rates. [Raw rows and calculations](throughput-20261017.json) preserve the evidence.
Every update had zero stale trajectories, mean version lag 1.00–1.19, and 184
actor/server connections. Checkpoint 640 saved successfully. This five-minute window
does not establish day-long stability. Update 636 included the rollout transition
and is excluded. The original learner's channels-last optimization remains unactivated;
the capacity gain here comes from the deployed actors.

Serving fallback: `act --no-fast-inference --no-opponent-affinity`.
Native fallback: set `CR_NATIVE_FEATURES=0` before starting processes.
Build native helpers with the deployment interpreter: `python tools/build_features.py`.
The extension is optional, platform/Python-specific, and not checked into Git.
PPO layout fallback: `learn --no-channels-last` (saved settings take precedence on resume).

## Remaining limit toward 100k

At the original update time, the learner's approximate ceiling is 52.5k specialist
trajectories/hour. 100k requires about 36.9 seconds per 1,024-game update, versus 70.25
previously, plus roughly twice the optimized actor capacity. Next measure full DDP stage
times, padding by trajectory length/object count, optimizer/metric synchronization and
native direct-to-buffer feature construction. Preserve recurrent sequence semantics and
validate numerical changes against a fixed arena panel before claiming equal strength.
