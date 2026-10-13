#!/bin/bash
# More games for a running `il.rl --remote` learner (tools/train.sbatch): this job's collectors and decision servers
# (MPS on its GPU) play the learner's games, the weights come from the learner (il.rl join --servers). It also runs
# il.autopilot, which evaluates for ~5 min per snapshot; at nice 0 it takes the CPUs it needs from the collectors.
#
#   cd ~/cr-engine && RUN=~/cr-engine/runs/ppo-general sbatch tools/satellite.sbatch   # waits for the learner, reconnects
#   RUN=... AUTOPILOT=0 sbatch tools/satellite.sbatch
#   autopilot: START = the run's starting checkpoint (default the IL model the learner started from), PANEL = more
#   "name=checkpoint" panel models, DECK = a specialist's deck
#   scancel --name=cr-ppo                                    # stops it and the learner
#
# qos_p107-rtx5090 allows 16 CPUs per user (separate from the A100 job's QoS); a second GPU would add no CPUs.
#SBATCH --job-name=cr-ppo
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/satellite-%j.out

ulimit -u 65536
cd "$HOME/cr-engine"
RUN=${RUN:?RUN: the run directory of the learner}
PY="env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python"
if [ "${AUTOPILOT:-1}" = 1 ]; then
  until [ -f "$RUN/endpoint.json" ]; do sleep 30; done
  START=${START:-$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['init'])" "$RUN/endpoint.json")}
  $PY -m il.autopilot "$HOME/clapha-engine" "$RUN" --start "$START" --panel "il=$START" \
    $(for p in $PANEL; do echo --panel "$p"; done) ${DECK:+--deck "$DECK"} \
    > runs/autopilot-$SLURM_JOB_ID.out 2>&1 &
fi
exec nice -n 5 $PY -m il.rl join "$RUN" --collectors "${COLLECTORS:-36}" --servers "${SERVERS:-3}"
