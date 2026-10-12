#!/bin/bash
# More games for a running `il.rl --remote` learner (tools/train.sbatch): this job's collectors and decision servers
# (MPS on its GPU) play the learner's games, the weights come from the learner (il.rl join --servers). It also runs
# il.autopilot, which evaluates for ~5 min per snapshot; at nice 0 it takes the CPUs it needs from the collectors.
#
#   cd ~/cr-engine && sbatch tools/satellite.sbatch          # waits for the learner, reconnects when it restarts
#   RUN=~/cr-engine/runs/other AUTOPILOT=0 sbatch tools/satellite.sbatch
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
R=$HOME/cr-engine/runs
RUN=${RUN:-$R/ppo-ebarbs2}
PY="env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python"
if [ "${AUTOPILOT:-1}" = 1 ]; then
  $PY -m il.autopilot "$HOME/clapha-engine" "$RUN" --start "${START:-$R/ebarbs-start.pt}" \
    --panel general=$R/ppo-general/latest.pt --panel il=$R/il-002/best.pt \
    --deck 26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026 \
    > runs/autopilot-$SLURM_JOB_ID.out 2>&1 &
fi
exec nice -n 5 $PY -m il.rl join "$RUN" --collectors "${COLLECTORS:-36}" --servers "${SERVERS:-3}"
