#!/bin/bash
# il.autopilot for the E-Barbs specialist run: a CPU-only job next to the training job (engines need CPU, not a GPU).
#
#   cd ~/cr-engine && sbatch tools/autopilot.sbatch
#   RUN=~/cr-engine/runs/other START=... sbatch tools/autopilot.sbatch
#
# Same partition / QoS as tools/collectors.sbatch (16 CPUs per user there).
#SBATCH --job-name=cr-autopilot
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/autopilot-%j.out

ulimit -u 65536
cd "$HOME/cr-engine"
R=$HOME/cr-engine/runs
exec nice -n 10 env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python -m il.autopilot "$HOME/clapha-engine" \
  "${RUN:-$R/ppo-ebarbs2}" --start "${START:-$R/ebarbs-start.pt}" \
  --panel general=$R/ppo-general/latest.pt --panel il=$R/il-002/best.pt \
  --deck 26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026
