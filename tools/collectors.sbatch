#!/bin/bash
# A CPU-only job of PPO collectors that joins a running `il.rl --remote` learner (it finds it through
# <run>/endpoint.json on the shared home; it waits and reconnects whenever the learner restarts).
#
#   cd ~/cr-engine && sbatch tools/collectors.sbatch                          # 30 collectors for runs/ppo-general
#   RUN=~/cr-engine/runs/other COLLECTORS=16 sbatch tools/collectors.sbatch
#
#SBATCH --job-name=cr-collect
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_cpu_long
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/collect-%j.out

ulimit -u 65536
cd "$HOME/cr-engine"
exec nice -n 10 env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python -m il.collect \
  "${RUN:-$HOME/cr-engine/runs/ppo-general}" --collectors "${COLLECTORS:-30}"
