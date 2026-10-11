#!/bin/bash
# A CPU-only job of PPO collectors that joins a running `il.rl --remote` learner (it finds it through
# <run>/endpoint.json on the shared home; it waits and reconnects whenever the learner restarts).
#
#   cd ~/cr-engine && sbatch tools/collectors.sbatch                          # 30 collectors for runs/ppo-general
#   RUN=~/cr-engine/runs/other COLLECTORS=16 sbatch tools/collectors.sbatch
#
# No GPU is requested: the collectors only need CPU. qos_p107-rtx5090 allows 16 CPUs per user (separate from the
# A100 job's QoS). qos_stu_cpu_long (32 CPUs) would be better, but its association names the partition 'students'
# while the partition is 'Students', so Slurm rejects it ("Invalid qos specification") until the admin fixes it.
#SBATCH --job-name=cr-collect
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/collect-%j.out

ulimit -u 65536
cd "$HOME/cr-engine"
exec nice -n 10 env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python -m il.rl join \
  "${RUN:-$HOME/cr-engine/runs/ppo-general}" --collectors "${COLLECTORS:-30}"
