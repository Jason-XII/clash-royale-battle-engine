#!/bin/bash
# The PPO learner (il.rl) from a clapha-il checkpoint: learner on GPU 0, decision servers (MPS) on GPU 1 (GPU 0 too with one
# GPU), collectors on the job's CPUs; --remote lets tools/satellite.sbatch join it. Resumes from $RUN/latest.pt.
#
#   cd ~/cr-engine && INIT=~/il-runs/<run>/latest.pt RUN=~/cr-engine/runs/ppo-general sbatch tools/train.sbatch
#   RUN=~/cr-engine/runs/ppo-general sbatch tools/satellite.sbatch       # more collectors for it
#   scancel --name=cr-ppo                                                # stop both
#   # smoke: one GPU, 20 updates of 32 games
#   INIT=... RUN=~/cr-engine/runs/ppo-smoke UPDATES=20 FLAGS="--games-per-update 32" sbatch --gres=gpu:1 tools/train.sbatch
#
# FLAGS adds il.rl options (a specialist: --learner-deck ... --self-play 0 --il-share 0.4 --deck-mix 0.3,0.4,0.3).
# qos_p107-a100 allows 16 CPUs per user however many GPUs: collectors (CPU) are the limit.
#SBATCH --job-name=cr-ppo
#SBATCH --account=competition
#SBATCH --partition=P107-A100
#SBATCH --qos=qos_p107-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=anode02
#SBATCH --output=runs/train-%j.out

echo "node $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
ulimit -u 65536
cd "$HOME/cr-engine"
SERVER_DEVICE=cuda:$(( $(nvidia-smi -L | wc -l) > 1 ? 1 : 0 ))
exec nice -n 10 env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python -m il.rl "$HOME/clapha-engine" \
  "${INIT:?INIT: the clapha-il checkpoint to start from}" "${RUN:?RUN: the run directory}" --remote \
  --collectors "${COLLECTORS:-40}" --servers "${SERVERS:-4}" --server-device "$SERVER_DEVICE" \
  --pool "$HOME/cr-engine/data/deck_pool.json" --updates "${UPDATES:-100000}" $FLAGS
