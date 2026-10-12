#!/bin/bash
# The PPO learner of the E-Barbs specialist run (ppo-ebarbs2, README): learner on GPU 0, decision servers (MPS) on
# GPU 1, collectors on the job's CPUs; --remote lets tools/satellite.sbatch join it. Resumes from <run>/latest.pt.
#
#   cd ~/cr-engine && sbatch tools/train.sbatch && sbatch tools/satellite.sbatch    # start
#   scancel --name=cr-ppo                                                          # stop both
#
# qos_p107-a100 allows 16 CPUs per user however many GPUs: collectors (CPU) are the limit, so more A100s add nothing.
#SBATCH --job-name=cr-ppo
#SBATCH --account=competition
#SBATCH --partition=P107-A100
#SBATCH --qos=qos_p107-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/train-%j.out

ulimit -u 65536
cd "$HOME/cr-engine"
R=$HOME/cr-engine/runs
exec nice -n 10 env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python -m il.rl "$HOME/clapha-engine" \
  $R/ppo-general/latest.pt $R/ppo-ebarbs2 --remote \
  --il $R/il-002/best.pt --il-share 0.4 --deck-mix 0.3,0.4,0.3 --shaping 0.1 \
  --learner-deck 26000043:1,26000020,26000036:1,26000052,26000017:2,26000083,28000015,28000026 \
  --self-play 0 --opponent-pool $R/ppo-general/pool \
  --collectors 40 --servers 4 --server-device cuda:1 --pool $HOME/cr-engine/data/deck_pool.json \
  --lr 2e-4 --games-per-update 256 --lam 0.98 --kl-coef 0.01
