#!/bin/bash
# What must work before a PPO run starts from a clapha-il checkpoint, on one GPU node (~30 min):
#   1. a game played here gives the model the inputs imitation learning gave it (tools/check_clapha_inputs.py)
#   2. where a decision's CPU time goes (tools/time_decision.py)
#   3. 16 arena games of the checkpoint against itself (il.arena: il.policy's policy, abilities included)
#   4. 5 PPO updates of 32 games (il.rl: collectors, decision servers, the update); games per hour in its metrics
#
#   cd ~/cr-engine && CKPT=~/il-runs/pilot-base/latest.pt sbatch tools/check.sbatch
#   REF=<branch> ...: checks that branch of ~/cr-engine (a clone in the job) before it is deployed; the native_engine
#   package (engine wrapper) stays the installed ~/cr-engine one
#   (on the RTX 5090s: sbatch -A competition -p P107-RTX5090 -q qos_p107-rtx5090 tools/check.sbatch)
#SBATCH --job-name=cr-check
#SBATCH --account=competition
#SBATCH --partition=P107-A100
#SBATCH --qos=qos_p107-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --exclude=anode02
#SBATCH --output=runs/check-%j.out

echo "node $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
ulimit -u 65536
cd "$HOME/cr-engine"
if [ -n "$REF" ]; then
  git clone -q -b "$REF" "$HOME/cr-engine" "/tmp/cr-check-$SLURM_JOB_ID" && cd "/tmp/cr-check-$SLURM_JOB_ID" || exit 1
  echo "checking $REF at $(git log --oneline -1)"
fi
PY="env PYTHONPATH=. /public/app/miniconda3/py312_24.4.0-0/bin/python"
CKPT=${CKPT:?CKPT: a clapha-il checkpoint}
OUT=$HOME/cr-engine/runs/check-$SLURM_JOB_ID
mkdir -p "$OUT"
echo "== 1. inputs in self-play = inputs in imitation learning"
$PY tools/check_clapha_inputs.py "$HOME/clapha-engine" "$CKPT" 2>&1 | grep -v '^crx:' | tail -6
echo "== 2. CPU time of a decision"
$PY tools/time_decision.py "$HOME/clapha-engine" "$CKPT" 2>&1 | tail -1
echo "== 3. arena"
$PY -m il.arena "$HOME/clapha-engine" "$CKPT" "$CKPT" --games 16 --procs 16 --pool "$HOME/cr-engine/data/deck_pool.json" \
  --out "$OUT/arena.jsonl" 2>&1 | tail -4
echo "== 4. PPO"
timeout 45m $PY -m il.rl "$HOME/clapha-engine" "$CKPT" "$OUT/ppo" --collectors 24 --servers 2 --games-per-update 32 \
  --updates 5 --pool "$HOME/cr-engine/data/deck_pool.json" 2>&1 | grep -v '^crx:' | tail -12
echo "exit $?"
[ -n "$REF" ] && rm -rf "/tmp/cr-check-$SLURM_JOB_ID"
