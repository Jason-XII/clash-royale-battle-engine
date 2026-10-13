#!/bin/bash
# Accepted replays -> raw engine records at every decision (clapha-il convert.py, docs/CONVERT.md), on the engine the
# version oracle (tools/oracle.sbatch) accepted each on. Resumable: games already converted are skipped, so a growing
# assignment (a partial oracle report) can be converted in rounds. Then: check_converted.py OUT.
#
#   cd ~/cr-engine && sbatch tools/convert.sbatch                                    # ~/oracle -> ~/il-raw
#   ASSIGNMENT=~/oracle-partial/assignment.jsonl LIMIT=20000 sbatch -A competition -p P107-RTX5090 \
#       -q qos_p107-rtx5090 -c 16 tools/convert.sbatch                              # the pilot's games, PPO's CPUs
#SBATCH --job-name=cr-convert
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/convert-%j.out

set -e
ulimit -u 65536  # each engine starts dozens of threads
cd "$HOME/clapha-il"
/public/app/miniconda3/py312_24.4.0-0/bin/python convert.py run --assignment "${ASSIGNMENT:-$HOME/oracle/assignment.jsonl}" \
    --replays "$HOME/cr-engine/data/IL_Replay/replays" --accept exact \
    --engine d86 ~/engine-86/engine ~/engine-86/data/assets --engine d29 ~/engine-29/engine ~/engine-29/data/assets \
    --out "${OUT:-$HOME/il-raw}" --workers $((SLURM_CPUS_PER_TASK - 1)) --limit "${LIMIT:-0}" --firstlight ~/firstlight-cr
