#!/bin/bash
# Which IL_Replay games replay exactly, and on which game data (clapha-engine docs/training/VERSION_ORACLE.md): the
# 15.535.86 data first, then 15.535.29 on the games still not exact, then the report (~/oracle/assignment.jsonl is
# what clapha-il's convert.py reads). Resumable: submit again and it continues. Engines unpacked in ~/engine-86, -29.
#
#   cd ~/cr-engine && sbatch tools/oracle.sbatch
#
# qos_p107-rtx5090 allows 16 CPUs per user: this job and tools/satellite.sbatch cannot run at the same time.
#SBATCH --job-name=cr-oracle
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/oracle-%j.out

set -e
cd "$HOME/clapha-engine"
COMMON="--replays $HOME/cr-engine/data/IL_Replay/replays --out $HOME/oracle --workers 15 --seed-check 2
        --firstlight $HOME/firstlight-cr --converter $HOME/clapha-engine/training/cr"
python3 crx/version_oracle.py run --engine d86 ~/engine-86/engine ~/engine-86/data/assets $COMMON
python3 crx/version_oracle.py run --engine d29 ~/engine-29/engine ~/engine-29/data/assets --fallback $COMMON
python3 crx/version_oracle.py report --out ~/oracle --prefer d86 d29
