#!/bin/bash
# Which IL_Replay games replay exactly, and on which game data (clapha-engine docs/training/VERSION_ORACLE.md): the
# 15.535.86 data first, then 15.535.29 on the games still not exact, then the report (~/oracle/assignment.jsonl is
# what clapha-il's convert.py reads). Resumable: submit again and it continues. Engines unpacked in ~/engine-86, -29.
#
#   cd ~/cr-engine && sbatch tools/oracle.sbatch
#
# The student QoS (24 CPUs, 1 day; plain qos_stu_medium is rejected on every partition), so PPO keeps its CPUs.
#SBATCH --job-name=cr-oracle
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/oracle-%j.out

set -e
cd "$HOME/clapha-engine"
PY=/public/app/miniconda3/py312_24.4.0-0/bin/python  # has pyarrow (the node's python3 does not)
COMMON="--replays $HOME/cr-engine/data/IL_Replay/replays --out $HOME/oracle --workers 23 --seed-check 2
        --firstlight $HOME/firstlight-cr --converter $HOME/clapha-engine/training/cr"
$PY crx/version_oracle.py run --engine d86 ~/engine-86/engine ~/engine-86/data/assets $COMMON
$PY crx/version_oracle.py run --engine d29 ~/engine-29/engine ~/engine-29/data/assets --fallback $COMMON
$PY crx/version_oracle.py report --out ~/oracle --prefer d86 d29
