#!/bin/bash
# The full IL run (docs/IL_PILOT.md, "Starting the full run"): every exact IL_Replay game, the settings the pilot chose
# (FLAGS), until the validation score has not improved for 5 validations; best.pt is the model. Resumable: the job's
# 4-day limit ends it, the same command continues it from latest.pt.
#
#   cd ~/cr-engine && FLAGS="<the pilot's choice, e.g. empty for the base settings>" sbatch tools/il_full.sbatch
#   # after a time-limit end: the same line again (it resumes)
#
# Watch: cd ~/clapha-il && python3 inspect_il.py run ~/cr-engine/runs/il-full --out ~/cr-engine/runs/il-full-report.html
#SBATCH --job-name=cr-il-full
#SBATCH --account=competition
#SBATCH --partition=P107-A100
#SBATCH --qos=qos_p107-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/il-full-%j.out

set -e
ulimit -u 65536
cd "$HOME/clapha-il"
exec /public/app/miniconda3/py312_24.4.0-0/bin/python train.py --games $HOME/il-raw/il/games \
  --out $HOME/cr-engine/runs/${RUN:-il-full} --epochs 20 --patience 5 --val-every 5000 --val-games 1000 \
  --workers 14 --device cuda --firstlight $HOME/firstlight-cr --resume $FLAGS
