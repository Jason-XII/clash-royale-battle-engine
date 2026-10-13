#!/bin/bash
# One imitation-learning run of clapha-il's train.py (docs/TRAINING.md) on converted games, on one GPU with the job's
# CPUs building features (they are the limit: ~900 decisions/s a core). Runs of one comparison share a vocabulary
# (VOCAB, built once from 3,000 of the games when missing).
#
#   cd ~/cr-engine && RUN=pilot-base sbatch tools/il.sbatch
#   RUN=pilot-no-attention FLAGS="--attention-layers 0" sbatch -A competition -p P107-RTX5090 -q qos_p107-rtx5090 tools/il.sbatch
#
# The pilot (docs/TRAINING.md): 20k games, 4 epochs, from a frozen copy of the first converted games (~/il-raw-pilot),
# so the full conversion can go on appending to ~/il-raw meanwhile.
#SBATCH --job-name=cr-il
#SBATCH --account=competition
#SBATCH --partition=P107-A100
#SBATCH --qos=qos_p107-a100
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/il-%j.out

set -e
PY=/public/app/miniconda3/py312_24.4.0-0/bin/python
RUN=${RUN:?RUN=name of the run}
GAMES=${GAMES:-$HOME/il-raw-pilot/games}
VOCAB=${VOCAB:-$HOME/il-runs/vocab-pilot.json}
mkdir -p ~/il-runs
cd "$HOME/clapha-il"
if [ ! -f "$VOCAB" ]; then
  $PY - "$GAMES" "$VOCAB" <<'EOF'
import glob, itertools, json, os, sys
from clapha_il import features as FE
from convert import read_games
import train
games, out = sys.argv[1], sys.argv[2]
sample = itertools.islice((g for p in sorted(glob.glob(games + '/*/*.bin')) for g in read_games(p)), 3000)
effects = {int(i) for i, _ in json.loads(open('clapha_il/effect_ids.json').read())['effects']}
FE.Vocab.build(sample, extra=effects | train._all_ids()).save(out + '.tmp')
os.replace(out + '.tmp', out)
EOF
fi
exec $PY train.py --games "$GAMES" --out "$HOME/il-runs/$RUN" --vocab "$VOCAB" --limit-games "${LIMIT:-20000}" \
    --epochs "${EPOCHS:-4}" --workers $((SLURM_CPUS_PER_TASK - 1)) --device cuda --firstlight ~/firstlight-cr $FLAGS
