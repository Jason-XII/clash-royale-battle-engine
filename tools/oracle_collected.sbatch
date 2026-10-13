#!/bin/bash
# Which of Arron's collected battles (68k, dated) replay, on the engine of their date (clapha-engine
# docs/training/VERSION_ORACLE.md, "Arron's collected battles"): winner + crowns + elixir spent, one side only, tower
# troops tried where a battle fails (--tower-trials: 15 more runs per failing row, the slow part). Then the report
# (~/oracle-collected/assignment.jsonl is what clapha-il's convert.py reads). Resumable: submit again and it continues.
#
#   cd ~/cr-engine && sbatch tools/oracle_collected.sbatch
#   LIMIT=60 OUT=~/oracle-collected-smoke sbatch tools/oracle_collected.sbatch   # a trial: check it has no "error" rows
#   TRIALS= sbatch tools/oracle_collected.sbatch     # without the tower-troop trials (fast; they can be added later)
#
# Engines unpacked as for tools/oracle.sbatch, one folder per bundle of the Drive folder:
#   ~/engine-22  clapha-engine-files-150535022.tar.gz    (July battles)
#   ~/engine-86  clapha-engine-files-150535086.tar.gz    (August; also IL_Replay's)
#   ~/engine-29  clapha-engine-files-150535029.tar.gz    (August fallback)
#   ~/engine-12  clapha-engine-files-160402012.tar.gz    (September)
#   ~/engine-19  clapha-engine-files-160402020.tar.gz    (September fallback)
# Battles: arron-collected-battles-2026-07-to-09-v2.tar unpacked in ~/collected (-> ~/collected/replays). The v2 shards
# carry the elixir stats the check needs; the first tar did not.
#SBATCH --job-name=cr-oracle-col
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/oracle-collected-%j.out

set -e
ulimit -u 65536
cd "$HOME/clapha-engine"
PY=/public/app/miniconda3/py312_24.4.0-0/bin/python
O=${OUT:-$HOME/oracle-collected}
COMMON="--replays $HOME/collected/replays --out $O --workers 23 --limit ${LIMIT:-0} ${TRIALS---tower-trials}
        --firstlight $HOME/firstlight-cr --converter $HOME/clapha-engine/training/cr"
E() { echo "$HOME/engine-$1/engine $HOME/engine-$1/data/assets"; }
$PY crx/version_oracle.py run --engine c22 $(E 22) --between 2026-07-01 2026-08-04 $COMMON
$PY crx/version_oracle.py run --engine c86 $(E 86) --between 2026-08-05 2026-09-01 $COMMON
$PY crx/version_oracle.py run --engine c29 $(E 29) --between 2026-08-05 2026-09-01 --fallback --accept crowns $COMMON
$PY crx/version_oracle.py run --engine c12 $(E 12) --between 2026-09-02 2026-09-30 $COMMON
$PY crx/version_oracle.py run --engine c19 $(E 19) --between 2026-09-02 2026-09-30 --fallback --accept crowns $COMMON
$PY crx/version_oracle.py report --out $O --accept crowns --prefer c86 c29 c22 c12 c19
