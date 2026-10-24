"""Payload JSON-lines files -> convert.py's assignment (tag, part, row, engine[, alt]): each battle on the game data that was
live at its time (docs/CONVERSION.md, "Game data by battle time"). A battle on the day an update reached players also gets
'alt', the data on the other side of the update: convert.py runs it there when the first did not reproduce it.

    python tools/assign_by_date.py OUT.jsonl PAYLOADS_DIR [...]

Each line of a payload file: {"tag", "battle_time" (UTC ISO), "payload"} (clapha-engine crx/replay_audit/schema5_json.py).
The names below are convert.py --engine names; give one --engine per data version you convert:
    --engine sep17 ENGINES/160402012 DATA/sep17   --engine oct1 ENGINES/160402012 DATA/oct1
    --engine d19 ENGINES/160402020 DATA/16.402.19 --engine d22 ENGINES/160402020 DATA/16.402.22   (and older ones)
"""
import json
import sys
from pathlib import Path

# (first moment it is live, UTC; data name; engine code): measured 2026-10-23/24 on battles. Before a later entry, the
# earlier one. 'sep17', 'oct1', 'jul6', 'aug26' are rebuilds (the asset server serves only the current data).
WINDOWS = [
    ('2026-07-01', 'd22jul', '150535029'),      # 15.535.22
    ('2026-07-06', 'jul6', '150535029'),        # 6 Jul balance update (rebuilt)
    ('2026-08-04', 'd86', '150535029'),         # 15.535.29 / .86
    ('2026-08-26', 'aug26', '150535029'),       # 26 Aug update (rebuilt)
    ('2026-09-01', 'd2a', '160402002'),         # 16.402.2 from app 160402002
    ('2026-09-07T05:30', 'd2b', '160402012'),   # 16.402.2 from app 160402012 (8 Sep post); 09-07 05:02 still d2a
    ('2026-09-17', 'sep17', '160402012'),       # 16 Sep post (4 changes)
    ('2026-09-28', 'oct1', '160402012'),        # Hero Ice Wizard rework + V16 flags (no post), + Electro Wizard hero (5 Oct)
    ('2026-10-06T06:30', 'd19', '160402020'),   # 16.402.19 (6 Oct post)
    ('2026-10-19T09:40', 'd22', '160402020'),   # 16.402.22 (19 Oct post)
]
# an update's first day (and 28-30 Sep, where the Hero Ice Wizard rework's day is not known): both versions
SWITCH = {('2026-09-07', '2026-09-08'): ('d2b', 'd2a'), ('2026-09-28', '2026-10-01'): ('oct1', 'sep17'),
          ('2026-10-06', '2026-10-07'): ('d19', 'oct1'), ('2026-10-19', '2026-10-20'): ('d22', 'd19')}


def data_at(when):
    """(data name, alt or None) for a battle time 'YYYY-MM-DDTHH:MM:SSZ'; None before the first window."""
    name = None
    for start, data, _ in WINDOWS:
        if when >= start:
            name = data
    if name is None:
        return None
    for (lo, hi), (new, old) in SWITCH.items():
        if lo <= when < hi:
            return (new, old) if name == new else (name, new if name == old else None)
    return name, None


def main(argv):
    out, dirs = Path(argv[0]), argv[1:]
    rows, skipped = [], 0
    for directory in dirs:
        for path in sorted(Path(directory).expanduser().glob('*.jsonl')):
            for i, line in enumerate(open(path)):
                if not line.strip():
                    continue
                j = json.loads(line)
                chosen = data_at(j['battle_time'])
                if chosen is None:
                    skipped += 1
                    continue
                row = {'tag': j['tag'], 'part': path.name, 'row': i, 'engine': chosen[0]}
                if chosen[1]:
                    row['alt'] = chosen[1]
                rows.append(row)
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'{len(rows)} assigned ({sum(1 for r in rows if "alt" in r)} with an alternative), {skipped} before July 2026 left out')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
