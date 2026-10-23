"""Card and unit stats from the game's own logic tables (22 Oct): what each card and each unit on the board does,
for the network's card inputs (it only knew ids and costs).

    python card_stats.py CSV_LOGIC VOCAB.json REPLAY_CARDS.json UNIT_STATE_TABLES.json OUT.json

CSV_LOGIC: the build's data/assets/csv_logic (compressed CSV and TOML, decoded here); UNIT_STATE_TABLES: the export that
names the units' data ids (clapha-engine crx/export_tables.py; e.g. ~/cr-live/models/unit_state_tables_150535029.json).

OUT: {'fields': [...], 'stats': {vocab id: [values]}, 'missing': [...]} for card ids (26/27/28: the summoned unit, or
the spell's projectile / area effect) and unit data ids (what objects report, named by the unit-state tables).
Values are at the tables' base level and roughly scaled to [0, 1]."""
import csv, io, json, lzma, math, sys, tomllib
from pathlib import Path

ROOT, VOCAB, CARDS, UNITS, OUT = map(Path, sys.argv[1:6])


def raw(p):
    d = p.read_bytes()
    return lzma.LZMADecompressor(lzma.FORMAT_ALONE).decompress(d[:9] + b'\0' * 4 + d[9:]) if d[:1] == b']' else d


def csv_table(p):
    rows = list(csv.reader(io.StringIO(raw(p).decode('utf-8', 'replace'))))
    out, last = {}, None
    for r in rows[2:]:
        if r and r[0]:
            last = r[0]
            out[last] = {h: v for h, v in zip(rows[0], r) if v != ''}
        elif last is not None:  # a continuation row (per-level values): keep the first level's
            for h, v in zip(rows[0], r):
                out[last].setdefault(h, v) if v != '' else None
    return out


TABLES = {'CHARACTER': ['characters.csv'], 'BUILDING': ['buildings.csv'], 'PROJECTILE': ['projectiles.csv'],
          'AEO': ['area_effect_objects.csv'],
          'SPELL': ['spells_characters.csv', 'spells_buildings.csv', 'spells_other.csv', 'spells_evolved.csv',
                    'spells_hero_form.csv']}
TOML_KINDS = {'CHARACTER': 'CHARACTER', 'BUILDING': 'BUILDING', 'PROJECTILE': 'PROJECTILE', 'AEO': 'AEO',
              'SPELL_CHARACTER': 'SPELL', 'SPELL_BUILDING': 'SPELL', 'SPELL_OTHER': 'SPELL', 'SPELL_HERO': 'SPELL',
              'SPELL_EVOLVED': 'SPELL'}
T = {k: {} for k in TABLES}
for kind, files in TABLES.items():
    for f in files:
        if (ROOT / f).exists():
            for name, row in csv_table(ROOT / f).items():
                T[kind].setdefault(name, {}).update(row)
for p in sorted(ROOT.rglob('*.toml')):
    try:
        t = tomllib.loads(raw(p).decode('utf-8', 'replace'))
    except Exception:  # noqa: BLE001
        continue
    for kind, entries in t.items():
        if kind in TOML_KINDS and isinstance(entries, dict):
            for name, e in entries.items():
                if isinstance(e, dict):
                    T[TOML_KINDS[kind]].setdefault(name, {}).update({k: v for k, v in e.items() if not isinstance(v, (dict, list))})


def num(e, k, default=0.0):
    v = e.get(k)
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(str(v).lower() == 'true') if isinstance(v, str) and v.lower() in ('true', 'false') else default


def log(v, scale):
    return math.log1p(max(v, 0.0)) / scale


FIELDS = ('troop', 'building', 'spell', 'count', 'hitpoints', 'damage', 'dps', 'hit_speed_s', 'range', 'min_range',
          'sight_range', 'speed', 'deploy_s', 'attacks_air', 'attacks_ground', 'only_buildings', 'flying', 'splash',
          'death_damage', 'lifetime_s', 'spawner', 'spell_radius', 'spell_damage', 'collision', 'mass', 'charge',
          'shield', 'kamikaze')


def unit_values(e, building):
    hs = num(e, 'HitSpeed') / 1000
    dmg = num(e, 'Damage')
    proj = T['PROJECTILE'].get(e.get('Projectile'), {}) if e.get('Projectile') else {}
    dmg = dmg or num(proj, 'Damage')
    splash = num(e, 'AreaDamageRadius') or num(proj, 'Radius')
    return dict(troop=float(not building), building=float(building), hitpoints=log(num(e, 'Hitpoints'), 9),
                damage=log(dmg, 8), dps=log(dmg / hs if hs else 0, 8), hit_speed_s=hs / 3,
                range=num(e, 'Range') / 12000, min_range=num(e, 'MinimumRange') / 12000,
                sight_range=num(e, 'SightRange') / 12000, speed=num(e, 'Speed') / 120,
                deploy_s=num(e, 'DeployTime') / 2000, attacks_air=num(e, 'AttacksAir'),
                attacks_ground=num(e, 'AttacksGround'), only_buildings=num(e, 'TargetOnlyBuildings'),
                flying=float(num(e, 'FlyingHeight') > 0), splash=splash / 5000,
                death_damage=log(num(e, 'DeathDamage'), 8), lifetime_s=num(e, 'LifeTime') / 60000,
                spawner=float(bool(e.get('SpawnCharacter'))), collision=num(e, 'CollisionRadius') / 1000,
                mass=num(e, 'Mass') / 20, charge=float(num(e, 'ChargeRange') > 0),
                shield=log(num(e, 'ShieldHitpoints'), 9), kamikaze=num(e, 'Kamikaze'))


LOWER = {kind: {**{n.lower(): n for n in T[kind]}, **{n.lower().replace('_', ''): n for n in T[kind]}}
         for kind in ('CHARACTER', 'BUILDING')}


def entity(name):
    """A unit's values; an evolution or hero form not in the tables takes its base unit's (ponytail: evolutions differ
    in a few stats; read their own tables if the input proves useful)."""
    for n in (name, name.replace('_EV1', '').replace('_ev1', ''), name.replace('Hero', ''),
              name.split('_rework_')[-1]):
        for kind, building in (('CHARACTER', False), ('BUILDING', True)):
            real = LOWER[kind].get(n.lower().replace('_', '')) or LOWER[kind].get(n.lower())
            if real:
                return unit_values(T[kind][real], building)
    return None


def card_values(name):
    s = T['SPELL'].get(name)
    if s is None:
        return None
    count = num(s, 'SummonNumber', 1.0) if s.get('SummonCharacter') else 0.0
    v = entity(s['SummonCharacter']) if s.get('SummonCharacter') else None
    if v is None:  # a spell: its projectile or area effect
        p = T['PROJECTILE'].get(s.get('Projectile'), {})
        a = T['AEO'].get(s.get('AreaEffectObject'), {})
        v = dict(spell=1.0, spell_radius=(num(s, 'Radius') or num(p, 'Radius') or num(a, 'Radius')) / 5000,
                 spell_damage=log(num(p, 'Damage') or num(a, 'Damage') or num(s, 'Damage'), 8),
                 lifetime_s=num(a, 'LifeDuration') / 60000)
    v['count'] = count / 10
    return v


vocab = json.loads(VOCAB.read_text())
cards = json.loads(CARDS.read_text())['cards']
units = json.loads(UNITS.read_text())['characters']
hero_of = {u: int(c) for c, u in json.loads(CARDS.read_text())['hero_units'].items()}  # hero unit id -> its card
stats, missing = {}, []
for i in vocab:
    cls = i // 1000000
    if i in hero_of and str(hero_of[i]) in cards:  # ponytail: a hero unit takes its card's stats
        name = cards[str(hero_of[i])]['name']
        v = card_values(name)
    elif cls in (26, 27, 28) and str(i) in cards:
        name = cards[str(i)]['name']
        v = card_values(name)
    elif str(i) in units:
        name = units[str(i)]['name']
        v = entity(name)
    else:
        continue
    if v is None:
        missing.append((i, name))
        continue
    stats[i] = [round(float(v.get(f, 0.0)), 4) for f in FIELDS]
OUT.write_text(json.dumps(dict(fields=FIELDS, stats=stats, missing=missing)))
print(len(stats), 'with stats;', len(missing), 'missing:', missing[:40])
for i in (26000000, 27000000, 28000000, 34000000):
    if i in stats:
        print(i, dict(zip(FIELDS, stats[i])))
