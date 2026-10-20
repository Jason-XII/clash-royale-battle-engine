"""The improvement loop around a PPO run: a frozen champion, a learning challenger, a league, gates, recovery, a report.

    python loop.py RUN        # on the cluster login node: it only runs squeue / sbatch / scancel and reads small files

RUN/loop.json configures it (see LOOP_DEFAULTS; "jobs" maps a name to its sbatch command, which must print the job id,
i.e. use --parsable). Every minute:
- Jobs: any job not queued or running is resubmitted (a learner resumes from RUN/latest.pt; actors reconnect). A learner
  whose metrics.jsonl has been silent for `hang_min` is cancelled, then resubmitted.
- Gates: each new snapshot pool/vN.pt (the newest, if several) plays evaluation games (RUN/eval.json; the collectors play
  them between frozen models, ppo.py --eval-frac) against the champion and each benchmark model. With enough games:
  promoted if it scores >= `promote` against the champion and its benchmark mean is at most `bench_drop` below the
  champion's. Promotion: league/vN.pt (a permanent opponent) and champion.json (the learner's new KL reference).
- Rollback: after `rollback_after` straight gates scoring below `worse` against the champion, or a collapsed policy
  (entropy below `min_entropy` for 3 updates), the learner returns to the champion's full checkpoint (rollback.json).
  The initial champion has none: the loop stops instead (the settings, not luck, are then at fault).
- Stop: after `stall_after` straight gates without a promotion, or `max_hours`: every job is cancelled.
- Report: RUN/report.md (rewritten every minute) and RUN/loop-events.log.
"""
import getpass
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOOP_DEFAULTS = dict(promote=0.55, worse=0.45, bench_drop=0.03, champion_games=400, bench_games=200,
                     eval_timeout_min=90, rollback_after=2, stall_after=10, hang_min=30, min_entropy=0.05,
                     max_hours=72.0, keep_ckpts=3)


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


ENDED = ('BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED',
         'TIMEOUT')


def ended(job):
    """Slurm confirms the job finished: an end state, or 'Invalid job id' once it is purged (this cluster keeps no
    accounting, so sacct is empty). Any other answer, e.g. Slurm unreachable: not ended."""
    result = subprocess.run(['scontrol', 'show', 'job', str(job)], capture_output=True, text=True)
    state = re.search(r'JobState=(\w+)', result.stdout)
    return bool(state and state.group(1) in ENDED) or 'Invalid job id specified' in result.stderr + result.stdout


class Loop:
    def __init__(self, run):
        self.run = run
        self.cfg = {**LOOP_DEFAULTS, **json.loads((run / 'loop.json').read_text())}
        path = run / 'loop-state.json'
        self.state = json.loads(path.read_text()) if path.exists() else dict(
            started=time.time(), champion=self.cfg['champion'], champion_ckpt=None, jobs={}, restarts={},
            gates=[], pending=None, bench={}, fails=0, worse=0, rollbacks=0, stopped=None)
        self.write_champion()

    # ---- small files ----
    def save(self):
        tmp = self.run / 'loop-state.tmp'
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.run / 'loop-state.json')

    def put(self, name, data):
        tmp = self.run / f'{name}.tmp'
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.run / f'{name}.json')

    def event(self, text):
        with (self.run / 'loop-events.log').open('a') as f:
            f.write(f'{now()} {text}\n')

    def write_champion(self):
        self.put('champion', self.state['champion'])

    def metrics(self, n=5):
        """The last n rows of metrics.jsonl (read from its end: the file grows all run)."""
        path = self.run / 'metrics.jsonl'
        if not path.exists():
            return []
        with path.open('rb') as f:
            f.seek(0, 2)
            start = max(0, f.tell() - 40000 * n)
            f.seek(start)
            lines = f.read().decode(errors='replace').splitlines()[1 if start else 0:]
        rows = []
        for line in lines[-n:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        return rows

    def scores(self):
        """{evaluation match id: [A's scores]} from evals.jsonl."""
        out = {}
        path = self.run / 'evals.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                out.setdefault(row['eval'], []).append(row['score'])
        return out

    # ---- jobs ----
    def jobs(self):
        """Resubmit any job that is not queued or running; cancel a hung learner (resubmitted next minute)."""
        listing = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i %T %M'], capture_output=True,
                                 text=True)
        if listing.returncode:  # Slurm itself is unwell: never resubmit on a guess
            return {}
        queued = {line.split()[0]: line.split()[1:] for line in listing.stdout.splitlines() if line.strip()}
        alive = {}
        for name, command in self.cfg['jobs'].items():
            job = self.state['jobs'].get(name)
            status = queued.get(str(job), []) if job else []
            if not status and job and not ended(job):  # missing from squeue but not ended (squeue can come back
                continue                               # empty while Slurm's user lookup fails): never a duplicate
            if not status:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                new = result.stdout.strip().split(';')[0]
                if result.returncode or not new.isdigit():
                    self.event(f'job {name}: sbatch failed: {result.stderr.strip()[:300]}')
                    continue
                if job:
                    self.state['restarts'][name] = self.state['restarts'].get(name, 0) + 1
                    self.event(f'job {name}: {job} gone; resubmitted as {new}')
                else:
                    self.event(f'job {name}: submitted {new}')
                self.state['jobs'][name] = new
                status = ['PENDING', '0:00']
            alive[name] = status
        learner = alive.get('learner')
        metrics = self.run / 'metrics.jsonl'
        if learner and learner[0] == 'RUNNING' and minutes(learner[1]) > self.cfg['hang_min'] + 10 and \
                (not metrics.exists() or time.time() - metrics.stat().st_mtime > 60 * self.cfg['hang_min']):
            subprocess.run(['scancel', str(self.state['jobs']['learner'])])
            self.event(f"learner {self.state['jobs']['learner']} silent for {self.cfg['hang_min']} min: cancelled")
        return alive

    def stop(self, reason):
        for job in self.state['jobs'].values():
            subprocess.run(['scancel', str(job)])
        self.put('eval', {'matches': []})
        self.state['stopped'] = f'{now()}: {reason}'
        self.event(f'STOPPED: {reason}')

    # ---- gates ----
    def matches(self, name, path):
        """The evaluation matches a gate of `name` needs: against the champion and each benchmark model, and the
        champion's own benchmark if it was never measured (the initial champion)."""
        c, cfg = self.state['champion'], self.cfg
        out = [dict(id=f'{name}:{c["name"]}', a=[name, path], b=[c['name'], c['path']], games=cfg['champion_games'])]
        for b, bpath in cfg['benchmark'].items():
            out.append(dict(id=f'{name}:{b}', a=[name, path], b=[b, bpath], games=cfg['bench_games']))
            if c['name'] not in self.state['bench']:
                out.append(dict(id=f'{c["name"]}:{b}', a=[c['name'], c['path']], b=[b, bpath], games=cfg['bench_games']))
        return out

    def gates(self, scores):
        pending, cfg = self.state['pending'], self.cfg
        if pending is None:
            done = max((g['name'] for g in self.state['gates']), default='')
            snaps = sorted(p for p in (self.run / 'pool').glob('v*.pt') if p.stem > done)  # v000020 < v000040
            if not snaps:
                return
            newest = snaps[-1]
            pending = self.state['pending'] = dict(name=newest.stem, path=str(newest), started=time.time(),
                                                   matches=self.matches(newest.stem, str(newest)))
            self.event(f"gate {newest.stem}: evaluation started ({sum(m['games'] for m in pending['matches'])} games)")
        left = [m for m in pending['matches'] if len(scores.get(m['id'], [])) < m['games']]
        late = time.time() - pending['started'] > 60 * cfg['eval_timeout_min']
        if left and not late:
            self.put('eval', {'gate': pending['name'], 'matches': [
                dict(m, games=m['games'] - len(scores.get(m['id'], []))) for m in left]})
            return
        self.put('eval', {'matches': []})
        self.decide(pending, scores, late and bool(left))
        self.state['pending'] = None

    def decide(self, gate, scores, late):
        cfg, c = self.cfg, self.state['champion']
        mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
        bench = {b: mean(scores.get(f"{gate['name']}:{b}", [])) for b in cfg['benchmark']}
        if c['name'] not in self.state['bench']:
            self.state['bench'][c['name']] = {b: mean(scores.get(f"{c['name']}:{b}", [])) for b in cfg['benchmark']}
        vs = scores.get(f"{gate['name']}:{c['name']}", [])
        h2h = mean(vs)
        mine = mean([v for v in bench.values() if v is not None])
        theirs = mean([v for v in self.state['bench'][c['name']].values() if v is not None])
        row = dict(name=gate['name'], champion=c['name'], h2h=h2h, games=len(vs), bench=bench, bench_mean=mine,
                   champion_bench_mean=theirs, at=now(), minutes=round((time.time() - gate['started']) / 60))
        if h2h is None or len(vs) < cfg['champion_games'] // 2:
            row['result'] = 'inconclusive'
        elif h2h >= cfg['promote'] and (mine is None or theirs is None or mine >= theirs - cfg['bench_drop']):
            row['result'] = 'promoted'
        else:
            row['result'] = 'kept champion'
        self.state['gates'].append(row)
        self.event(f"gate {gate['name']}: {row['result']} (vs {c['name']} {fmt(h2h)} over {len(vs)}, benchmark "
                   f"{fmt(mine)} vs champion's {fmt(theirs)}{', timed out' if late else ''})")
        if row['result'] == 'promoted':
            league = self.run / 'league' / f"{gate['name']}.pt"
            shutil.copyfile(gate['path'], league)
            self.state['champion'] = dict(name=gate['name'], path=str(league))
            ckpt = self.run / 'ckpt' / f"{gate['name']}.pt"
            self.state['champion_ckpt'] = f"ckpt/{gate['name']}.pt" if ckpt.exists() else None
            self.state['bench'][gate['name']] = bench
            self.state['fails'] = self.state['worse'] = 0
            self.write_champion()
            self.prune()
            return
        if row['result'] == 'inconclusive':
            return
        self.state['fails'] += 1
        self.state['worse'] = self.state['worse'] + 1 if h2h < cfg['worse'] else 0
        if self.state['worse'] >= cfg['rollback_after']:
            self.rollback(f"{self.state['worse']} straight gates below {cfg['worse']:.0%} against the champion")
        if self.state['fails'] >= cfg['stall_after'] and not self.state['stopped']:
            self.stop(f"{self.state['fails']} straight gates without a promotion")

    def rollback(self, why):
        self.state['worse'] = 0
        if not self.state['champion_ckpt']:
            self.stop(f'{why}, and the champion ({self.state["champion"]["name"]}) has no checkpoint to return to')
            return
        self.state['rollbacks'] += 1
        self.put('rollback', dict(id=f"r{self.state['rollbacks']}", path=self.state['champion_ckpt']))
        self.event(f"rollback {self.state['rollbacks']} to {self.state['champion_ckpt']}: {why}")

    def prune(self):
        """Full checkpoints are 0.2 GB each: keep the champion's and the newest `keep_ckpts`."""
        keep = {Path(self.state['champion_ckpt'] or '').name}
        ckpts = sorted((self.run / 'ckpt').glob('v*.pt'))
        for p in ckpts[:-self.cfg['keep_ckpts']]:
            if p.name not in keep:
                p.unlink()

    def health(self, rows):
        """A collapsed policy (the fork test's NaN: entropy 0, no cards played) -> rollback, at most once per update."""
        policy_rows = [r for r in rows if not r.get('value_only')][-3:]
        if len(policy_rows) == 3 and all(r.get('entropy', 1.0) < self.cfg['min_entropy'] for r in policy_rows):
            if self.state.get('collapse_seen') != policy_rows[-1]['version']:
                self.state['collapse_seen'] = policy_rows[-1]['version']
                self.rollback(f"policy entropy below {self.cfg['min_entropy']} for 3 updates")

    # ---- report ----
    def report(self, alive, rows):
        st, cfg = self.state, self.cfg
        last = rows[-1] if rows else {}
        recent = [r for r in rows if not r.get('value_only')] or rows
        avg = lambda k: sum(r.get(k, 0) for r in recent) / len(recent) if recent else 0  # noqa: E731
        lines = [f"# {self.run.name}: loop report ({now()})", '',
                 f"**Status:** {('STOPPED ' + st['stopped']) if st['stopped'] else 'running'}; "
                 f"{(time.time() - st['started']) / 3600:.1f} h since the loop started",
                 f"**Champion:** {st['champion']['name']} (the KL reference); rollbacks {st['rollbacks']}; "
                 f"gates without promotion {st['fails']} (stop at {cfg['stall_after']})", '']
        if last:
            lines += [f"**Learner:** version {last['version']}{' (critic warm-up)' if last.get('value_only') else ''}, "
                      f"update {last.get('update_s')} s, waits {last.get('wait_s')} s for data",
                      f"**Throughput (last {len(recent)} updates):** trained {avg('trained_games_per_hour') / 1000:.1f}k "
                      f"games/h, played {avg('games_per_hour') / 1000:.1f}k games/h (evaluation games "
                      f"{sum(r.get('eval_games', 0) for r in recent)}), stale {avg('stale'):.0f} per update",
                      f"**Health:** entropy {avg('entropy'):.3f}, approx_kl {avg('approx_kl'):.4f}, kl_ref "
                      f"{avg('kl_ref'):.4f}, critic win_ev {avg('win_ev'):.2f}, skipped steps {avg('skipped'):.3f} / "
                      f"critic {avg('critic_skipped'):.3f}, refused plays/game {avg('refused_per_game'):.2f}",
                      f"**Training scores:** {last.get('score')}", '']
        lines += ['**Jobs:** ' + ', '.join(f"{n} {st['jobs'].get(n)} {' '.join(a)} (restarts {st['restarts'].get(n, 0)})"
                                           for n, a in alive.items()), '']
        if st['pending']:
            sc = self.scores()
            lines.append(f"**Gate in progress:** {st['pending']['name']}: " + ', '.join(
                f"{m['id']} {len(sc.get(m['id'], []))}/{m['games']}" for m in st['pending']['matches']))
            lines.append('')
        if st['gates']:
            names = list(cfg['benchmark'])
            lines += ['| Gate | vs champion | Games | Benchmark mean (champion\'s) | ' + ' | '.join(names) +
                      ' | Result | At |', '|' + '---|' * (6 + len(names))]
            for g in st['gates'][-15:]:
                lines.append(f"| {g['name']} | {fmt(g['h2h'])} vs {g['champion']} | {g['games']} | "
                             f"{fmt(g['bench_mean'])} ({fmt(g['champion_bench_mean'])}) | " +
                             ' | '.join(fmt(g['bench'].get(n)) for n in names) + f" | {g['result']} | {g['at']} |")
        events = (self.run / 'loop-events.log')
        if events.exists():
            lines += ['', '**Recent events:**', '```', *events.read_text().splitlines()[-12:], '```']
        (self.run / 'report.md').write_text('\n'.join(lines) + '\n')

    def step(self):
        if self.state['stopped']:
            return False
        alive = self.jobs()
        rows = self.metrics()
        self.health(rows)
        self.gates(self.scores())
        if (time.time() - self.state['started']) / 3600 > self.cfg['max_hours'] and not self.state['stopped']:
            self.stop(f"the compute limit of {self.cfg['max_hours']} h")
        self.report(alive, rows)
        self.save()
        return not self.state['stopped']


def minutes(elapsed):
    """squeue's %M ([days-]hours:minutes:seconds or minutes:seconds) -> minutes."""
    days, _, clock = elapsed.rpartition('-')
    parts = [int(x) for x in clock.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days or 0) * 24 + parts[0]) * 60 + parts[1] + parts[2] / 60


def fmt(x):
    return '-' if x is None else f'{x:.1%}'


def main():
    run = Path(sys.argv[1]).expanduser().resolve()
    loop = Loop(run)
    loop.event('loop started')
    while loop.step():
        time.sleep(60)
    loop.report({}, loop.metrics())


if __name__ == '__main__':
    main()
