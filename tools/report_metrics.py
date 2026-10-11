"""Send a run's recent metrics to the repo's `reports` branch, so a collaborator can read them with `git fetch`.

    python tools/report_metrics.py runs/ppo-general               # once
    python tools/report_metrics.py runs/ppo-general --every 900   # every 15 minutes, until stopped

Writes reports/<run name>.jsonl (the last --lines rows of <run>/metrics.jsonl) and reports/<run name>.txt (host,
time, GPUs, latest row) in a separate clone (--clone, default ~/.cr-reports) on its own branch, then pushes that
branch. It never touches this checkout, its branches or unpushed commits. No network (campus GitHub down): the
commit waits and the next round pushes it. Contents: training metrics only (no data, models or paths outside the
run name).

Read them: git fetch origin reports && git show origin/reports:reports/<run name>.jsonl
"""
import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

BRANCH = 'reports'


def git(cwd, *args, check=True):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=check)


def prepare(repo: Path, clone: Path) -> None:
    """A clone of this repo's origin on the reports branch (made empty the first time)."""
    if not (clone / '.git').exists():
        origin = git(repo, 'remote', 'get-url', 'origin').stdout.strip()
        clone.parent.mkdir(parents=True, exist_ok=True)
        git(clone.parent, 'clone', '--quiet', '--no-checkout', origin, clone.name)
        if git(clone, 'ls-remote', '--exit-code', '--heads', 'origin', BRANCH, check=False).returncode == 0:
            git(clone, 'checkout', '--quiet', '-B', BRANCH, f'origin/{BRANCH}')
        else:
            git(clone, 'checkout', '--quiet', '--orphan', BRANCH)
            git(clone, 'rm', '-rf', '--quiet', '.', check=False)
    else:
        git(clone, 'fetch', '--quiet', 'origin', BRANCH, check=False)
        git(clone, 'rebase', '--quiet', f'origin/{BRANCH}', check=False)


def gpus() -> str:
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'no nvidia-smi'


def report(run: Path, clone: Path, lines: int) -> str:
    rows = (run / 'metrics.jsonl').read_text().splitlines()[-lines:] if (run / 'metrics.jsonl').exists() else []
    out = clone / 'reports'
    out.mkdir(exist_ok=True)
    (out / f'{run.name}.jsonl').write_text('\n'.join(rows) + ('\n' if rows else ''))
    latest = json.loads(rows[-1]) if rows else {}
    (out / f'{run.name}.txt').write_text(
        f'host {socket.gethostname()}\ntime {time.strftime("%Y-%m-%d %H:%M:%S %z")}\ngpus\n{gpus()}\n'
        f'latest\n{json.dumps(latest, indent=1)}\n')
    git(clone, 'add', 'reports')
    if git(clone, 'diff', '--cached', '--quiet', check=False).returncode == 0:
        return 'nothing new'
    git(clone, 'commit', '--quiet', '-m', f'{run.name}: update {latest.get("update", "?")}')
    pushed = git(clone, 'push', '--quiet', 'origin', BRANCH, check=False)
    return 'pushed' if pushed.returncode == 0 else f'committed, push failed (next round retries): {pushed.stderr.strip()[:200]}'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, help='the il.rl output directory (has metrics.jsonl)')
    parser.add_argument('--every', type=float, default=0, help='seconds between reports (0: once)')
    parser.add_argument('--lines', type=int, default=300, help='latest metrics rows to send')
    parser.add_argument('--clone', type=Path, default=Path.home() / '.cr-reports')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    while True:
        try:
            prepare(repo, args.clone)
            print(time.strftime('%H:%M:%S'), report(args.run.resolve(), args.clone, args.lines), flush=True)
        except (OSError, subprocess.CalledProcessError) as error:
            print(time.strftime('%H:%M:%S'), 'report failed:', getattr(error, 'stderr', '') or error, flush=True)
        if not args.every:
            return 0
        time.sleep(args.every)


if __name__ == '__main__':
    raise SystemExit(main())
