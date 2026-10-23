"""The improvement loop: gates, promotion, rollback, stopping, job accounting."""
import argparse
import json

import pytest

import loop


def test_loop_promotes_rolls_back_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(loop.Loop, 'jobs', lambda self: {})
    for d in ('pool', 'ckpt', 'league'):
        (tmp_path / d).mkdir()
    (tmp_path / 'loop.json').write_text(json.dumps(dict(
        jobs={}, champion=dict(name='il', path='/il.pt'), benchmark={'b1': '/b1.pt'}, champion_games=4, bench_games=2,
        stall_after=3)))
    run = loop.Loop(tmp_path)
    assert json.loads((tmp_path / 'champion.json').read_text())['name'] == 'il'

    def gate(name, h2h, bench, champion_bench=None):
        (tmp_path / 'pool' / f'{name}.pt').write_bytes(b'x')
        (tmp_path / 'ckpt' / f'{name}.pt').write_bytes(b'x')
        run.step()
        ids = [m['id'] for m in json.loads((tmp_path / 'eval.json').read_text())['matches']]
        rows = [dict(eval=f'{name}:{run.state["champion"]["name"]}', score=x) for x in h2h]
        rows += [dict(eval=f'{name}:b1', score=x) for x in bench]
        rows += [dict(eval=f'il:b1', score=x) for x in (champion_bench or [])]
        with (tmp_path / 'evals.jsonl').open('a') as f:
            f.write(''.join(json.dumps(r) + '\n' for r in rows))
        run.step()
        assert not json.loads((tmp_path / 'eval.json').read_text())['matches'], 'evaluation over: back to training'
        return ids, run.state['gates'][-1]['result']

    ids, result = gate('v000020', [1, 1, 1, 0], [1, 0], champion_bench=[0.5, 0.5])
    assert set(ids) == {'v000020:il', 'v000020:b1', 'il:b1'} and result == 'promoted'
    assert json.loads((tmp_path / 'champion.json').read_text())['name'] == 'v000020'
    assert (tmp_path / 'league' / 'v000020.pt').exists() and run.state['champion_ckpt'] == 'ckpt/v000020.pt'
    assert gate('v000040', [1, 1, 1, 1], [0, 0])[1] == 'kept champion', 'a benchmark regression blocks promotion'
    assert gate('v000060', [0, 0, 0, 1], [1, 1])[1] == 'kept champion' and not (tmp_path / 'rollback.json').exists()
    assert gate('v000080', [0, 0, 0, 0], [1, 1])[1] == 'kept champion'
    assert json.loads((tmp_path / 'rollback.json').read_text()) == dict(id='r1', path='ckpt/v000020.pt')
    assert run.state['stopped'], 'three gates without a promotion: stalled'
    assert 'v000080' in (tmp_path / 'report.md').read_text()


@pytest.mark.parametrize('bench', [0, 1])
def test_a_timed_out_gate_without_benchmarks_promotes_nothing(tmp_path, monkeypatch, bench):
    monkeypatch.setattr(loop.Loop, 'jobs', lambda self: {})
    for d in ('pool', 'ckpt', 'league'):
        (tmp_path / d).mkdir()
    (tmp_path / 'loop.json').write_text(json.dumps(dict(
        jobs={}, champion=dict(name='il', path='/il.pt'), benchmark={'b1': '/b1.pt'}, champion_games=4, bench_games=2)))
    run = loop.Loop(tmp_path)
    (tmp_path / 'pool' / 'v000020.pt').write_bytes(b'x')
    run.step()
    with (tmp_path / 'evals.jsonl').open('a') as f:  # the head-to-head is won, 0 or 1 of each benchmark's 2 games finish
        f.write(''.join(json.dumps(dict(eval=e, score=1)) + '\n' for e in ['v000020:il'] * 4 + ['v000020:b1', 'il:b1'] * bench))
    run.state['pending']['started'] -= 10 ** 6
    run.step()
    assert run.state['gates'][-1]['result'] == 'inconclusive' and run.state['champion']['name'] == 'il'
    assert 'il' not in run.state['bench'], 'a partial baseline is not cached: the next gate measures it again'
    (tmp_path / 'loop.json').write_text(json.dumps(dict(
        jobs={}, champion=dict(name='il', path='/il.pt'), benchmark={'v000200': '/other-run/pool/v000200.pt'})))
    with pytest.raises(SystemExit):
        loop.Loop(tmp_path)  # another run's v000200 would share a name with this run's own snapshot


def test_a_job_counts_as_ended_only_when_accounting_says_so(monkeypatch):
    answers = [('JobId=1 JobState=CANCELLED Reason=None', '', True), ('JobId=1 JobState=RUNNING', '', False),
               ('', 'slurm_load_jobs error: Invalid job id specified', True),
               ('', 'sbatch: fatal: Unable to process configuration file', False), ('', '', False)]
    for out, err, want in answers:
        monkeypatch.setattr(loop.subprocess, 'run', lambda *a, **k: argparse.Namespace(stdout=out, stderr=err))
        assert loop.ended(1) is want, (out, err)
