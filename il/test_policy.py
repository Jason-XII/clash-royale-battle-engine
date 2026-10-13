"""il.policy and il.rl's update on converted games (no engine needed):

    CLAPHA_IL=~/clapha-il-dev python -m il.test_policy CHECKPOINT GAMES      # GAMES: convert.py's OUT/games

- packed inputs (present objects only, legal tiles as bits) give the model the same outputs as the full inputs;
- the log-probs evaluate() gives the actions sample() drew equal sample()'s own;
- one PPO update runs, starting at ratio 1 and KL 0 against itself.
"""
import argparse
import copy
import glob
import itertools
import sys

import numpy as np
import torch

from . import policy
from .rl import evaluate, update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('games')
    args = ap.parse_args()
    sys.path.insert(0, str(policy.CLAPHA_IL))
    from convert import read_games
    torch.manual_seed(0)
    model, vocab, _ = policy.load(args.checkpoint)
    legal = policy.legal_fn()
    games = list(itertools.islice(read_games(sorted(glob.glob(f'{args.games}/*/*.bin'))[0]), 3))
    trajectories = []
    for g, owner in itertools.product(games, (0, 1)):
        x, _, _ = policy.FE.game_features(g, owner, vocab, legal=legal)
        x = {k: v[:300] for k, v in x.items()}
        T = len(x['scalars'])
        rows = [policy.pack({k: v[t] for k, v in x.items()}) for t in range(T)]
        inputs = policy.stack(rows)
        packed = policy.tensors({k: v[None] for k, v in inputs.items()}, 'cpu')
        full = {k: torch.from_numpy(v[None]) for k, v in x.items()}
        with torch.no_grad():
            a, b = model(full), model(packed)
            for k in ('what', 'which', 'value'):
                assert torch.allclose(a[k], b[k], atol=1e-4, equal_nan=True), k
            action, logp = policy.sample(model, b, packed, policy.ABILITY_EXPLORE)  # as the decision servers
            again = evaluate(model, packed, action[None], torch.ones(1, T, dtype=torch.bool))
        assert torch.allclose(again['logp'], logp, atol=1e-4), (again['logp'] - logp).abs().max()
        assert torch.isfinite(logp).all()
        reward = np.zeros(T, np.float32)
        reward[-1] = 1.0 if owner == 0 else -1.0
        trajectories.append(dict(inputs=inputs, action=action.numpy(), logp=logp.numpy(), reward=reward,
                                 value=b['value'][0].numpy()))
    plays = sum(int((t['action'] >= policy.CARD).sum()) for t in trajectories)
    print(f'{len(trajectories)} sequences: packed inputs give the same outputs; {plays} plays sampled, log-probs agree')
    reference = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    args = argparse.Namespace(gamma=0.999, lam=0.95, minibatch=2, epochs=1, clip=0.2, value_coef=0.5, entropy=0.002,
                              kl_coef=0.05)
    model.train()
    stats = update(model, reference, opt, trajectories, args, torch.device('cpu'))
    print({k: round(v, 5) for k, v in stats.items()})
    assert all(np.isfinite(v) for v in stats.values())
    assert stats['kl_ref'] < 1e-3 and stats['approx_kl'] < 1e-3, 'the first update starts from the reference'
    print('ok')


if __name__ == '__main__':
    main()
