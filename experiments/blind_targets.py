"""What the live target bug cost (23 Oct): arena.py with model A's target inputs at 0 (has_target, target_dx and
target_dy of every object, as play.py gave them before 0f8200b), B unchanged. Same arguments as arena.py; A against
itself measures the bug alone.

    python experiments/blind_targets.py V240.pt V240.pt --engine ~/clapha-engine --pool deck_pool.json --games 100 \\
        --out OUT.jsonl --a-temperature 0.05 --b-temperature 0.05 --device cuda
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arena  # noqa: E402
from cr import observation as C  # noqa: E402
from cr import policy  # noqa: E402

TARGET = [C.OBJ_FLOAT_NAMES.index(n) for n in ('has_target', 'target_dx', 'target_dy')]
loaded = []


def load(path, device):
    model = policy.load(path, device)
    loaded.append(model[0])  # arena.py loads A first, then B (separate objects even for one checkpoint)
    return model


class Blind(policy.Agent):
    def act(self, inputs):
        if self.model is loaded[0]:
            inputs = dict(inputs, obj_float=inputs['obj_float'].copy())
            inputs['obj_float'][..., TARGET] = 0.0
        return super().act(inputs)


if __name__ == '__main__':
    arena.load, arena.Agent = load, Blind
    arena.main()
