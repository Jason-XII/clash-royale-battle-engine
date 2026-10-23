"""Synthetic converted games for the tests: arrays() of one player's inputs and labels, dataset() of a few such games
written and sealed as convert.py does."""
import json

import numpy as np

import convert
from cr import observation as C


def arrays(size, marker):
    zeros = lambda *shape: np.zeros((size, *shape), np.float32)  # noqa: E731
    ids = lambda *shape: np.zeros((size, *shape), np.int64)      # noqa: E731
    x = dict(obj_card=ids(2), obj_data=ids(2), obj_side=ids(2), obj_kind=ids(2), obj_float=zeros(2, C.OBJ_FLOATS),
             eff_id=ids(2, 1), eff_left=zeros(2, 1), hand_card=np.full((size, 4), 2, np.int64), mirror_card=ids(4),
             hand_float=zeros(4, C.HAND_FLOATS), cycle_card=ids(4), ability=zeros(2, 4), own_flight_card=ids(),
             own_flight_float=zeros(4), opp_flight_card=ids(2), opp_flight_float=zeros(2, 4), opp_seen_card=ids(8),
             opp_seen_float=zeros(8, 3), opp_hero_card=ids(2), scalars=zeros(C.SCALARS),
             legal=np.ones((size, 4, C.H, C.W), bool))
    x['obj_side'][:, 0], x['obj_card'][:, 0] = 1, 2
    x['obj_float'][:, 0, 0] = np.arange(size) / size
    x['scalars'][:, 0], x['scalars'][:, 2] = marker / 10, 1
    y = dict(action_kind=ids(), action_card=ids(), action_cell=ids(), action_controller=ids(),
             label_ok=np.ones(size, bool), value=np.full(size, 1 if marker % 2 else -1, np.float32))
    y['action_kind'][::3], y['action_card'][::3] = 1, 2
    return x, y


def dataset(root, monkeypatch, sizes=(7, 11, 5)):
    """A converted dataset of synthetic games, written by convert.write_archive and sealed like convert.convert."""
    root.mkdir()
    entries = []
    for index, size in enumerate(sizes):
        tag = next(f'g{index}-{i}' for i in range(10000) if convert.held_out(f'g{index}-{i}') == (index == 2))
        monkeypatch.setattr(convert, 'game_features', lambda game, owner, vocab, legal: arrays(size, index + owner))
        path = convert.archive_path(root, tag)
        convert.write_archive(path, {}, C.Vocab([]), None, dict(tag=tag, kept=True))
        entries += convert.read_metadata(path)['entries']
    (root / 'manifest.json').write_text(json.dumps(dict(schema=dict(vocab=[26000000]), entries=entries)))
    return root
