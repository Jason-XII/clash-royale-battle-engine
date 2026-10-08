"""Self-play a checkpoint in the native engine with decks taken from human replays (Mac side).

    python -m il.play runs/il-001/best.pt data/IL_Replay/replays/part-000051.parquet --games 4
"""
import argparse
import json
from pathlib import Path

import torch

from native_engine import Engine, Play
from native_engine.replay import DECISION_TICKS, match_dict, prepare_collected_replay

from .agent import Agent, load


def play_game(engine, agents, match):
    state = engine.reset(match=match)
    for agent in agents:
        agent.reset()
    tried = {0: 0, 1: 0}
    done = {0: 0, 1: 0}
    while not state.finalized and state.tick < 7200:
        plays, abilities = [], []
        for agent in agents:
            choice = agent.act(state)
            if choice[0] == 'card':
                plays.append(Play(agent.owner, choice[1], choice[2], choice[3]))
            elif choice[0] == 'ability':
                abilities.append(Engine.ability_command(agent.owner, (), state.tick + 1))
            tried[agent.owner] += choice[0] != 'wait'
        if abilities:
            engine.schedule(abilities)  # ponytail: ability success is not counted in `done`
        transition = engine.step(plays, DECISION_TICKS)
        for play, ok in zip(plays, transition.executed):
            done[play.owner] += ok
        state = transition.state
    return {'result': state.result, 'crowns': state.crowns, 'tick': state.tick, 'actions': tried, 'executed': done}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('replays', type=Path, help='a parquet part; its decks are used')
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--argmax', action='store_true')
    parser.add_argument('--port', type=int, default=26789)
    args = parser.parse_args()
    torch.set_num_threads(2)
    import pyarrow.parquet as pq
    rows = pq.read_table(args.replays, columns=['payload_json']).slice(0, args.games).to_pylist()
    model, vocab = load(args.checkpoint)
    agents = [Agent(model, vocab, owner, sample=not args.argmax) for owner in (0, 1)]
    engine = Engine(port=args.port)
    for row in rows:
        replay = prepare_collected_replay(json.loads(row['payload_json'])).replay
        decks = (replay.episode_config.deck0, replay.episode_config.deck1)
        print(json.dumps(play_game(engine, agents, match_dict(replay, decks, replay.episode_config.seed))), flush=True)


if __name__ == '__main__':
    main()
