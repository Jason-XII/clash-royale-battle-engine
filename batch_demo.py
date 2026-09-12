"""Run from crack-cr: python -m native_engine.batch_demo [port]

Sixteen battles in one lane, advanced together. Leaves the lane in resident
mode; call native_engine.stop_resident(port) to return it to single-battle mode.
"""
import sys

from . import Batch, Play

battles = 16
port = int(sys.argv[1]) if len(sys.argv) > 1 else 26789
batch = Batch(battles=battles, port=port)
try:
    states = batch.reset(seed=1)
    print('tick after reset:', [state.tick for state in states[:4]], '...')
    actions = [(Play(owner=0, slot=0, x=3500, y=13000),) for _ in range(battles)]
    states = batch.step(actions, ticks=100)
    print('entities in battle 0:', states[0].entities)
    states = batch.step(ticks=7000)
    print('finalized:', [state.finalized for state in states[:4]], '...')
    print('result:', [state.result for state in states[:4]], 'crowns:', [state.crowns for state in states[:4]])
finally:
    batch.close()
