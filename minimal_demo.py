"""Run from crack-cr: python -m native_engine.minimal_demo [port]."""
import sys

from . import Engine, Observer, Play

port = int(sys.argv[1]) if len(sys.argv) > 1 else 26789
engine = Engine(port=port)
observer = Observer(owner=0)
print(observer.observe(engine.reset(seed=1)))
transition = engine.step([Play(owner=0, slot=0, x=3500, y=13000)], ticks=100)
print('Executed:', transition.executed)
print(observer.observe(transition.state).entities)
terminal = engine.step(ticks=7000).state
print('Finalized:', terminal.finalized, 'result:', terminal.result, 'crowns:', terminal.crowns)
