from native_engine import Play, RenderedEngine

engine = RenderedEngine()
state = engine.reset(seed=41)
transition = engine.step(
    [Play(owner=0, slot=0, x=3500, y=13000)],
    ticks=20,
)
engine.resume()  # Continue in real time