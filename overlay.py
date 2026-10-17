"""Chess-style marks over the emulator's arena: 0.5 s after each play lands (ours or the opponent's), the change in the
mover's win chance ((value + 1) / 2 from the live log) gives !! / ! / ? / ??, drawn above the tile it was played on.

A transparent click-through panel over the arena, placed by the 18 x 32 grid profile of crack-cr/queue_overlay
(calibrate with its arena_overlay.py, which saves the profile). Reads the newest log in ~/cr-live/runs/live.
Needs PyObjC: /opt/anaconda3/bin/python overlay.py [--profile P] [--logs DIR]"""
import argparse
import json
import time
from pathlib import Path

JUDGE_AFTER = 10    # ticks after landing (0.5 s): the value head has seen the new unit, the other side not yet answered
BEFORE = 25         # ticks before landing: the win chance the play started from
SHOW_S = 3.0
# (lowest change for the side that played, mark, rgb). Measured 2026-10-17 on today's live games, 0.5 s after landing:
# plays sd 0.076, quiet windows sd 0.028 (2.7x; 5 s after was only 2.1x); these cuts mark ~28% of plays
# (!! 3%, ! 11%, ? 11%, ?? 3%)
MARKS = ((0.15, '!!', (0.11, 0.73, 0.73)), (0.07, '!', (0.36, 0.69, 0.25)),
         (-0.07, None, None), (-0.15, '?', (0.95, 0.60, 0.10)), (-9.0, '??', (0.86, 0.16, 0.13)))


def mark(change):
    """The mover's change in win chance -> (mark, rgb), or (None, None) for an ordinary play."""
    for low, symbol, rgb in MARKS:
        if change >= low:
            return symbol, rgb
    return None, None


def screen_fraction(world, owner):
    """World units -> (x from the left, y from the bottom) as fractions of the arena; our side is at the bottom and owner 0
    sees x mirrored (play.world_screen)."""
    x, y = world
    vx, vy = (18 - x / 1000, y / 1000) if owner == 0 else (x / 1000, 32 - y / 1000)
    return vx / 18, vy / 32


class Judge:
    """Follows one live log's events -> finished judgements (mark, rgb, world, owner, change, card)."""

    def __init__(self):
        self.owner, self.values, self.plays = None, [], []

    def p_at(self, tick):
        """Our win chance at the last decision at or before `tick` (the first one if none)."""
        best = self.values[0][1] if self.values else None
        for t, p in self.values:
            if t > tick:
                break
            best = p
        return best

    def feed(self, event):
        out = []
        kind = event.get('event')
        if kind == 'roster':
            self.owner = event['owner']
        elif kind == 'decision' and 'value' in event:
            self.values.append((event['tick'], (event['value'] + 1) / 2))
            while self.plays and event['tick'] >= self.plays[0]['tick'] + JUDGE_AFTER:
                play = self.plays.pop(0)
                change = self.values[-1][1] - self.p_at(play['tick'] - BEFORE)
                if play['event'] == 'opponent':
                    change = -change  # judged for the side that played
                symbol, rgb = mark(change)
                if symbol and self.owner is not None:
                    out.append((symbol, rgb, play['world'], self.owner, change, play['card']))
        elif kind in ('landed', 'opponent') and event.get('world'):
            self.plays.append(event)
            self.plays.sort(key=lambda e: e['tick'])
        return out


def main():
    from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSBezierPath,
                        NSColor, NSFont, NSFontAttributeName, NSForegroundColorAttributeName, NSMakePoint, NSMakeRect,
                        NSPanel, NSScreen, NSStatusWindowLevel, NSTimer, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
                        NSWindowStyleMaskBorderless)
    from Foundation import NSObject, NSString

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument('--profile', type=Path, default=here.parent / 'crack-cr/queue_overlay/arena_grid_profile.json')
    parser.add_argument('--logs', type=Path, default=Path.home() / 'cr-live/runs/live')
    args = parser.parse_args()
    box = json.loads(args.profile.read_text())['frame_top_left']
    state = {'path': None, 'file': None, 'judge': Judge(), 'shown': []}

    class View(NSView):
        def drawRect_(self, _rect):
            w, h = self.bounds().size.width, self.bounds().size.height
            now = time.monotonic()
            for symbol, rgb, world, owner, change, _card, until in state['shown']:
                if until < now:
                    continue
                fx, fy = screen_fraction(world, owner)
                cx, cy, r = fx * w, fy * h + h / 32 * 1.6, max(11.0, w / 18 * 0.85)  # just above the tile
                NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgb, 0.95).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - r, cy - r, 2 * r, 2 * r)).fill()
                text = NSString.stringWithString_(symbol)
                attrs = {NSFontAttributeName: NSFont.boldSystemFontOfSize_(r * 1.05),
                         NSForegroundColorAttributeName: NSColor.whiteColor()}
                size = text.sizeWithAttributes_(attrs)
                text.drawAtPoint_withAttributes_(NSMakePoint(cx - size.width / 2, cy - size.height / 2), attrs)
                small = NSString.stringWithString_(f'{100 * change:+.0f}%')
                sattrs = {NSFontAttributeName: NSFont.boldSystemFontOfSize_(r * 0.6),
                          NSForegroundColorAttributeName: NSColor.whiteColor()}
                ssize = small.sizeWithAttributes_(sattrs)
                small.drawAtPoint_withAttributes_(NSMakePoint(cx - ssize.width / 2, cy + r), sattrs)

    class Pump(NSObject):
        def tick_(self, _timer):
            newest = max(args.logs.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, default=None)
            if newest != state['path']:  # a new game: a fresh judge, from the log's start
                state.update(path=newest, file=newest and newest.open(), judge=Judge())
            f, changed = state['file'], False
            while f:
                at, line = f.tell(), f.readline()
                if not line.endswith('\n'):
                    f.seek(at)
                    break
                for judged in state['judge'].feed(json.loads(line)):
                    state['shown'].append((*judged, time.monotonic() + SHOW_S))
                    changed = True
            now = time.monotonic()
            live = [s for s in state['shown'] if s[-1] >= now]
            if changed or len(live) != len(state['shown']):
                state['shown'] = live
                panel.contentView().setNeedsDisplay_(True)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    screen_h = NSScreen.mainScreen().frame().size.height
    frame = NSMakeRect(box['x'], screen_h - box['y'] - box['height'], box['width'], box['height'])
    panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(frame, NSWindowStyleMaskBorderless,
                                                                          NSBackingStoreBuffered, False)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(NSColor.clearColor())
    panel.setHasShadow_(False)
    panel.setLevel_(NSStatusWindowLevel)
    panel.setIgnoresMouseEvents_(True)  # click-through
    panel.setHidesOnDeactivate_(False)
    panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces)
    panel.setContentView_(View.alloc().initWithFrame_(NSMakeRect(0, 0, box['width'], box['height'])))
    panel.orderFrontRegardless()
    pump = Pump.alloc().init()
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.1, pump, 'tick:', None, True)
    app.run()


if __name__ == '__main__':
    main()
