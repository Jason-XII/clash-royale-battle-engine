"""A window with a blue/red bar: the live model's predicted win chance, (value + 1) / 2, from its newest log in DIR.
    python winbar.py [DIR]    (default ~/cr-live/runs/live; run next to play.py)"""
import json
import sys
import tkinter as tk
from pathlib import Path

LOGS = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / 'cr-live/runs/live')
W, H = 90, 600
root = tk.Tk()
root.title('P(win)')
root.attributes('-topmost', True)
canvas = tk.Canvas(root, width=W, height=H + 40, bg='black', highlightthickness=0)
canvas.pack()
state = {'path': None, 'file': None}


def poll():
    newest = max(LOGS.glob('*.jsonl'), key=lambda p: p.stat().st_mtime, default=None)
    if newest != state['path']:  # a new game: follow its log from the start
        state['path'], state['file'] = newest, newest and newest.open()
    value = None
    f = state['file']
    while f:
        at, line = f.tell(), f.readline()
        if not line.endswith('\n'):  # nothing new, or a line still being written: read it whole next time
            f.seek(at)
            break
        event = json.loads(line)
        if event.get('event') == 'decision' and 'value' in event:
            value = event['value']
    if value is not None:
        p = min(max((value + 1) / 2, 0.0), 1.0)
        canvas.delete('all')
        canvas.create_rectangle(0, 0, W, H * (1 - p), fill='#d03030', width=0)  # red: the opponent's share, on top
        canvas.create_rectangle(0, H * (1 - p), W, H, fill='#3060e0', width=0)  # blue: ours
        canvas.create_line(0, H / 2, W, H / 2, fill='white', dash=(4, 4))
        canvas.create_text(W / 2, H + 20, text=f'{100 * p:.0f}%', fill='white', font=('Helvetica', 20, 'bold'))
    root.after(200, poll)


poll()
root.mainloop()
