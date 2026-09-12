"""Show each lane's mode, occupancy, and per-slot progress.

Run from crack-cr: python -m native_engine.status [port ...]
Defaults to the four cluster lane ports.
"""
import json
import socket
import sys

DEFAULT_PORTS = (26789, 26790, 26791, 26792)


def status(port):
    with socket.create_connection(('127.0.0.1', port), timeout=5) as sock:
        session = sock.makefile('rb')
        sock.sendall(b'session-v1\n')
        session.readline()
        sock.sendall(b'multi-status\n')
        return json.loads(session.readline() or b'null')


def describe(port):
    try:
        reply = status(port)
    except OSError:
        return f'{port:>5}  no lane'
    if not reply:
        # adb accepts the connection but no engine listens on the device port.
        return f'{port:>5}  no lane (forward only)'
    mode = reply['mode']
    if mode != 'resident-headless':
        return f'{port:>5}  {mode:<16} occupied={reply["occupied"]}'
    slots = [slot for slot in reply['slots'] if not slot['ended']]
    live = f'{len(slots)} running' if len(slots) != reply['occupied'] else 'all running'
    ticks = sorted(slot['tick'] for slot in reply['slots'])
    span = f'ticks {ticks[0]}..{ticks[-1]}' if ticks else ''
    return (f'{port:>5}  {mode:<16} occupied={reply["occupied"]:>2}  '
            f'{live:<11} bound={reply["boundEnvId"]:>2}  {span}')


def main():
    ports = [int(arg) for arg in sys.argv[1:]] or DEFAULT_PORTS
    print(' port  mode              occupancy                    progress')
    for port in ports:
        print(describe(port))


if __name__ == '__main__':
    main()
