#!/usr/bin/env bash
# Pause PID (SIGSTOP) while the hottest die sensor is above HIGH °C; resume (SIGCONT) below LOW.
# Exits when PID ends.  Usage: tools/heat_guard.sh PID [HIGH=70] [LOW=62]
PID=$1; HIGH=${2:-70}; LOW=${3:-62}
TEMP="$(dirname "$0")/cputemp"
paused=0
while kill -0 "$PID" 2>/dev/null; do
  t=$("$TEMP")
  if [ $paused = 0 ] && awk "BEGIN{exit !($t > $HIGH)}"; then
    kill -STOP "$PID"; paused=1; echo "$(date +%T) ${t}C > ${HIGH}C: paused $PID"
  elif [ $paused = 1 ] && awk "BEGIN{exit !($t < $LOW)}"; then
    kill -CONT "$PID"; paused=0; echo "$(date +%T) ${t}C < ${LOW}C: resumed $PID"
  fi
  sleep 5
done
