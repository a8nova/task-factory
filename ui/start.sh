#!/bin/sh
# Start the monitor in the background, detached from this terminal, and keep it up: if it ever exits it is
# started again. It restarts itself when ui/server.py changes; open pages reload when ui/index.html changes.
#   ui/start.sh          start (does nothing if it is already running)
#   ui/start.sh stop     stop it
# Log: logs/ui.log    Page: http://127.0.0.1:8777
cd "$(dirname "$0")/.." || exit 1
PIDFILE=logs/ui.pid
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
if [ "$1" = "stop" ]; then
  if running; then pkill -TERM -g "$(cat "$PIDFILE")" 2>/dev/null || kill "$(cat "$PIDFILE")"; fi
  rm -f "$PIDFILE"; echo "monitor stopped"; exit 0
fi
if running; then echo "monitor already running (pid $(cat "$PIDFILE")): http://127.0.0.1:8777"; exit 0; fi
mkdir -p logs
nohup python3 -c '
import os, subprocess, sys, time
os.setsid()  # its own session, so closing the terminal does not stop it
with open("logs/ui.pid", "w") as f:
    f.write(str(os.getpid()))
while True:
    code = subprocess.call([sys.executable, "ui/server.py"] + sys.argv[1:])
    print(time.strftime("%H:%M:%S"), "monitor exited with", code, "- starting it again in 3s", flush=True)
    time.sleep(3)
' "$@" >> logs/ui.log 2>&1 &
sleep 2
if running; then echo "monitor running (pid $(cat "$PIDFILE")): http://127.0.0.1:8777  log: logs/ui.log"
else echo "monitor did not start; see logs/ui.log"; exit 1; fi
