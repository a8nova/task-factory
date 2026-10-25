#!/bin/bash
# Prints how much of the task's time budget is left. The clock starts when this container started.
python3 - <<'PY'
import os
hz = os.sysconf("SC_CLK_TCK")
start = int(open("/proc/1/stat").read().rsplit(")", 1)[1].split()[19]) / hz
elapsed = float(open("/proc/uptime").read().split()[0]) - start
budget = int(os.environ.get("TF_AGENT_TIMEOUT", "3600"))
left = max(0, int(budget - elapsed))
print("time left: %d s (%d min) of %d s; used %d s" % (left, left // 60, budget, int(elapsed)))
PY
