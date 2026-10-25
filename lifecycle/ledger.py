"""The ledger: every stage verdict for a task, append-only, at runs/ledger/<task>.jsonl.

Each line is one verdict (see CONTRACT.md) plus the task, stage number and name, round and timestamp.
"""
import datetime as dt
import json

from common import RUNS

LEDGER = RUNS / "ledger"


def path(task):
    LEDGER.mkdir(parents=True, exist_ok=True)
    return LEDGER / f"{task}.jsonl"


def append(task, stage, name, verdict, round_=1):
    # The verdict's own "stage" field is a name; the ledger keys stages by number.
    entry = {**verdict, "ts": dt.datetime.now().isoformat(timespec="seconds"), "task": task, "stage": stage,
             "stage_name": name, "round": round_}
    with path(task).open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def history(task, profile=None):
    """All entries, or only one profile's (entries written before profiles existed count as "full")."""
    p = path(task)
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if profile is None or r.get("profile", "full") == profile]


def latest(task, profile=None):
    """Latest verdict per stage number."""
    out = {}
    for entry in history(task, profile):
        out[entry["stage"]] = entry
    return out


def current_stage(task, last_stage=16, profile=None):
    """The first stage, in order, whose latest verdict has not passed (None when every stage has passed)."""
    seen = latest(task, profile)
    for stage in range(2, last_stage + 1):
        entry = seen.get(stage)
        if entry is None or not entry.get("passed"):
            return stage
    return None
