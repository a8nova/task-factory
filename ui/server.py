"""Task Factory monitor: one page that shows every stage, every agent and what is running where, read live from
the ledger, the run folders and the logs on the laptop, plus the gateway's log and processes on the Mac mini.

    python3 ui/server.py                  # then open http://127.0.0.1:8777
    python3 ui/server.py --no-remote      # don't ssh to the Mac mini
    ui/start.sh                           # in the background, restarted if it exits; log in logs/ui.log

It never calls Gemini and sends only GET /health to the gateways. It writes only when you add a repo on the page:
the clone goes to src/repos/ and its digest to runs/repos/ (see ui/repos.py).
Hidden prompts never leave the Mac mini: quoted prompt text in its log is replaced before it reaches the page.
Plain stdlib, so the system python3 (3.9) runs it as well as the project's venv.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

UI = Path(__file__).resolve().parent
ROOT = UI.parent
RUNS, LOGS = ROOT / "runs", ROOT / "logs"
HOME = str(Path.home())
sys.path.insert(0, str(UI))
import repos  # noqa: E402  (ui/repos.py: digest a repo from a link, split it into chunks)

# Mirrors factory.STAGES (plus stage 1, Mine). `where` names the machines a stage uses (see NODES in index.html).
STAGES = {
    1: ("mine", "Mine", "Argon reads one port's code and git history and picks real speedup commits", ["gemini"]),
    2: ("propose", "Propose", "Argon writes the task proposal", ["gemini"]),
    3: ("proposal_qa", "Proposal QA", "Flash-Lite checks the proposal against the rules", ["gemini"]),
    4: ("review_proposal", "Proposal review", "Argon reviews the proposal", ["gemini"]),
    5: ("implement", "Implement", "Argon writes the Harbor task package", ["gemini"]),
    6: ("static_checks", "Static checks", "Package layout and rule checks", []),
    7: ("noop", "No-op", "The unchanged code must score 0 on the phone", ["cloud", "fleet"]),
    8: ("baseline", "Baseline", "The starter solution must score about 0", ["cloud", "fleet"]),
    9: ("implementation_qa", "Implementation QA", "Argon audits the task package", ["gemini"]),
    10: ("checkin_review", "Check-in review", "Argon checks the package against the ledger", ["gemini"]),
    11: ("explore", "Explore", "A Flash agent tries the task end to end", ["sandbox", "gemini", "cloud", "fleet"]),
    12: ("calibrate", "Calibrate", "Measure base and reference on the phone; set the target", ["sandbox", "gemini", "cloud", "fleet"]),
    13: ("redteam", "Red team + fixer", "Argon agents try to cheat the grader; the fixer patches it", ["sandbox", "gemini", "cloud", "fleet"]),
    14: ("rollouts", "Rollouts", "Target-model agents solve it; check the difficulty", ["sandbox", "gemini", "cloud", "fleet"]),
    15: ("reference", "Reference", "The real commit must score 1 on the phone", ["cloud", "fleet"]),
    16: ("final_signoff", "Final sign-off", "Argon approves the task for shipping", ["gemini"]),
}
ROLE_STAGE = {"explorer": 11, "calibrator": 12, "redteam": 13, "rollout": 14, "solver": 14}
# Mirrors agents.ROLES: time budget, validation calls and submissions per agent.
ROLE_LIMITS = {"redteam": (1800, 2, 2), "solver": (3600, 6, 1), "explorer": (900, 2, 1), "calibrator": (7200, 10, 1),
               "rollout": (3600, 6, 1)}
GATEWAYS = [{"url": "https://tf-gateway-507690274413.us-central1.run.app", "label": "cloud"}]
PASSED = {"pass", "approve"}

SALT = secrets.token_hex(8)  # prompt labels are stable within one server run only


def redact(text):
    text = str(text).replace(HOME, "~")
    text = re.sub(r"/Users/[^/\s'\"]+", "~", text)
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[key]", text)


def redact_remote(text):
    """The Mac mini's log quotes prompt prefixes, some of them hidden prompts: show a per-run label instead."""
    def label(m):
        return "prompt-" + hashlib.sha1((SALT + m.group(1)).encode()).hexdigest()[:4]
    return re.sub(r"'([^']{1,120})'(?=:)", label, redact(text))


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(path):
    out = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0


def trim(obj, depth=0):
    """Details for the page: strings shortened, long lists cut, nothing nested too deep."""
    if depth > 4:
        return "…"
    if isinstance(obj, dict):
        return {k: trim(v, depth + 1) for k, v in list(obj.items())[:40] if k not in ("interaction", "interactions")}
    if isinstance(obj, list):
        return [trim(v, depth + 1) for v in obj[:30]]
    if isinstance(obj, str):
        return redact(obj[:1500])
    return obj


def clock(hms, ref):
    """A log line's HH:MM:SS as epoch seconds on the date of `ref` (the log file's mtime)."""
    try:
        d = dt.datetime.fromtimestamp(ref).date()
        t = dt.datetime.strptime(hms, "%H:%M:%S").time()
        return dt.datetime.combine(d, t).timestamp()
    except ValueError:
        return None


def etime_seconds(text):
    days, _, rest = text.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


# --- local processes and logs ---------------------------------------------------------------------------------

def local_procs():
    try:
        out = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, cmd = parts
        argv = cmd.split()
        exe = os.path.basename(argv[0])
        if exe == "ssh" and "-L" in argv:
            fwd = next((a for a in argv if re.match(r"^(127\.0\.0\.1:)?\d+:", a)), "")
            m = re.match(r"^(?:127\.0\.0\.1:)?(\d+):[^:]+:(\d+)$", fwd)
            if m:
                procs.append({"pid": int(pid), "seconds": etime_seconds(etime), "kind": "tunnel",
                              "local": int(m.group(1)), "remote": int(m.group(2)), "host": argv[-1]})
            continue
        if not exe.lower().startswith("python") or len(argv) < 2:
            continue
        script = next((a for a in argv[1:] if a.endswith(".py")), "")
        name = os.path.basename(script)
        if name not in ("factory.py", "agents.py", "cut.py", "mine.py", "pack.py", "reproduce.py", "devrun_google.py",
                        "devrun.py", "gateway.py"):
            continue
        args = argv[argv.index(script) + 1:] if script in argv else []
        procs.append({"pid": int(pid), "seconds": etime_seconds(etime), "kind": name[:-3], "args": redact(" ".join(args)),
                      "argv": args})
    return procs


def arg(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        return argv[i + 1] if i + 1 < len(argv) else None
    return None


class LogTail:
    """Follows every logs/*.log; keeps the newest lines from all of them, merged by time."""
    LINE = re.compile(r"^(\d\d:\d\d:\d\d) (\w+) (.*)$")

    def __init__(self, keep=400):
        self.offsets, self.lines, self.keep, self.lock = {}, [], keep, threading.Lock()

    def poll(self):
        new = []
        for path in sorted(LOGS.glob("*.log")):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            off = self.offsets.get(path)
            if off is None:  # first look: the last few lines of logs touched in the last 6 hours
                off = max(0, size - 6000) if time.time() - mtime(path) < 6 * 3600 else size
            if size < off:
                off = 0
            if size == off:
                self.offsets[path] = off
                continue
            with path.open("rb") as f:
                f.seek(off)
                chunk = f.read(size - off).decode("utf-8", "replace")
            if off and not chunk.endswith("\n"):
                cut = chunk.rfind("\n") + 1
                chunk = chunk[:cut]
            self.offsets[path] = off + len(chunk.encode("utf-8"))
            ref = mtime(path)
            for raw in chunk.splitlines():
                m = self.LINE.match(raw)
                if not m:
                    continue
                at = clock(m.group(1), ref)
                new.append({"at": at, "src": path.stem, "where": "laptop", "level": m.group(2),
                            "text": redact(m.group(3))[:600]})
        if new:
            self.add(new)

    def add(self, entries):
        with self.lock:
            self.lines.extend(entries)
            self.lines.sort(key=lambda e: e["at"] or 0)
            del self.lines[:-self.keep]

    def recent(self, n=160):
        with self.lock:
            return list(self.lines[-n:])


# --- the Mac mini, the gateways and the phone --------------------------------------------------------------------

class Remote:
    """ssh to the Mac mini: follow its gateway log, and every few seconds list what is running there."""
    PROBE = ("ps -axo pid=,etime=,command= | grep -E 'gateway.py|devrun|cmake|ninja|clang|adb (-s [^ ]+ )?shell' "
             "| grep -v grep; echo ---; lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -E ':878[0-9]' ; echo ---; "
             "ls ~/task-factory/phone.lock >/dev/null 2>&1 && echo lockfile")

    def __init__(self, logs, host="macmini", enabled=True):
        self.logs, self.host, self.enabled = logs, host, enabled
        self.state = {"enabled": enabled, "reachable": None, "checked": None, "error": None, "gateways": [],
                      "listening": [], "builds": [], "phone_runs": [], "procs": 0}
        self.lock = threading.Lock()
        if enabled:
            threading.Thread(target=self.follow, daemon=True).start()
            threading.Thread(target=self.probe_loop, daemon=True).start()

    def ssh(self, command, **kw):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=15", self.host, command]

    def follow(self):
        while True:
            try:
                p = subprocess.Popen(self.ssh("tail -n 60 -F ~/task-factory/logs/devrun.log"), stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
                for raw in p.stdout:
                    m = LogTail.LINE.match(raw.rstrip("\n"))
                    if not m or "GET /health" in raw:
                        continue
                    self.logs.add([{"at": clock(m.group(1), time.time()), "src": "devrun", "where": "macmini",
                                    "level": m.group(2), "text": redact_remote(m.group(3))[:600]}])
            except OSError:
                pass
            time.sleep(10)

    def probe_loop(self):
        while True:
            self.probe()
            time.sleep(6)

    def probe(self):
        try:
            r = subprocess.run(self.ssh(self.PROBE), capture_output=True, text=True, timeout=20)
            ok, out, err = r.returncode in (0, 1), r.stdout, r.stderr.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            ok, out, err = False, "", str(exc)
        state = {"enabled": True, "reachable": ok and "---" in out, "checked": time.time(),
                 "error": None if ok else redact(err)[:300], "gateways": [], "listening": [], "builds": [],
                 "phone_runs": [], "lockfile": "lockfile" in out}
        procs, listen = (out.split("---") + ["", ""])[:2]
        for line in procs.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, etime, cmd = parts
            entry = {"pid": int(pid), "seconds": etime_seconds(etime)}
            if "gateway.py" in cmd:
                state["gateways"].append(entry)
            elif re.search(r"\badb\b.*\bshell\b", cmd):
                m = re.search(r"/data/local/tmp/tf/([^/\s]+)/([0-9a-f]{6,}[^/\s]*)", cmd)
                entry.update({"port": m.group(1), "tag": m.group(2)} if m else {})
                state["phone_runs"].append(entry)
            elif re.search(r"\b(cmake|ninja|clang)", cmd):
                m = re.search(r"/wt/([0-9a-f]{6,}[^/\s]*)", cmd)
                entry["tag"] = m.group(1) if m else None
                state["builds"].append(entry)
        builds = {}
        for b in state["builds"]:
            if b["tag"] or not builds:
                builds.setdefault(b["tag"], b)
        state["builds"] = list(builds.values())
        state["listening"] = sorted({int(m) for m in re.findall(r":(878\d) \(LISTEN\)", listen)})
        with self.lock:
            self.state = state

    def get(self):
        with self.lock:
            return dict(self.state)


class Health:
    """GET /health on each gateway through its tunnel, every few seconds, never on the request path."""

    def __init__(self):
        self.state, self.lock = {}, threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            for g in GATEWAYS:
                threading.Thread(target=self.check, args=(g,), daemon=True).start()
            time.sleep(15)  # /health runs adb on the phone; keep it rare so it can't disturb a timed run

    def check(self, g):
        t0 = time.time()
        out = {**g, "tunnel": True, "checked": t0, "ok": False}
        if out["tunnel"]:
            try:
                with urllib.request.urlopen(g["url"] + "/health", timeout=10) as r:
                    body = json.loads(r.read())
                body = body.get("result", body)
                raw = (body.get("device") or {}).get("raw") or []
                out.update({"ok": True, "ms": round((time.time() - t0) * 1000), "busy": body.get("busy"),
                            "served": body.get("served"), "device_error": (body.get("device") or {}).get("error"),
                            "phone": parse_phone(raw)})
            except Exception as exc:  # timeouts and refused connections are both "not answering"
                out["error"] = type(exc).__name__ + (": " + str(exc)[:120] if str(exc) else "")
        with self.lock:
            self.state[g["label"]] = out

    def get(self):
        with self.lock:
            return [self.state.get(g["label"], {**g, "checked": None}) for g in GATEWAYS]


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def parse_phone(raw):
    """devrun.device_state(): [thermal milli-C, battery %, GPU model, product model words...]."""
    if not raw:
        return None
    out = {"raw": raw}
    try:
        out["temp_c"] = round(int(raw[0]) / 1000, 1)
        out["battery"] = int(raw[1])
    except (ValueError, IndexError):
        pass
    if len(raw) > 2:
        out["gpu"] = raw[2]
        out["model"] = " ".join(raw[3:])
    return out


# --- one task's snapshot -------------------------------------------------------------------------------------

def tasks():
    names = {p.stem for p in (RUNS / "ledger").glob("*.jsonl")} | {p.name for p in (RUNS / "tasks").glob("*") if p.is_dir()}

    def last(t):
        return max(mtime(RUNS / "ledger" / f"{t}.jsonl"), mtime(RUNS / "tasks" / t), mtime(RUNS / "agents" / t),
                   *[mtime(p) for p in (RUNS / "agents" / t).glob("*")])
    return sorted(names, key=last, reverse=True)


AGENT_LINE = re.compile(r"^(?:(?P<task>[\w.\-]+)/)?(?P<role>[a-z]+)-(?P<i>\d+) (?P<rest>.*)$")


def agent_log(task):
    """Live agent progress from logs/agents.log, by agent: start, each tool call, finish."""
    path = LOGS / "agents.log"
    ref = mtime(path)
    runs, by_tag = [], {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return runs, by_tag
    for raw in lines:
        m = LogTail.LINE.match(raw)
        if not m:
            continue
        at, text = clock(m.group(1), ref), m.group(3)
        r = re.match(r"^([\w.\-]+): (\d+) (\w+) agent\(s\) on ([\w.\-]+), budget (\d+)s, (\d+) validation calls each, "
                     r"gateway (\S+)", text)
        if r:
            if r.group(1) == task:
                runs.append({"at": at, "n": int(r.group(2)), "role": r.group(3), "model": r.group(4),
                             "budget": int(r.group(5)), "phone": int(r.group(6)), "gateway": r.group(7).rstrip(",")})
            continue
        a = AGENT_LINE.match(text)
        if not a or (a.group("task") and a.group("task") != task):
            continue
        tag = "%s-%s" % (a.group("role"), a.group("i"))
        rest = a.group("rest")
        if rest.startswith("started"):
            sm = re.match(r"started \(([\w.\-]+)(?:, budget (\d+)s, (\d+) validation calls)?\)", rest)
            by_tag[tag] = {"started": at, "model": sm.group(1) if sm else None,
                           "budget": int(sm.group(2)) if sm and sm.group(2) else None,
                           "phone": int(sm.group(3)) if sm and sm.group(3) else None, "events": [], "finished": None}
            continue
        cur = by_tag.get(tag)
        if not cur:
            continue
        fm = re.match(r"finished (\w+) in (\d+)s", rest)
        if fm:
            cur["finished"] = {"at": at, "status": fm.group(1), "seconds": int(fm.group(2))}
            continue
        em = re.match(r"(\w+) #(\d+) \((\d+) diff lines\) -> (.*)$", rest)
        if em:
            try:
                result = json.loads(em.group(4))
            except ValueError:
                result = {"text": em.group(4)[:300]}
            cur["events"].append({"t": round(at - cur["started"]) if at and cur["started"] else None,
                                  "tool": em.group(1), "patch_lines": int(em.group(3)), "result": result})
    return runs, by_tag


def event_text(e):
    r = e.get("result") or {}
    tool = e.get("tool")
    if tool == "time_left":
        return "checked the clock: %ss left" % r.get("seconds_left"), "info"
    if r.get("error") or r.get("infra_error"):
        kind = r.get("error_kind") or ("infra" if r.get("infra_error") else "submission")
        return "%s error: %s" % (kind, redact(r.get("error") or r.get("infra_error"))[:220]), "err"
    bits = []
    if r.get("decode_tok_s_external") or r.get("decode_tok_s"):
        bits.append("%s tok/s" % (r.get("decode_tok_s_external") or r.get("decode_tok_s")))
    if r.get("speedup"):
        bits.append("%s× speedup" % r.get("speedup"))
    for k in ("correct", "valid"):
        if r.get(k) is not None:
            bits.append("tokens match" if r[k] else "tokens differ")
            break
    if r.get("reward") is not None:
        bits.append("reward %s" % r.get("reward"))
    if r.get("flags"):
        bits.append("flags: " + ", ".join(map(str, r["flags"])))
    verb = {"submit": "submitted", "run_on_phone": "ran on the phone", "run_validation": "ran on the phone"}.get(tool, tool)
    return "%s: %s" % (verb, "; ".join(bits) or "no result"), ("hit" if (r.get("reward") or 0) > 0 else "info")


def report_title(text):
    m = re.search(r"^#+\s*Red-Team Report:?\s*(.+)$", text or "", re.M)
    return m.group(1).strip(" `") if m else None


def agents(task, procs):
    base = RUNS / "agents" / task
    live_runs, live = agent_log(task)
    running_roles = {arg(p["argv"], "--role") for p in procs if p["kind"] == "agents" and task in p["argv"]}
    # Red-team agent i ran attack i of cut.json's list (agents.run_role); older summaries don't record it.
    planned = [a["name"] for a in (read_json(RUNS / "tasks" / task / "cut.json") or {}).get("red_team_attacks", [])]
    out = []
    for trial in sorted(p for p in base.glob("**/*") if p.is_dir() and re.search(r"-\d+$", p.name)) if base.exists() else []:
        m = re.match(r"^([a-z]+)-(\d+)$", trial.name)
        if not m:
            continue
        role, i = m.group(1), int(m.group(2))
        run = trial.parent.name if trial.parent != base else None
        rm = re.match(r"^(\d{8}-\d{6})-", run or "")
        started = dt.datetime.strptime(rm.group(1), "%Y%m%d-%H%M%S").timestamp() if rm else None
        s = read_json(trial / "summary.json")
        if s is None and run:  # finished runs also list every trial in results.json
            s = next((r for r in read_json(trial.parent / "results.json") or [] if r.get("i") == i), None)
        budget, phone, submit = ROLE_LIMITS.get(role, (None, None, None))
        a = {"id": "%s/%s" % (run, trial.name) if run else trial.name, "run": run or "first red-team run", "role": role,
             "i": i, "stage": ROLE_STAGE.get(role), "dir": str(trial.relative_to(ROOT)), "budget": budget,
             "limits": {"phone": phone, "submit": submit}, "started": started,
             "files": sorted(str(f.relative_to(ROOT)) for f in trial.glob("*.diff"))}
        if s:
            best = s.get("best") or {}
            final_text = s.get("final_text") or ""
            a.update({"model": s.get("model"), "status": s.get("status"), "seconds": s.get("seconds"),
                      "budget": s.get("budget_sec") or budget, "calls": s.get("calls"),
                      "reward": s.get("reward", s.get("best_reward")),
                      "best_reward": s.get("best_reward", (best or {}).get("reward")), "speedup": best.get("speedup"),
                      "decode": (best.get("decode_tok_s") or {}).get("candidate") if isinstance(best.get("decode_tok_s"), dict)
                      else best.get("decode_tok_s"), "flags": best.get("flags"), "dry_run": s.get("dry_run"),
                      "valid": best.get("valid", best.get("correct")), "scoring": best.get("scoring"),
                      "gateway": s.get("gateway"),
                      "attack": s.get("attack") or (planned[i - 1] if role == "redteam" and i <= len(planned) else None)
                      or report_title(final_text),
                      "report": redact(final_text[:4000]), "trajectory": trim(s.get("trajectory")),
                      "environment": s.get("environment_id")})
            events = s.get("events") or []
            if started is None and a["seconds"]:
                started = mtime(trial / "summary.json") - a["seconds"]
                a["started"] = started
        else:
            lv = live.get(trial.name) or {}
            if lv.get("started") and started and lv["started"] < started - 60:
                lv = {}  # that log entry belongs to an earlier run of the same role
            fin = lv.get("finished")
            alive = role in running_roles
            a.update({"model": lv.get("model"), "budget": lv.get("budget") or budget,
                      "status": fin["status"] if fin else ("running" if alive else "stopped"),
                      "seconds": fin["seconds"] if fin else None, "live": True})
            if lv.get("phone") is not None:
                a["limits"]["phone"] = lv["phone"]
            if lv.get("started"):
                a["started"] = lv["started"]
            events = lv.get("events") or []
            calls = {}
            for e in events:
                calls[e["tool"]] = calls.get(e["tool"], 0) + 1
            a["calls"] = calls
            subs = [e["result"] for e in events if e["tool"] == "submit"]
            if subs:
                b = max(subs, key=lambda r: r.get("reward") or 0)
                a.update({"best_reward": b.get("reward"), "speedup": b.get("speedup"), "flags": b.get("flags"),
                          "valid": b.get("valid", b.get("correct")), "scoring": b.get("scoring")})
        a["regrade"] = regrade_for(trial)
        a["events"] = []
        for e in events[-40:]:
            text, kind = event_text(e)
            a["events"].append({"t": e.get("t"), "tool": e.get("tool"), "text": text, "kind": kind,
                                "patch_lines": e.get("patch_lines")})
        out.append(a)
    out.sort(key=lambda a: (a["status"] != "running", -(a.get("started") or 0), a["i"]))
    return out, live_runs


def regrade_for(trial):
    """The latest regrade of one agent's saved submission: runs/agents/<task>/<run>/regrade.json, plus
    logs/regrade.log to tell whether a regrade of it is running now."""
    path = trial.parent / "regrade.json"
    rg = (read_json(path) or {}).get(trial.name) if path.exists() else None
    task = trial.parent.parent.name
    pending = False
    try:
        lines = (LOGS / "regrade.log").read_text(errors="replace").splitlines()[-400:]
        # current format names the task: "grade <task> (rollout-N submit-K) started: ..." then "... (rollout-N submit-K): ..."
        tag = re.compile(r"grade %s \(%s submit-\d+\)( started)?:" % (re.escape(task), re.escape(trial.name)))
        hits = [(i, bool(m.group(1))) for i, l in enumerate(lines) for m in [tag.search(l)] if m]
        if hits:
            pending = hits[-1][1]
        elif path.exists():  # older lines carry no task, so they only count for the run that has a regrade.json
            grading = max((i for i, l in enumerate(lines) if (" %s: grading " % trial.name) in l), default=-1)
            wrote = max((i for i, l in enumerate(lines) if ("wrote " + str(path.relative_to(ROOT))) in l), default=-1)
            pending = grading > wrote
        pending = pending and time.time() - mtime(LOGS / "regrade.log") < 3600
    except (OSError, ValueError):
        pass
    if not rg and not pending:
        return None
    rg = rg or {}
    return {"pending": pending, "reward": rg.get("reward"), "valid": rg.get("valid"), "speedup": rg.get("speedup"),
            "scoring": rg.get("scoring"),
            "flags": rg.get("flags"), "infra_error": redact(rg.get("infra_error") or "")[:240] or None,
            "error": redact(rg.get("error") or "")[:240] or None, "at": mtime(path) if path.exists() else None}


def toml_numbers(path):
    """Numbers and number lists from a simple TOML file, by section (the system python3 has no tomllib)."""
    out, sec = {}, None
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^\[([\w.]+)\]$", line)
        if m:
            sec = out.setdefault(m.group(1), {})
            continue
        m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if m and sec is not None:
            v = m.group(2).strip()
            try:
                sec[m.group(1)] = json.loads(v)
            except ValueError:
                sec[m.group(1)] = v.strip('"')
    return out


def fleet_scoring(task, ags):
    """The scoring rule as the cloud gateway applies it: the latest `scoring` block any /try or /grade for this
    task returned (regrades, agent submits, logs/cloud-gateway.log), the acceptance window for rollouts, and one
    row per rollout with the result that counts (a finished regrade replaces the agent's own submit)."""
    blocks = []
    for a in ags:
        for src in (a.get("regrade") or {}, a):
            sc = src.get("scoring")
            if isinstance(sc, dict) and sc.get("target_speedup"):
                blocks.append((src.get("at") or a.get("started") or 0, sc))
    try:
        log = re.sub(r"\x1b\[[0-9;]*m", "", (LOGS / "cloud-gateway.log").read_text(errors="replace")).splitlines()[-600:]
        ref = mtime(LOGS / "cloud-gateway.log")
        for l in log:
            m = re.match(r"(\d\d:\d\d:\d\d) INFO (?:try|grade) %s \([^)]*\): (\{.*\})" % re.escape(task), l)
            if m:
                try:
                    d = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    continue
                sc = d.get("scoring") if isinstance(d, dict) else None
                if isinstance(sc, dict) and sc.get("target_speedup"):
                    blocks.append((clock(m.group(1), ref) or 0, sc))
    except OSError:
        pass
    if not blocks:
        return None
    blocks.sort(key=lambda b: b[0])
    sc = blocks[-1][1]
    acc = toml_numbers(ROOT / "private" / "acceptance.toml").get("rollouts", {})
    rows = []
    for a in ags:
        if a["role"] not in ("rollout", "solver"):
            continue
        g = a.get("regrade") or {}
        if g and not g.get("pending") and not g.get("infra_error") and not g.get("error") and g.get("reward") is not None:
            r, sp, valid, via = g["reward"], g.get("speedup"), g.get("valid"), "regrade"
        elif a["status"] == "running":
            r, sp, valid, via = None, None, None, "running"
        elif a["status"] == "infra_error" or (g and g.get("infra_error")):
            r, sp, valid, via = None, None, None, "infra error"
        else:
            r = a.get("best_reward") if a.get("best_reward") is not None else a.get("reward")
            sp, valid, via = a.get("speedup"), a.get("valid"), "submit" if a.get("best_reward") is not None else "no submission"
        rows.append({"id": a["id"], "run": a.get("run"), "i": a["i"], "model": a.get("model"), "status": a["status"],
                     "reward": r, "speedup": sp, "valid": valid, "via": via, "started": a.get("started")})
    rows.sort(key=lambda r: (r["run"] or "", r["i"]))
    return {"rule": {k: sc.get(k) for k in ("metric", "measured", "aggregate", "mapping", "baseline", "target_source")},
            "target": sc.get("target_speedup"), "base_tok_s": sc.get("baseline_decode_tok_s"), "at": blocks[-1][0],
            "window": {"mean_max": acc.get("target_mean_max"), "variance_min": acc.get("target_variance_min"),
                       "trials": acc.get("target_model_trials")}, "rows": rows}


def candidates_summary(port):
    c = read_json(RUNS / "mine" / port / "candidates.json")
    if not c:
        return None
    pack = read_json(RUNS / "mine" / port / "pack.json") or {}
    usage = {}
    try:
        text = (LOGS / "mine.log").read_text()
        m = re.findall(r"done in (\d+)s: (\d+) candidates, (\d+) rejected, of (\d+) commits; tokens in (\d+) out (\d+)", text)
        if m:
            s, nc, nr, n, tin, tout = map(int, m[-1])
            usage = {"seconds": s, "commits": n, "tokens_in": tin, "tokens_out": tout}
    except OSError:
        pass
    return {"port": port, "commits_reviewed": c.get("commits_reviewed"), "usage": usage,
            "pack": {k: pack.get(k) for k in ("files", "commits", "est_tokens")},
            "candidates": [{"commit": x["commit"][:7], "title": x.get("title"), "claimed": x.get("claimed_speedup"),
                            "metric": x.get("metric"), "difficulty": x.get("difficulty"), "quality": x.get("task_quality"),
                            "device": redact(str(x.get("device_dependence") or ""))[:300],
                            "cheats": len(x.get("cheat_risks") or [])} for x in c.get("candidates", [])],
            "rejected": [{"commit": x["commit"][:7], "title": redact(x.get("title") or ""), "reason": redact(x.get("reason") or "")}
                         for x in c.get("rejected", [])]}


def cloud_gateway(ags_all):
    """The Cloud Run gateway, from logs/cloud-gateway.log: current image and revision, whether a deploy is running,
    and the last calls through it."""
    path = LOGS / "cloud-gateway.log"
    try:
        lines = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")).splitlines()
    except OSError:
        return None
    out = {"url": None, "revision": None, "image": None, "deploying": None, "last_deploy": None, "recent": [], "infra_errors": 0}
    for l in lines:
        m = re.search(r"Service URL: (https://\S+)", l)
        if m:
            out["url"] = m.group(1)
        m = re.search(r"revision \[([\w-]+)\] has been deployed", l)
        if m:
            out["revision"] = m.group(1)
        m = re.match(r"(\d\d:\d\d:\d\d) build\+deploy (\d+) (start|exit (\d+))(.*)", l)
        if m:
            if m.group(3) == "start":
                out["deploying"] = {"image": int(m.group(2)), "since": clock(m.group(1), mtime(path)), "what": m.group(5).strip(" ()")[:200]}
            else:
                out["last_deploy"] = {"image": int(m.group(2)), "exit": int(m.group(4)), "at": clock(m.group(1), mtime(path))}
                if out["deploying"] and out["deploying"]["image"] == int(m.group(2)):
                    out["deploying"] = None
                if int(m.group(4)) == 0:
                    out["image"] = int(m.group(2))
        m = re.match(r"(\d\d:\d\d:\d\d) INFO (try|grade|calibrate|rollout-\d+)\b([^:]*): (.*)", l)
        if m and not m.group(3).endswith(" started") and not m.group(4).startswith("grading "):
            # "rollout-N: {...}" lines with no task are regrades of saved rollout patches, not rollouts
            call = ("regrade " + m.group(2)) if m.group(2).startswith("rollout-") else (m.group(2) + m.group(3)).strip()
            m = (m.group(1), call[:60], m.group(4))
            bad = re.search(r"['\"](infra_error|http_error)['\"]: (?!None|null)", m[2]) is not None
            out["recent"].append({"at": clock(m[0], mtime(path)), "call": m[1], "ok": not bad, "text": redact(m[2])[:220]})
    out["recent"] = out["recent"][-12:]
    out["infra_errors"] = sum(1 for r in out["recent"] if not r["ok"])
    out["agents"] = sum(1 for a in ags_all if a.get("status") == "running" and "run.app" in str(a.get("gateway") or ""))
    return out


def fleet():
    md = sorted((RUNS / "device").glob("google-gateB-*.md"), key=mtime)
    if not md:
        return None
    text = md[-1].read_text()
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in text.splitlines() if line.startswith("|")]
    rows = [r for r in rows if r and not set("".join(r)) <= set("-: ")]
    head = text.splitlines()[0].lstrip("# ").strip()
    session = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d-\d\d:\d\d)", text)
    findings = re.search(r"Findings: (.*?)(?:\nCost|$)", text, re.S)
    cost = re.search(r"Cost: (.*)", text)
    return {"title": head, "file": str(md[-1].relative_to(ROOT)), "when": session.group(1) if session else None,
            "at": mtime(md[-1]), "header": rows[0] if rows else [], "rows": rows[1:],
            "findings": " ".join(findings.group(1).split()) if findings else None,
            "cost": cost.group(1).strip() if cost else None}


def from_files(task, work, stage, ags, port):
    """What the run folders say about a stage that has no ledger verdict yet (scripts run outside factory.py)."""
    if stage == 1:
        c = candidates_summary(port)
        if c:
            return {"verdict": "pass", "passed": True, "source": "runs/mine/%s/candidates.json" % port,
                    "seconds": c["usage"].get("seconds"), "model": "gemini-4-argon",
                    "findings": ["%d candidates, %d rejected" % (len(c["candidates"]), len(c["rejected"]))]}
    if stage == 5 and (ROOT / "tasks" / task / "instruction.md").exists():
        return {"verdict": "pass", "passed": True, "source": "tasks/%s/" % task, "model": "gemini-4-argon",
                "findings": ["task package written"], "at": mtime(ROOT / "tasks" / task / "instruction.md")}
    if stage == 7:
        r = (read_json(work / "gateA-noop.json") or {}).get("result")
        if r:
            ok = (r.get("reward") or 0) == 0
            return {"verdict": "pass" if ok else "fail", "passed": ok, "source": "runs/tasks/%s/gateA-noop.json" % task,
                    "seconds": r.get("gateway_seconds"), "at": mtime(work / "gateA-noop.json"),
                    "findings": ["no-op scored %s (%s× speed, %s)" % (r.get("reward"), r.get("speedup"),
                                                                     "; ".join(r.get("flags") or []) or "no flags")]}
    if stage == 9:
        r = (read_json(work / "implementation_qa.json") or {}).get("result")
        if r:
            crit = [f for f in r.get("findings", []) if f.get("severity") in ("critical", "high")]
            return {"verdict": "fail" if r.get("solvable") is False or crit else "pass",
                    "passed": not (r.get("solvable") is False or crit), "model": "gemini-4-argon",
                    "source": "runs/tasks/%s/implementation_qa.json" % task, "at": mtime(work / "implementation_qa.json"),
                    "findings": [redact(r.get("summary", ""))[:600]] + ["%s: %s" % (f.get("severity"), redact(f.get("finding") or f.get("issue") or f.get("description") or json.dumps(f))[:300]) for f in crit[:6]]}
    if stage == 12:
        for name in ("calibrate-v2c.json", "calibrate-v2b.json", "calibrate-v2.json", "calibrate.json"):
            d = read_json(work / name)
            if d:
                r = d.get("result", d)
                err = r.get("error")
                return {"verdict": "fail" if err else "pass", "passed": not err, "source": "runs/tasks/%s/%s" % (task, name),
                        "seconds": r.get("gateway_seconds"), "at": mtime(work / name),
                        "findings": [redact(err)] if err else ["target speedup %s (base %s, reference %s tok/s)" % (
                            r.get("target_speedup"), r.get("base_decode_tok_s", r.get("base_decode")),
                            r.get("reference_decode_tok_s", r.get("reference_decode")))]}
    if stage in (11, 13, 14):
        mine = [a for a in ags if a.get("stage") == stage]
        if mine:
            latest = max(mine, key=lambda a: a.get("started") or 0)
            batch = [a for a in mine if a["run"] == latest["run"]]
            if stage == 13:
                batch = mine  # every red-team agent so far counts
                broke = [a for a in batch if (a.get("best_reward") or 0) > 0]
                fixes = read_jsonl(work / "fixes.jsonl")
                verdict = "fail" if broke and not fixes else "pass"
                f = ["%d of %d attacks scored on the grader" % (len(broke), len(batch))]
                if fixes:
                    f.append("the fixer closed %d hole(s); the grader is now v2" % sum(len(x.get("holes", [])) for x in fixes))
                return {"verdict": verdict, "passed": verdict == "pass", "source": "runs/agents/%s/" % task,
                        "model": "gemini-4-argon", "findings": f, "at": max((a.get("started") or 0) for a in batch)}
            statuses = [a["status"] for a in batch]
            ok = all(s == "completed" for s in statuses)
            verdict = "running" if "running" in statuses else ("pass" if ok else "fail")
            return {"verdict": verdict, "passed": ok, "source": "runs/agents/%s/%s" % (task, latest["run"]),
                    "model": latest.get("model"), "at": latest.get("started"),
                    "findings": ["%s-%d %s, reward %s" % (a["role"], a["i"], a["status"], a.get("best_reward")) for a in batch]}
    if stage == 15:
        r = (read_json(work / "gateA-oracle.json") or {}).get("result")
        if r:
            ok = (r.get("reward") or 0) >= 1
            return {"verdict": "pass" if ok else "fail", "passed": ok, "source": "runs/tasks/%s/gateA-oracle.json" % task,
                    "seconds": r.get("gateway_seconds"), "at": mtime(work / "gateA-oracle.json"),
                    "findings": ["reference scored %s at %s× (%s tok/s vs base %s)" % (
                        r.get("reward"), r.get("speedup"), (r.get("decode_tok_s") or {}).get("candidate"),
                        (r.get("decode_tok_s") or {}).get("base"))]}
    return None


def factory_progress(task, procs):
    """Which stage factory.py is on right now: its process arguments plus the ledger rows written since it started."""
    p = next((p for p in procs if p["kind"] == "factory" and arg(p["argv"], "--task") == task), None)
    if not p:
        return None
    lo, hi = int(arg(p["argv"], "--from") or 2), int(arg(p["argv"], "--to") or 16)
    t0 = time.time() - p["seconds"]
    stage, fixing, since_ts = lo, False, t0
    for e in read_jsonl(RUNS / "ledger" / f"{task}.jsonl"):
        try:
            at = dt.datetime.fromisoformat(e["ts"]).timestamp()
        except (KeyError, ValueError):
            continue
        if at < t0 - 2:
            continue
        n, v = e.get("stage"), e.get("verdict")
        stage, since_ts = (n + 1 if e.get("passed") or v == "skipped" else n), at
        fixing = not (e.get("passed") or v == "skipped")  # failed or returned: its author is fixing it
    try:  # a fix that gave up leaves a "stopping" line; the process exits right after
        tail = (LOGS / "factory.log").read_text().splitlines()[-3:]
    except OSError:
        tail = []
    return {"pid": p["pid"], "stage": min(stage, hi), "fixing": fixing, "from": lo, "to": hi, "seconds": p["seconds"],
            "stage_since": since_ts, "profile": arg(p["argv"], "--profile"),
            "stopping": any("stopping" in l for l in tail)}


def gemini_usage(task, ledger_rows, ags, cand):
    use = {}

    def add(model, calls=1, tin=0, tout=0, active=0):
        u = use.setdefault(model, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "active": 0})
        u["calls"] += calls
        u["tokens_in"] += tin or 0
        u["tokens_out"] += tout or 0
        u["active"] += active
    if cand and cand.get("usage"):
        add("gemini-4-argon", 1, cand["usage"].get("tokens_in"), cand["usage"].get("tokens_out"))
    for e in ledger_rows:
        ag = e.get("agent") or {}
        if ag.get("model"):
            add(ag["model"], 1, ag.get("input_tokens"), ag.get("output_tokens"))
    for a in ags:
        if a.get("model"):
            add(a["model"], 1, active=1 if a["status"] == "running" else 0)
    return use


def everywhere(task, ags, prog, procs):
    """What is running for every task, so the machine map shows all traffic, not just the selected task's."""
    running = [dict(a, task=task) for a in ags if a["status"] == "running"]
    factories = [dict(prog, task=task)] if prog else []
    for t in tasks():
        if t == task:
            continue
        if any(p["kind"] == "agents" and t in p["argv"] for p in procs):
            running += [{"task": t, **{k: a.get(k) for k in ("id", "role", "i", "model", "started", "budget", "stage")},
                         "status": "running", "last": a["events"][-1]["text"] if a["events"] else None}
                        for a in agents(t, procs)[0] if a["status"] == "running"]
        f = factory_progress(t, procs)
        if f:
            factories.append(dict(f, task=t, label=STAGES.get(f["stage"], ("", ""))[1]))
    return {"agents": running, "factories": factories}


def build_stages(task, procs):
    """Every stage's state for one task: ledger verdicts first, run files second, live processes on top."""
    if not task:
        return {"task": None, "stages": [dict(n=n, key=k, label=l, what=w, where=wh, history=[], status="pending")
                                          for n, (k, l, w, wh) in STAGES.items()],
                "ags": [], "live_runs": [], "prog": None, "cand": None, "spec": {}, "port": None, "ledger_rows": [],
                "work": None}
    work = RUNS / "tasks" / task
    spec = read_json(work / "spec.json") or {}
    port = spec.get("port") or "smollm2-135m-instruct"
    ledger_rows = read_jsonl(RUNS / "ledger" / f"{task}.jsonl")
    ags, live_runs = agents(task, procs)
    prog = factory_progress(task, procs)
    cand = candidates_summary(port)
    stages = []
    for n, (key, label, what, where) in STAGES.items():
        rows = [e for e in ledger_rows if e.get("stage") == n]
        s = {"n": n, "key": key, "label": label, "what": what, "where": where, "history": []}
        for e in rows:
            s["history"].append({"verdict": e.get("verdict"), "passed": e.get("passed"), "round": e.get("round"),
                                 "ts": e.get("ts"), "seconds": e.get("seconds"),
                                 "model": (e.get("agent") or {}).get("model"),
                                 "tokens": [(e.get("agent") or {}).get("input_tokens"), (e.get("agent") or {}).get("output_tokens")],
                                 "findings": [redact(f)[:1200] for f in (e.get("findings") or [])][:12],
                                 "details": trim(e.get("details"))})
        if rows:
            last = s["history"][-1]
            if (last.get("details") or {}).get("approved_by") == "user":
                s["approved_by"] = "user"
            s.update({"status": last["verdict"], "source": "ledger", "round": last["round"], "seconds": last["seconds"],
                      "model": last["model"], "findings": last["findings"],
                      "at": dt.datetime.fromisoformat(last["ts"]).timestamp() if last.get("ts") else None})
        else:
            f = from_files(task, work, n, ags, port)
            if f:
                s.update({"status": f["verdict"], "source": f["source"], "seconds": f.get("seconds"),
                          "model": f.get("model"), "findings": f.get("findings"), "at": f.get("at"), "from_files": True})
            else:
                s["status"] = "pending"
        stages.append(s)
    running = {}
    if prog:
        running[prog["stage"]] = "fixing" if prog["fixing"] else "running"
    for p in procs:
        if p["kind"] == "agents" and task in p["argv"]:
            st = ROLE_STAGE.get(arg(p["argv"], "--role"))
            if st:
                running[st] = "running"
        elif p["kind"] == "cut" and task in p["argv"]:
            running[5] = "running"
        elif p["kind"] in ("mine", "pack") and port in p["argv"]:
            running[1] = "running"
    for s in stages:
        if s["n"] in running:
            s["running"] = running[s["n"]]
            if prog and prog["stage"] == s["n"]:
                s["running_since"] = prog["stage_since"]
            else:
                secs = next((p["seconds"] for p in procs if p["kind"] in ("agents", "cut", "mine") and task in p["argv"]), None)
                s["running_since"] = time.time() - secs if secs is not None else None
    return {"task": task, "stages": stages, "ags": ags, "live_runs": live_runs, "prog": prog, "cand": cand, "spec": spec,
            "port": port, "ledger_rows": ledger_rows, "work": work}


def task_title(task, stages):
    st2 = next((s for s in stages if s["n"] == 2 and s["history"]), None)
    if st2 and (st2["history"][-1].get("details") or {}).get("title"):
        return st2["history"][-1]["details"]["title"]
    prop = read_json(RUNS / "tasks" / task / "proposal.json") or {}
    return (prop.get("proposal") or {}).get("title")


def task_repo(spec):
    """Which repo a task came from: its port is a model folder in the factory's own repo."""
    port = spec.get("port")
    if port and (repos.BUILTIN["path"] / repos.BUILTIN["units"] / port).is_dir():
        return repos.BUILTIN["slug"]
    return spec.get("repo")


AGENT_FIELDS = ("id", "role", "i", "model", "status", "started", "seconds", "budget", "calls", "limits", "best_reward",
                "speedup", "attack", "stage", "dry_run", "run", "regrade", "reward")


def catalog(procs, sink=None):
    """Every task, newest first, with how far it got. Accepted = final sign-off (stage 16) passed. Every managed
    agent of every task goes into `sink`."""
    out = []
    for t in tasks():
        b = build_stages(t, procs)
        if sink is not None:
            for a in b["ags"]:
                sink.append(dict({k: a.get(k) for k in AGENT_FIELDS}, task=t,
                                 last=a["events"][-1]["text"] if a.get("events") else None))
        st = b["stages"]
        by = {s["n"]: s for s in st}
        blocker = next((s for s in st if s["status"] in ("fail", "reject", "return") and not s.get("running")), None)
        running = [s["n"] for s in st if s.get("running")]
        team = [a for a in b["ags"] if a["role"] == "redteam"]
        work = b["work"]
        oracle = (read_json(work / "gateA-oracle.json") or {}).get("result") or {}
        out.append({
            "task": t, "port": b["port"], "repo": task_repo(b["spec"]), "title": task_title(t, st),
            "difficulty": ((by[2]["history"][-1].get("details") or {}).get("difficulty") if by[2]["history"] else None),
            "reference": b["spec"].get("reference"), "base": (b["spec"].get("base") or "")[:12],
            "target_speedup": b["spec"].get("reference_speedup"), "phone": b["spec"].get("phone"),
            "stages": [{"n": s["n"], "status": s["status"], "running": s.get("running")} for s in st],
            "passed": sum(1 for s in st if s["status"] in PASSED), "accepted": by[16]["status"] in PASSED,
            "blocker": {"n": blocker["n"], "label": blocker["label"], "status": blocker["status"],
                        "finding": (blocker.get("findings") or [""])[0][:300]} if blocker else None,
            "running": running, "last_at": max([s.get("at") or 0 for s in st] + [mtime(work)]),
            "agents": len(b["ags"]), "agents_running": sum(1 for a in b["ags"] if a["status"] == "running"),
            "redteam": {"n": len(team), "broke": sum(1 for a in team if (a.get("best_reward") or 0) > 0)},
            "reference_reward": oracle.get("reward"), "reference_measured": oracle.get("speedup"),
            "package": (ROOT / "tasks" / t / "instruction.md").exists(),
            "scoring": fleet_scoring(t, b["ags"]),
        })
    return out


def repo_view(slug, cat):
    """One repo's digest plus, for each chunk, what the factory has done with it (mined, tasks)."""
    rec = repos.load(slug)
    if not rec:
        return None
    rec = json.loads(json.dumps(rec))
    for c in rec.get("chunks") or []:
        names = [p.rsplit("/", 1)[-1] for p in c["paths"]]
        c["mined"] = []
        for n in names:
            cs = read_json(RUNS / "mine" / n / "candidates.json")
            pk = read_json(RUNS / "mine" / n / "pack.json")
            if cs or pk:
                c["mined"].append({"unit": n, "candidates": len((cs or {}).get("candidates", [])) if cs else None,
                                   "rejected": len((cs or {}).get("rejected", [])) if cs else None,
                                   "packed_tokens": (pk or {}).get("est_tokens")})
        c["tasks"] = [t["task"] for t in cat if t["repo"] == slug and t["port"] in names]
    rec["tasks"] = [t["task"] for t in cat if t["repo"] == slug]
    return rec


def snapshot(task, repo_slug, procs, logs, remote, health, cat):
    b = build_stages(task, procs)
    stages, ags, live_runs, prog, cand = b["stages"], b["ags"], b["live_runs"], b["prog"], b["cand"]
    spec, port, ledger_rows, work = b["spec"], b["port"], b["ledger_rows"], b["work"] or RUNS / "tasks" / "-none-"
    work_files = {
        "spec": {k: spec.get(k) for k in ("port", "base", "reference", "base_tps", "reference_tps", "reference_speedup",
                                          "needed_speedup", "phone")},
        "noop": (read_json(work / "gateA-noop.json") or {}).get("result"),
        "oracle": (read_json(work / "gateA-oracle.json") or {}).get("result"),
        "calibrate": [dict((read_json(work / n) or {}).get("result") or read_json(work / n) or {}, file=n)
                      for n in ("calibrate.json", "calibrate-v2.json", "calibrate-v2b.json", "calibrate-v2c.json")
                      if (work / n).exists()],
        "fixes": trim(read_jsonl(work / "fixes.jsonl")),
        "cheats": trim(read_json(work / "cheats" / "index.json") or []),
        "qa_mutation": trim({k: v for k, v in (read_json(work / "qa-mutation.json") or {}).items() if k != "control"}),
        "implementation_qa": trim((read_json(work / "implementation_qa.json") or {}).get("result")),
        "cut": trim({k: v for k, v in (read_json(work / "cut.json") or {}).items()
                     if k in ("red_team_attacks", "difficulty", "agent_timeout_sec", "tags", "in_scope_files", "off_limits_files")}),
        "package": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tasks" / task).rglob("*") if p.is_file())[:60]
        if task and (ROOT / "tasks" / task).exists() else [],
    }
    for c in work_files["calibrate"]:
        c.pop("dev_prompts", None)  # prompt text stays off the page
    work_files["scoring"] = fleet_scoring(task, ags) if task else None
    return {
        "task": task, "tasks": [t["task"] for t in cat if t["repo"] == repo_slug] if repo_slug else tasks(),
        "repo_slug": repo_slug, "repos": [{k: r.get(k) for k in ("slug", "name", "url", "status", "progress", "error",
                                                                 "builtin", "added", "digested")}
                                          | {"stats": {k: (r.get("stats") or {}).get(k) for k in ("files", "lines", "tokens", "commits")},
                                             "chunks": len(r.get("chunks") or [])} for r in repos.all_repos()],
        "repo": repo_view(repo_slug, cat) if repo_slug else None, "catalog": cat,
        "port": port, "now": time.time(),
        "stages": stages, "agents": ags, "agent_runs": live_runs, "factory": prog,
        "machines": {
            "laptop": {"procs": [{k: v for k, v in p.items() if k != "argv"} for p in procs],
                       "host": socket.gethostname().split(".")[0]},
            "gateways": health.get(), "macmini": remote.get(),
            "gemini": gemini_usage(task, ledger_rows, ags, cand), "fleet": fleet(), "cloud": cloud_gateway(ags),
        },
        "ui_version": mtime(UI / "index.html"), "mine": cand, "work": work_files, "log": logs.recent(),
        "everywhere": everywhere(task, ags, prog, procs),
    }


# --- one task, in full: proposal rounds, Harbor package, who did what -------------------------------------------

# The agent (or script) behind each stage, as the page names it.
STAGE_ROLE = {1: "miner", 2: "author", 3: "checker", 4: "reviewer", 5: "author", 6: "script", 7: "phone check",
              8: "phone check", 9: "auditor", 10: "reviewer", 11: "explorer", 12: "calibrator", 13: "attacker",
              14: "solver", 15: "phone check", 16: "signer"}
DEFAULT_MODEL = {1: "gemini-4-argon", 2: "gemini-4-argon", 3: "gemini-3.5-flash-lite", 4: "gemini-4-argon",
                 5: "gemini-4-argon", 9: "gemini-4-argon", 10: "gemini-4-argon", 16: "gemini-4-argon"}
HARBOR_PARTS = {"instruction.md": "agent reads this", "task.toml": "config", "README.md": "overview",
                "environment": "agent container", "solution": "reference (hidden from the agent)",
                "tests": "verifier (separate container)"}


def iso(ts):
    try:
        return dt.datetime.fromisoformat(ts).timestamp()
    except (TypeError, ValueError):
        return None


def did(e):
    """One line for what a ledger verdict produced."""
    d, n = e.get("details") or {}, e.get("stage")
    if d.get("crashed") or d.get("not_implemented"):
        return (e.get("findings") or ["crashed"])[0]
    if n == 2:
        return "wrote the proposal: %s" % d.get("title", "")
    if n == 3 and d.get("checks"):
        bad = [c["check"] for c in d["checks"] if not c.get("passed")]
        return "checked %d rules%s" % (len(d["checks"]), ": failed " + ", ".join(bad) if bad else ", all passed")
    if n == 4 and d.get("approved_by") == "user":
        return "approved by you" + (": " + d["summary"][:200] if d.get("summary") else "")
    if n == 4:
        return "%s: %s" % (d.get("verdict") or e.get("verdict"), (d.get("summary") or "")[:220])
    if n == 5:
        return "wrote the Harbor package"
    if n == 6:
        count = lambda v: v if isinstance(v, int) else len(v or [])
        return "%d errors, %d warnings in %s files" % (count(d.get("errors")), count(d.get("warnings")), d.get("files_checked", "?"))
    if n in (7, 8, 15) and "mean_reward" in d:
        trials = d.get("trials")
        return "mean reward %s over %s trial(s)" % (d.get("mean_reward"), len(trials) if isinstance(trials, list) else trials)
    if n == 9:
        return (d.get("summary") or "")[:240]
    if n == 10:
        return "%s: %s" % (d.get("decision") or e.get("verdict"), (d.get("summary") or "")[:220])
    return ((e.get("findings") or [""])[0] or "")[:240]


def safe_did(e):
    try:
        return did(e)
    except Exception:  # a details shape this page doesn't know yet: fall back to the first finding
        return str((e.get("findings") or [""])[0])[:240]


def task_detail(task, procs):
    if task not in tasks():
        return {"error": "unknown task"}
    b = build_stages(task, procs)
    work, pkg = RUNS / "tasks" / task, ROOT / "tasks" / task
    spec = b["spec"]
    # proposals, every round
    rounds = []
    for f in sorted(work.glob("proposal-r*.json"), key=lambda f: int(re.sub(r"\D", "", f.stem) or 0)):
        d = read_json(f) or {}
        rounds.append({"round": int(re.sub(r"\D", "", f.stem) or 0), "file": str(f.relative_to(ROOT)), "at": mtime(f),
                       "proposal": trim(d.get("proposal")), "agent": {k: v for k, v in (d.get("agent") or {}).items() if k != "interaction"}})
    cur = read_json(work / "proposal.json")
    if cur and (not rounds or cur.get("proposal") != (read_json(ROOT / rounds[-1]["file"]) or {}).get("proposal")):
        rounds.append({"round": "current", "file": str((work / "proposal.json").relative_to(ROOT)), "at": mtime(work / "proposal.json"),
                       "proposal": trim(cur.get("proposal")), "agent": {k: v for k, v in (cur.get("agent") or {}).items() if k != "interaction"}})
    # the Harbor package
    files = []
    if pkg.exists():
        for f in sorted(pkg.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                rel = str(f.relative_to(pkg))
                files.append({"path": rel, "full": str(f.relative_to(ROOT)), "bytes": f.stat().st_size,
                              "part": HARBOR_PARTS.get(rel.split("/")[0])})
    # who did what, in time order
    events = []
    cand = b["cand"]
    if cand:
        events.append({"at": mtime(RUNS / "mine" / b["port"] / "candidates.json"), "stage": 1, "role": "miner",
                       "model": "gemini-4-argon", "verdict": "pass", "seconds": cand["usage"].get("seconds"),
                       "tokens": [cand["usage"].get("tokens_in"), cand["usage"].get("tokens_out")],
                       "did": "read %s commits, picked %d candidates" % (cand["usage"].get("commits", "?"), len(cand["candidates"]))})
    cut = read_json(work / "cut.json") or {}
    ledger_has_5 = any(e.get("stage") == 5 for e in b["ledger_rows"])
    if cut and not ledger_has_5:
        u = cut.get("_usage") or {}
        assembled = (pkg / "task.toml").exists()  # cut.json is the statement; assembling writes the package
        events.append({"at": mtime(work / "cut.json"), "stage": 5, "role": "author", "model": u.get("model") or "gemini-4-argon",
                       "verdict": "pass", "seconds": u.get("seconds"), "tokens": [u.get("input"), u.get("output")],
                       "did": "wrote the Harbor package" if assembled else "wrote the task statement (package not assembled yet)",
                       "file": "tasks/%s/instruction.md" % task if (pkg / "instruction.md").exists() else None})
    for e in b["ledger_rows"]:
        n, ag = e.get("stage"), e.get("agent") or {}
        model = ag.get("model") or (cut.get("_usage") or {}).get("model") if n == 5 else ag.get("model")
        by_user = (e.get("details") or {}).get("approved_by") == "user"
        ev = {"at": iso(e.get("ts")), "stage": n, "round": e.get("round"), "role": "you" if by_user else STAGE_ROLE.get(n),
              "model": model, "verdict": e.get("verdict"), "seconds": e.get("seconds"),
              "tokens": [ag.get("input_tokens"), ag.get("output_tokens")], "did": redact(safe_did(e)),
              "findings": [redact(f)[:600] for f in (e.get("findings") or [])][:10]}
        if n == 2:
            f = work / ("proposal-r%s.json" % e.get("round"))
            ev["file"] = str(f.relative_to(ROOT)) if f.exists() else None
        if n == 5 and e.get("passed"):
            ev["file"] = "tasks/%s/instruction.md" % task
        events.append(ev)
    for fx in read_jsonl(work / "fixes.jsonl"):
        u = fx.get("usage") or {}
        if fx.get("holes") is not None:
            text = "closed %d hole(s) found by %s: %s" % (len(fx.get("holes") or []), fx.get("found_by", "?"), fx.get("attack", ""))
            stage = 13
        else:
            text = fx.get("summary") or "applied %d fix(es)" % len(fx.get("applied") or [])
            stage = None
        events.append({"at": iso(fx.get("at") or fx.get("ts")), "stage": stage, "role": "fixer",
                       "model": u.get("model") or ("gemini-4-argon" if stage == 13 else None), "verdict": "fix",
                       "seconds": u.get("seconds"), "tokens": [u.get("input") or u.get("input_tokens"), u.get("output") or u.get("output_tokens")],
                       "did": redact(text)[:300], "findings": [redact(str(x))[:400] for x in (fx.get("fixes") or fx.get("applied") or [])][:8]})
    for a in b["ags"]:
        events.append({"at": a.get("started"), "stage": a.get("stage"), "role": {"redteam": "attacker", "rollout": "solver"}.get(a["role"], a["role"]),
                       "model": a.get("model"), "verdict": a.get("status"), "seconds": a.get("seconds"), "agent_id": a["id"],
                       "did": ("attack: %s · " % a["attack"] if a.get("attack") else "") + "reward %s%s" % (
                           a.get("best_reward"), " at %s×" % a["speedup"] if a.get("speedup") else ""), "tokens": [None, None]})
    events = [e for e in events if e.get("at")]
    events.sort(key=lambda e: e["at"])
    roles = {}
    for e in events:
        r = roles.setdefault(e["role"], {"role": e["role"], "models": [], "runs": 0, "seconds": 0, "tokens_in": 0,
                                         "tokens_out": 0, "stages": []})
        if e.get("model") and e["model"] not in r["models"]:
            r["models"].append(e["model"])
        r["runs"] += 1
        r["seconds"] += e.get("seconds") or 0
        r["tokens_in"] += (e.get("tokens") or [0])[0] or 0
        r["tokens_out"] += ((e.get("tokens") or [0, 0]) + [0])[1] or 0
        if e.get("stage") and e["stage"] not in r["stages"]:
            r["stages"].append(e["stage"])
    st = {x["n"]: x for x in b["stages"]}
    toml = (pkg / "task.toml").read_text() if (pkg / "task.toml").exists() else ""
    timeout = re.search(r"^\[agent\][^\[]*?timeout_sec\s*=\s*([\d.]+)", toml, re.M | re.S)
    return {"task": task, "title": task_title(task, b["stages"]), "repo": task_repo(spec), "port": b["port"],
            "spec": {k: spec.get(k) for k in ("base", "reference", "base_tps", "reference_tps", "reference_speedup", "needed_speedup",
                                               "target_speedup", "phone")},
            "difficulty": re.search(r'^difficulty\s*=\s*"([^"]+)"', toml, re.M).group(1) if re.search(r'^difficulty\s*=\s*"([^"]+)"', toml, re.M) else None,
            "agent_timeout": float(timeout.group(1)) if timeout else None,
            "accepted": st[16]["status"] in PASSED,
            "stages": [{"n": x["n"], "label": x["label"], "status": x["status"], "running": x.get("running"),
                        "history": x["history"]} for x in b["stages"]],
            "proposals": rounds, "package": files, "package_dir": "tasks/%s" % task if pkg.exists() else None,
            "events": events, "roles": sorted(roles.values(), key=lambda r: min(r["stages"] or [99]))}


# --- HTTP ----------------------------------------------------------------------------------------------------

class App:
    def __init__(self, remote_enabled):
        self.logs = LogTail()
        self.remote = Remote(self.logs, enabled=False)  # no Mac mini in the loop
        self.health = Health()
        self.procs, self.cache, self.lock = [], {}, threading.Lock()
        self.cat, self.all_agents = (0, []), []
        repos.ensure_builtin()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            try:
                self.logs.poll()
                self.procs = local_procs()
            except Exception:  # keep following even if one pass fails
                pass
            time.sleep(1.5)

    def catalog(self):
        if time.time() - self.cat[0] > 3:
            sink = []
            self.cat = (time.time(), catalog(self.procs, sink))
            sink.sort(key=lambda a: (a["status"] != "running", -(a.get("started") or 0)))
            self.all_agents = sink
        return self.cat[1]

    def state(self, task, repo=""):
        cat = self.catalog()
        known = {r["slug"] for r in repos.all_repos()}
        by_task = {t["task"]: t for t in cat}
        if task in by_task and (not repo or by_task[task]["repo"] == repo):
            repo = by_task[task]["repo"] or repo
        if repo not in known:
            repo = by_task[task]["repo"] if task in by_task and by_task[task]["repo"] else repos.BUILTIN["slug"]
        mine = [t["task"] for t in cat if t["repo"] == repo]
        if task not in mine:
            task = mine[0] if mine else None
        key = (task, repo)
        with self.lock:
            hit = self.cache.get(key)
            if hit and time.time() - hit[0] < 1.0:
                return hit[1], hit[2]
        snap = snapshot(task, repo, self.procs, self.logs, self.remote, self.health, cat)
        snap["all_agents"] = self.all_agents
        body = json.dumps(snap, default=str)
        digest = hashlib.sha1(json.dumps({k: v for k, v in snap.items() if k != "now"}, default=str,
                                         sort_keys=True).encode()).hexdigest()
        with self.lock:
            self.cache[key] = (time.time(), body, digest)
        return body, digest


ALLOWED = (".diff", ".md", ".json", ".jsonl", ".txt", ".toml", ".sh", ".py", ".cl", ".cpp", ".h", ".yaml", ".yml")
ALLOWED_NAMES = ("Dockerfile",)


def safe_file(rel):
    p = (ROOT / rel).resolve()
    for top in (RUNS, ROOT / "tasks"):
        try:
            p.relative_to(top.resolve())
        except ValueError:
            continue
        if p.is_file() and (p.suffix in ALLOWED or p.name in ALLOWED_NAMES) and "private" not in p.parts:
            return p
    return None


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                return self.send(200, (UI / "index.html").read_bytes(), "text/html")
            if url.path == "/api/state":
                return self.send(200, app.state(q.get("task", ""), q.get("repo", ""))[0])
            if url.path == "/api/file":
                p = safe_file(q.get("path", ""))
                if not p:
                    return self.send(404, json.dumps({"error": "not found"}))
                return self.send(200, redact(p.read_text(errors="replace")[:400000]), "text/plain")
            if url.path == "/api/stream":
                return self.stream(q.get("task", ""), q.get("repo", ""))
            if url.path == "/api/task":
                return self.send(200, json.dumps(task_detail(q.get("task", ""), app.procs), default=str))
            return self.send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0) or 0) or b"{}")
            except ValueError:
                return self.send(400, json.dumps({"error": "bad JSON"}))
            if url.path == "/api/repos":
                rec = repos.add(body.get("url", ""))
                app.cache.clear()
                return self.send(400 if rec.get("error") and not rec.get("slug") else 200, json.dumps(rec, default=str))
            if url.path == "/api/repos/refresh":
                rec = repos.refresh(body.get("slug", ""))
                app.cache.clear()
                return self.send(200, json.dumps(rec, default=str))
            return self.send(404, json.dumps({"error": "not found"}))

        def stream(self, task, repo=""):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last, beat = None, time.time()
            try:
                while True:
                    body, digest = app.state(task, repo)
                    if digest != last:
                        self.wfile.write(("event: state\ndata: %s\n\n" % body).encode())
                        self.wfile.flush()
                        last, beat = digest, time.time()
                    elif time.time() - beat > 10:
                        self.wfile.write(b": alive\n\n")
                        self.wfile.flush()
                        beat = time.time()
                    time.sleep(1)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return  # a browser tab closed or reloaded mid-request
        super().handle_error(request, client_address)


def watch_code():
    """Restart in place when ui/server.py changes, so edits take effect without anyone restarting it. A file that
    doesn't compile yet (a half-finished edit) is skipped until it does."""
    me = Path(__file__).resolve()
    watched = [me, UI / "repos.py"]
    seen = [mtime(f) for f in watched]
    while True:
        time.sleep(1)
        if [mtime(f) for f in watched] == seen:
            continue
        time.sleep(0.5)
        seen = [mtime(f) for f in watched]
        try:
            for f in watched:
                compile(f.read_text(), str(f), "exec")
        except SyntaxError as exc:
            print("ui code changed but doesn't compile yet (%s); still running the old version" % exc, flush=True)
            continue
        print("%s ui code changed; restarting" % time.strftime("%H:%M:%S"), flush=True)
        os.execv(sys.executable, [sys.executable, str(me)] + sys.argv[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-remote", action="store_true", help="don't ssh to the Mac mini")
    args = ap.parse_args()
    app = App(not args.no_remote)
    server = Server((args.host, args.port), handler(app))
    threading.Thread(target=watch_code, daemon=True).start()
    print("Task Factory monitor on http://%s:%d  (Ctrl-C to stop)" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
