"""Calibrate several tasks on the cloud gateway in one batched call (one Device Run session per port), then run
one /try per task with its oracle patch.

    uv run python cloud/fleet_check.py --url https://tf-gateway-....run.app smollm2-rope-smoke smollm2-image2d-decode

Results: runs/tasks/<task>/cloud-calibrate.json and cloud-try.json   Progress: tail -f logs/cloud-gateway.log
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates import RUNS, call, log, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--url", required=True)
    ap.add_argument("--no-force", action="store_true")
    ap.add_argument("--batch", action="store_true", help="one Device Run session for all tasks of a port")
    ap.add_argument("--grade", action="store_true", help="also /grade the oracle and the unchanged base")
    args = ap.parse_args()
    force = not args.no_force
    if args.batch:  # one session for every task of a port
        replies = [call(args.url, "/calibrate", {"tasks": args.tasks, "force": force, "batch": True}, "X-TF-Factory", "gateway-factory-key")]
    else:  # one session per task, concurrently
        with ThreadPoolExecutor(len(args.tasks)) as pool:
            replies = list(pool.map(lambda t: call(args.url, "/calibrate", {"task": t, "force": force}, "X-TF-Factory",
                                                   "gateway-factory-key"), args.tasks))
    for cal in replies:
        log.info("calibrate (%ss): %s", cal.get("seconds"), json.dumps({k: v for k, v in cal.items() if k != "tasks"})[:400])
        for name, c in (cal.get("tasks") or {}).items():
            log.info("  %s: %s", name, c)
            (RUNS / "tasks" / name / "cloud-calibrate.json").write_text(json.dumps(c, indent=1))
    def oracle(name):
        # the oracle limited to the task's in-scope files, when the cut's full reference diff touches others
        d = RUNS / "tasks" / name
        return ((d / "oracle-in-scope.diff") if (d / "oracle-in-scope.diff").exists() else (d / "oracle.diff")).read_text()

    def one(job):
        name, path, which = job
        patch = oracle(name) if which == "oracle" else ""
        header, keyfile = ("X-TF-Agent", "gateway-agent-key") if path == "/try" else ("X-TF-Verifier", "gateway-verifier-key")
        r = call(args.url, path, {"task": name, "patch": patch}, header, keyfile)
        out = "cloud-try.json" if (path, which) == ("/try", "oracle") else "cloud-%s-%s.json" % (path.strip("/"), which)
        (RUNS / "tasks" / name / out).write_text(json.dumps(r, indent=1))
        return job, r
    jobs = [(n, "/try", "oracle") for n in args.tasks]
    if args.grade:  # the hidden-prompt grade of the oracle (expect reward ~1) and of the unchanged base (expect ~0)
        jobs += [(n, "/grade", w) for n in args.tasks for w in ("oracle", "base")]
    with ThreadPoolExecutor(len(jobs)) as pool:
        for (name, path, which), r in pool.map(one, jobs):
            log.info("%s %s (%s): %s scoring=%s", path.strip("/"), name, which, summary(r), r.get("scoring"))


if __name__ == "__main__":
    main()
