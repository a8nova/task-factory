"""Gates that run a task's own packaged solutions through Harbor and the device gateway: no-op, baseline and
reference (lifecycle stages 7, 8 and 15).

    uv run python -m lifecycle.gates smollm2-image2d-decode noop
    uv run python -m lifecycle.gates smollm2-image2d-decode baseline --trials 3
    uv run python -m lifecycle.gates smollm2-image2d-decode reference --trials 3
    uv run python -m lifecycle.gates smollm2-rope-smoke rescore --runs runs/agents/smollm2-rope-smoke/<run> [...]

Each run uses a local validation copy under runs/harbor-local/<task>-<gate>/ with the network opened, because
Harbor's local Docker backend can't enforce the task's "gateway only" allowlist (the shipped task keeps it).
The no-op uses Harbor's nop agent (no submission at all); baseline and reference use the oracle agent with
solve.sh pointed at baseline.sh or reference.sh. Results: runs/tasks/<task>/gates.json. Log: logs/gates.log
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from common import ROOT, RUNS, get_logger

log = get_logger("gates")
KEYS = Path.home() / ".config" / "task-factory"
DOCKER_BIN = "/Applications/Docker.app/Contents/Resources/bin"
GATEWAY_FOR_DOCKER = "http://host.docker.internal:8798"
NEAR_ZERO, NEAR_ONE = 0.1, 0.9  # tolerance for measurement noise between trials


def local_copy(task, gate):
    src = ROOT / "tasks" / task
    dst = RUNS / "harbor-local" / ("%s-%s" % (task, gate))
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    toml = dst / "task.toml"
    lines = [l for l in toml.read_text().splitlines() if not l.startswith("allowed_hosts")]
    toml.write_text("\n".join(l.replace('network_mode = "allowlist"', 'network_mode = "public"') for l in lines) + "\n")
    if gate == "reference":
        (dst / "solution" / "solve.sh").write_text(
            '#!/bin/bash\n# Validation copy: packages the reference submission.\nset -euo pipefail\n'
            'bash "$(dirname "$0")/reference.sh"\n')
    return dst


def run_harbor(task, gate, agent, trials):
    path = local_copy(task, gate)
    # One jobs folder per call: several agents run gates at once, so "newest job folder" would mix them up.
    jobs = RUNS / "harbor-jobs" / ("%s-%s-%s" % (task, gate, time.strftime("%Y%m%d-%H%M%S")))
    jobs.mkdir(parents=True, exist_ok=True)
    before = set()
    env = {**os.environ, "PATH": "%s:%s" % (DOCKER_BIN, os.environ["PATH"]), "TF_GATEWAY": GATEWAY_FOR_DOCKER,
           "TF_VERIFIER_KEY": (KEYS / "gateway-verifier-key").read_text().strip(),
           "TF_GATEWAY_HMAC": (KEYS / "gateway-hmac").read_text().strip()}
    t0 = time.time()
    with open(ROOT / "logs" / ("harbor-%s.log" % gate), "w") as out:
        done = subprocess.run(["harbor", "run", "-y", "--agent", agent, "--path", str(path), "-k", str(trials),
                               "-n", "1", "-o", str(jobs)], env=env, stdout=out, stderr=subprocess.STDOUT)
    new = sorted({p.name for p in jobs.glob("*")} - before)
    if done.returncode != 0 or not new:
        return {"error": "harbor exited %d (see logs/harbor-%s.log)" % (done.returncode, gate)}, time.time() - t0
    job = jobs / new[-1]
    trials_out = []
    for trial in sorted(job.glob("*/")):
        reward_file = trial / "verifier" / "reward.json"
        details = trial / "verifier" / "details.json"
        if not reward_file.exists():
            stdout = trial / "verifier" / "test-stdout.txt"
            trials_out.append({"trial": trial.name, "reward": None,
                               "error": stdout.read_text()[-600:] if stdout.exists() else "no verifier output"})
            continue
        r = json.loads(reward_file.read_text())
        d = json.loads(details.read_text()) if details.exists() else {}
        trials_out.append({"trial": trial.name, "reward": r.get("reward"), "valid": r.get("valid"),
                           "speedup": r.get("speedup"), "decode_tok_s": r.get("decode_tok_s"),
                           "flags": d.get("flags"), "error": d.get("error")})
    return {"job": str(job.relative_to(ROOT)), "trials": trials_out}, time.time() - t0


def verdict(stage, passed, findings, details, seconds):
    return {"stage": stage, "verdict": "pass" if passed else "fail", "passed": passed, "findings": findings,
            "details": details, "seconds": round(seconds), "agent": {"model": None, "interaction": None}}


def summarize(stage, task, agent, trials, check):
    res, secs = run_harbor(task, stage, agent, trials)
    if "error" in res:
        return verdict(stage, False, [res["error"]], res, secs)
    rewards = [t["reward"] for t in res["trials"] if t["reward"] is not None]
    infra = [t for t in res["trials"] if t["reward"] is None]
    findings = ["%d trial(s) wrote no reward (infrastructure error): %s" % (len(infra), infra[0]["error"][:200])] if infra else []
    mean = statistics.mean(rewards) if rewards else None
    passed, why = check(rewards, mean)
    if why:
        findings.append(why)
    res.update({"mean_reward": mean, "rewards": rewards})
    v = verdict(stage, passed and not infra, findings, res, secs)
    log.info("%s %s: %s, rewards %s, mean %s (%ds)", task, stage, v["verdict"], rewards, mean, secs)
    return v


def noop(task):
    return summarize("noop", task, "nop", 1, lambda rw, m: (bool(rw) and all(r == 0 for r in rw),
                                                           None if rw and all(r == 0 for r in rw) else "an empty submission earned credit"))


def baseline(task, trials=3):
    return summarize("baseline", task, "oracle", trials,
                     lambda rw, m: (m is not None and m <= NEAR_ZERO,
                                    None if m is not None and m <= NEAR_ZERO else "baseline mean %s is not near 0" % m))


def reference(task, trials=3):
    return summarize("reference", task, "oracle", trials,
                     lambda rw, m: (m is not None and m >= NEAR_ONE and len(rw) == trials,
                                    None if m is not None and m >= NEAR_ONE else "reference mean %s is not near 1" % m))


def reward_of(speedup, target, mapping, valid):
    """Same rule as device/gateway.py reward_of: 0 at the baseline, 1 at the per-task target, linear or log."""
    import math
    if not valid or not speedup or speedup <= 0:
        return 0.0
    r = math.log(speedup) / math.log(target) if mapping == "log" else (speedup - 1.0) / (target - 1.0)
    return max(0.0, min(1.0, r))


def rescore(task, run_dirs, mapping=None):
    """Re-score finished agent trials from their recorded final submission (validity and speedup) with the
    current scoring rule. Trials whose final grade was an infrastructure error get no score; trials still
    running are listed as pending. Writes runs/tasks/<task>/rollouts-rescored.json."""
    scoring = json.loads((RUNS / "tasks" / task / "scoring.json").read_text())
    target, mapping = scoring["target_speedup"], mapping or scoring["mapping"]
    rows, pending = [], []
    for d in run_dirs:
        d = Path(d)
        trials = sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []
        if not trials:
            pending.append(str(d))
        for t in trials:
            summary = t / "summary.json"
            if not summary.exists():
                pending.append(str(t))
                continue
            sm = json.loads(summary.read_text())
            final = sm.get("final") or {}
            infra = final.get("infra_error")
            valid, speedup = bool(final.get("valid")), final.get("speedup")
            rows.append({"run": d.name, "trial": t.name, "role": sm.get("role"), "model": sm.get("model"),
                         "status": sm.get("status"), "submitted": bool(final), "valid": valid, "speedup": speedup,
                         "old_reward": sm.get("reward"),
                         "new_reward": None if infra else round(reward_of(speedup, target, mapping, valid), 4),
                         "infra_error": infra, "flags": final.get("flags")})
    scored = [r["new_reward"] for r in rows if r["new_reward"] is not None and r["role"] == "rollout"]
    mean = statistics.mean(scored) if scored else None
    out = {"task": task, "mapping": mapping, "target_speedup": target, "formula": scoring["formula"],
           "trials": rows, "pending": pending,
           "rollout_stats": {"n": len(scored), "mean": mean,
                             "population_variance": statistics.pvariance(scored) if len(scored) > 1 else None}}
    (RUNS / "tasks" / task / "rollouts-rescored.json").write_text(json.dumps(out, indent=1))
    log.info("%s rescored %d trials (%d pending) with %s mapping: rollout mean %s", task, len(rows), len(pending),
             mapping, mean)
    return out


def record(task, v):
    path = RUNS / "tasks" / task / "gates.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[v["stage"]] = v
    path.write_text(json.dumps(data, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("gate", choices=["noop", "baseline", "reference", "rescore"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--mapping", choices=["linear", "log"])
    args = ap.parse_args()
    if args.gate == "rescore":
        out = rescore(args.task, args.runs, args.mapping)
        print(json.dumps({k: out[k] for k in ("mapping", "target_speedup", "rollout_stats", "pending")}, indent=1))
        for r in out["trials"]:
            print(r["run"], r["trial"], r["model"], "valid", r["valid"], "speedup", r["speedup"],
                  "old", r["old_reward"], "new", r["new_reward"])
        return
    v = noop(args.task) if args.gate == "noop" else globals()[args.gate](args.task, args.trials)
    record(args.task, v)
    print(json.dumps({k: v[k] for k in ("stage", "verdict", "findings")}, indent=1))


if __name__ == "__main__":
    main()
