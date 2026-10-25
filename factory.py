"""The task lifecycle state machine: runs stages 2-16 from CONTRACT.md in order, writes every verdict to the
ledger, sends failed or returned stages back to their author (at most 2 revisions), and stops on a reject.

    uv run python factory.py new-task --task smollm2-rope-smoke --port smollm2-135m-instruct --candidate 887bd7e
    uv run python factory.py run --task T --port P --candidate C [--profile smoke|full] [--from N] [--to M]
    uv run python factory.py run ... --skip-phone --trials 1      # cheap pass that skips phone stages
    uv run python factory.py status [--task T]
    uv run python factory.py qa-test --task T                   # planted-flaw test of proposal QA and review

Profiles:
  full  (default) Argon authors and reviews; trials and agent budgets as configured (--trials, role defaults).
  smoke gemini-3.8-flash authors and reviews; 1 trial per gate; 1 explorer, 1 calibrator, 1 red-team agent, 2
        rollout agents; every agent 180 s with at most 1 phone call. Smoke verdicts are tagged "smoke" in the
        ledger and never count toward acceptance: full-profile stages only ever see full-profile history.

Stages whose module or function doesn't exist yet record "fail: stage not implemented yet" and stop the run.
Ledger: runs/ledger/<task>.jsonl   Progress: tail -f logs/factory.log
"""
import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from common import ADRENO_LLMS, ARGON, FLASH, ROOT, RUNS, get_logger
from lifecycle import ledger

log = get_logger("factory")
MAX_ROUNDS = 2
STAGES = {2: "propose", 3: "proposal_qa", 4: "review_proposal", 5: "implement", 6: "static_checks", 7: "noop",
          8: "baseline", 9: "implementation_qa", 10: "checkin_review", 11: "explore", 12: "calibrate",
          13: "redteam", 14: "rollouts", 15: "reference", 16: "final_signoff"}
NEEDS_PHONE = {7, 8, 11, 12, 13, 14, 15}
PROFILES = {
    "full": {"author_model": ARGON, "gate_trials": None, "agents": {}, "budget_sec": None, "max_phone_calls": None,
             "timeout_min": 360, "max_rounds": MAX_ROUNDS, "keep_going": False, "lean_review": False},
    # keep_going: record a failed stage and move on, so one smoke pass shows every broken stage.
    "smoke": {"author_model": FLASH, "gate_trials": 1,
              "agents": {"explorer": 1, "calibrator": 1, "redteam": 1, "rollout": 2},
              "budget_sec": 180, "max_phone_calls": 1, "timeout_min": 3, "max_rounds": 0, "keep_going": True,
              "lean_review": True},
}
KEY_FILES = ("gateway-hmac", "gateway-verifier-key", "gateway-factory-key", "gateway-token")
DEFAULT_KEYS = Path.home() / ".config" / "task-factory"
# Modules whose authoring/reviewing model is the module-level ARGON name; the smoke profile rebinds it.
AUTHOR_MODULES = ("lifecycle.proposal", "cut", "lifecycle.review", "lifecycle.trajectories")


def lazy(module, name):
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


def missing(stage):
    return {"verdict": "fail", "passed": False, "findings": ["stage not implemented yet"],
            "details": {"stage": STAGES[stage], "not_implemented": True}}


def as_verdict(result, stage):
    """Stage functions return a contract verdict; wrap anything else."""
    if isinstance(result, dict) and "verdict" in result and "passed" in result:
        return result
    return {"verdict": "pass", "passed": True, "findings": [], "details": {"result": result}}


def call(fn, *args, **kwargs):
    """Call fn, passing only the keyword arguments it accepts (stage modules evolve independently)."""
    params = inspect.signature(fn).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **kwargs)


def apply_profile(profile):
    """Point every authoring/reviewing module at the profile's model (module-level ARGON name)."""
    model = PROFILES[profile]["author_model"]
    patched = []
    for name in AUTHOR_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "ARGON"):
            mod.ARGON = model
            patched.append(name)
    return model, patched


def apply_gateway(url, keys_dir):
    """Point every phone stage at one gateway. Tokens are files in keys_dir (never on the command line):
    gateway-hmac (verifies signed results), gateway-verifier-key (/grade), gateway-factory-key (/calibrate,
    /measure), and optionally gateway-token (a bearer token, if the gateway sits behind auth such as Cloud Run).
    Subprocesses (Harbor verifiers, scripts) get TF_GATEWAY, TF_GATEWAY_KEYS_DIR and TF_GATEWAY_TOKEN; in-process
    modules are rebound here: lifecycle.gates.GATEWAY_FOR_DOCKER / KEYS and agents.VERIFIER_KEY_FILE."""
    keys_dir = Path(keys_dir).expanduser()
    present = [k for k in KEY_FILES if (keys_dir / k).is_file()]
    os.environ["TF_GATEWAY"] = url
    os.environ["TF_GATEWAY_URL"] = url
    os.environ["TF_GATEWAY_KEYS_DIR"] = str(keys_dir)
    if "gateway-token" in present:
        os.environ["TF_GATEWAY_TOKEN"] = (keys_dir / "gateway-token").read_text().strip()
    gates = lazy("lifecycle.gates", "GATEWAY_FOR_DOCKER") is not None and importlib.import_module("lifecycle.gates")
    if gates:
        gates.GATEWAY_FOR_DOCKER = url
        gates.KEYS = keys_dir
    try:
        agents = importlib.import_module("agents")
        if "gateway-verifier-key" in present and hasattr(agents, "VERIFIER_KEY_FILE"):
            agents.VERIFIER_KEY_FILE = keys_dir / "gateway-verifier-key"  # read at first /grade
    except ImportError:
        pass
    return present


class Context:
    def __init__(self, args):
        self.args = args
        self.task, self.port = args.task, args.port
        self.profile = getattr(args, "profile", "full")
        self.settings = PROFILES[self.profile]
        self.work = RUNS / "tasks" / self.task
        self.work.mkdir(parents=True, exist_ok=True)
        self.task_dir = ROOT / "tasks" / self.task
        spec_path = self.work / "spec.json"
        self.spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        self.spec.setdefault("port", self.port)
        if self.spec.get("kind") == "spec":
            # Built from a spec: no mined candidate and no reference commit; calibration supplies both later.
            self.candidate = {"commit": self.spec["base"], "title": self.spec.get("title", self.task), "kind": "spec",
                              "metric": self.spec.get("metric")}
        else:
            if not args.candidate:
                raise SystemExit("--candidate is required for tasks built from a commit (new-task --spec for spec tasks)")
            self.candidate = find_candidate(self.port, args.candidate)
        if "base" not in self.spec:
            self.spec["base"] = parent_of(self.candidate["commit"])
        ensure_pack(self.port, self.spec["base"])
        acc = ROOT / "private" / "acceptance.toml"
        self.acceptance = tomllib.loads(acc.read_text()) if acc.exists() else {}
        oracle = self.work / "oracle.diff"
        self.oracle = oracle.read_text() if oracle.exists() else None
        prop = Path(args.proposal) if getattr(args, "proposal", None) else self.work / "proposal.json"
        self.proposed = json.loads(prop.read_text()) if prop.exists() else None
        self.qa = None

    def trials(self):
        return self.settings["gate_trials"] or self.args.trials

    def agent_kwargs(self, role):
        n = self.settings["agents"].get(role)
        kw = {"budget_sec": self.settings["budget_sec"], "max_phone_calls": self.settings["max_phone_calls"],
              "gateway_url": self.args.gateway}
        if n is not None:
            kw["n"] = n
        elif role != "explorer":
            kw["n"] = self.args.trials if self.args.trials > 1 else None
        return {k: v for k, v in kw.items() if v is not None}

    def history(self):
        """What review stages may see: only this profile's verdicts, so smoke never counts toward acceptance."""
        return ledger.history(self.task, self.profile)

    def save_proposal(self, round_):
        (self.work / "proposal.json").write_text(json.dumps(self.proposed, indent=1))
        (self.work / f"proposal-r{round_}.json").write_text(json.dumps(self.proposed, indent=1))


def find_candidate(port, commit):
    cands = json.loads((RUNS / "mine" / port / "candidates.json").read_text())["candidates"]
    return next(c for c in cands if c["commit"].startswith(commit) or commit.startswith(c["commit"][:8]))


def parent_of(commit):
    return subprocess.run(["git", "-C", str(ADRENO_LLMS), "rev-parse", commit + "^1"], capture_output=True, text=True,
                          check=True).stdout.strip()


def ensure_pack(port, base):
    pack_file = RUNS / "mine" / port / f"pack-{base[:10]}.txt"
    if not pack_file.exists():
        import pack
        text, stats = pack.pack(port, base)
        pack_file.write_text(text)
        pack_file.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    return pack_file


def port_oracle_diff(port, base, reference, files=None):
    """The reference commit's change for this port only, without docs; with files, only those paths
    (the mined speedup files), so the reference never touches scripts, timing code or other off-limits files."""
    paths = list(files) if files else [f"src/models/{port}", f":!src/models/{port}/*.md", f":!src/models/{port}/**/*.md"]
    return subprocess.run(["git", "-C", str(ADRENO_LLMS), "diff", base, reference, "--", *paths],
                          capture_output=True, text=True, check=True).stdout


# --- stage runners: each returns a contract verdict --------------------------------------------------------

def run_propose(ctx, round_):
    propose = lazy("lifecycle.proposal", "propose")
    if not propose:
        return missing(2)
    t0 = time.time()
    ctx.proposed = propose(ctx.candidate, ctx.spec)
    ctx.save_proposal(round_)
    p = ctx.proposed["proposal"]
    return {"verdict": "pass", "passed": True, "findings": [],
            "details": {"title": p["title"], "slug": p["slug"], "difficulty": p["expected_difficulty"]},
            "seconds": round(time.time() - t0), "agent": ctx.proposed["agent"]}


def run_proposal_qa(ctx, round_):
    qa = lazy("lifecycle.proposal", "proposal_qa")
    if not qa or not ctx.proposed:
        return missing(3)
    ctx.qa = call(qa, ctx.proposed, spec=ctx.spec)
    return ctx.qa


def run_review(ctx, round_):
    review = lazy("lifecycle.proposal", "review_proposal")
    if not review or not ctx.proposed:
        return missing(4)
    return review(ctx.proposed, ctx.qa, ctx.spec, ctx.oracle)


def fix_proposal(ctx, findings, round_):
    revise = lazy("lifecycle.proposal", "revise")
    if not revise:
        return False
    ctx.proposed = revise(ctx.proposed, findings)
    ctx.save_proposal(round_)
    return True


def run_implement(ctx, round_):
    """Stage 5: cut.py syncs the gateway's calibration into spec.json (--sync), then Cut (the profile's authoring
    model) + assemble."""
    import cut as cut_mod
    source = inspect.getsource(cut_mod)
    argv = ["cut.py", ctx.task]
    if '"--sync"' in source:
        argv.append("--sync")
    if '"--model"' in source:
        argv += ["--model", ctx.settings["author_model"]]
    if '"--timeout-min"' in source:
        argv += ["--timeout-min", str(ctx.settings["timeout_min"])]
    t0 = time.time()
    saved = sys.argv
    sys.argv = argv
    try:
        cut_mod.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            return {"verdict": "fail", "passed": False, "findings": [f"cut.py stopped: {exc.code}"],
                    "details": {"argv": argv}}
    finally:
        sys.argv = saved
    if not (ctx.task_dir / "instruction.md").exists():
        return {"verdict": "fail", "passed": False, "findings": [f"{ctx.task_dir}/instruction.md was not written"],
                "details": {"argv": argv}}
    spec_path = ctx.work / "spec.json"
    ctx.spec = json.loads(spec_path.read_text()) if spec_path.exists() else ctx.spec
    cut_json = ctx.work / "cut.json"
    usage = json.loads(cut_json.read_text()).get("_usage", {}) if cut_json.exists() else {}
    return {"verdict": "pass", "passed": True, "findings": [],
            "details": {"argv": argv[1:], "base_tps": ctx.spec.get("base_tps"),
                        "reference_tps": ctx.spec.get("reference_tps"), "target_speedup": ctx.spec.get("target_speedup"),
                        "dev_prompts": len(ctx.spec.get("dev_expected", [])), "weaker_check": ctx.spec.get("weaker_check")},
            "seconds": round(time.time() - t0),
            "agent": {"model": ctx.settings["author_model"], "interaction": usage.get("interaction"),
                      "input_tokens": usage.get("input"), "output_tokens": usage.get("output")}}


def simple(module, name, stage, args_of, kwargs_of=lambda c: {}):
    def run(ctx, round_):
        fn = lazy(module, name)
        if not fn:
            return missing(stage)
        return as_verdict(call(fn, *args_of(ctx), **kwargs_of(ctx)), stage)
    return run


def trial_view(r):
    return {k: r.get(k) for k in ("role", "i", "model", "attack", "status", "reward", "calls", "seconds")}


def roles(role, stage, judge=None):
    def run(ctx, round_):
        run_role = lazy("agents", "run_role")
        if not run_role:
            return missing(stage)
        results = call(run_role, ctx.task, role, **ctx.agent_kwargs(role))
        if judge:
            check = lazy(*judge)
            if not check:
                return missing(stage)
            v = as_verdict(call(check, results, ctx.acceptance), stage)
            v.setdefault("details", {})["trials"] = [trial_view(r) for r in results or []]
            return v
        if isinstance(results, dict) and "verdict" in results:
            return results
        bad = [r for r in results or [] if r.get("status") not in ("completed", "submitted", "done")]
        passed = bool(results) and not bad
        details = {"trials": [trial_view(r) for r in results or []]}
        if role == "calibrator":
            details["_results"] = results  # run_calibrate takes it out before the verdict is recorded
        return {"verdict": "pass" if passed else "fail", "passed": passed,
                "findings": [f"{r.get('role')}-{r.get('i')} ended {r.get('status')}" for r in bad],
                "details": details}
    return run


def promote_calibration(ctx, results):
    """Spec tasks: the best valid calibrator submission becomes the target (via the gateway's /set_target) and
    solution/reference/ (patch + summary). Returns (verdict details, findings)."""
    valid = [r for r in results or [] if (r.get("final") or {}).get("valid") and (r.get("final") or {}).get("speedup")]
    if not valid:
        return {}, ["no valid calibrator submission: nothing to set as the target or reference"]
    best = max(valid, key=lambda r: r["final"]["speedup"])
    subs = sorted(Path(best["dir"]).glob("submit-*.diff")) if best.get("dir") else []
    if not subs:
        return {}, [f"best calibrator ({best.get('model')}) left no submitted patch in {best.get('dir')}"]
    patch = subs[-1].read_text()
    ref_dir = ctx.task_dir / "solution" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "patch.diff").write_text(patch)
    (ref_dir / "summary.md").write_text("# Reference solution\n\nThe best valid calibrator submission (%s, speedup %.3fx "
                                        "on %s).\n\n%s\n" % (best.get("model"), best["final"]["speedup"],
                                                               ctx.spec.get("device", "the phone"),
                                                               (best.get("final_text") or "")[:4000]))
    (ctx.work / "oracle.diff").write_text(patch)
    target = round(best["final"]["speedup"], 3)
    details = {"target_speedup": target, "source": "calibration", "model": best.get("model"),
               "patch_lines": patch.count("\n")}
    key_file = Path(os.environ.get("TF_GATEWAY_KEYS_DIR", str(DEFAULT_KEYS))) / "gateway-factory-key"
    if ctx.args.gateway and key_file.is_file():
        import urllib.request
        headers = {"X-TF-Factory": key_file.read_text().strip(), "Content-Type": "application/json"}
        if os.environ.get("TF_GATEWAY_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["TF_GATEWAY_TOKEN"]
        req = urllib.request.Request(ctx.args.gateway.rstrip("/") + "/set_target", headers=headers, data=json.dumps(
            {"task": ctx.task, "target_speedup": target, "source": "calibration"}).encode())
        try:
            details["set_target"] = json.load(urllib.request.urlopen(req, timeout=60)).get("result")
        except Exception as exc:
            return details, [f"could not set the target on the gateway: {exc}"]
    ctx.spec.update({"target_speedup": target, "target_source": "calibration"})
    (ctx.work / "spec.json").write_text(json.dumps(ctx.spec, indent=1))
    return details, []


def run_calibrate(ctx, round_):
    verdict = roles("calibrator", 12)(ctx, round_)
    if ctx.spec.get("kind") != "spec" or verdict.get("details", {}).get("not_implemented"):
        return verdict
    results = verdict.get("details", {}).pop("_results", None)
    details, findings = promote_calibration(ctx, results)
    verdict["details"]["promotion"] = details
    if findings:
        verdict.update({"verdict": "fail", "passed": False, "findings": verdict.get("findings", []) + findings})
    return verdict


def run_redteam(ctx, round_):
    """Red-team agents, then a replay of every cheat ever saved against the current grader (always; it's cheap)."""
    run_role = lazy("agents", "run_role")
    replay = lazy("agents", "replay_cheats")
    if not run_role:
        return missing(13)
    results = call(run_role, ctx.task, "redteam", **ctx.agent_kwargs("redteam"))
    broke = [r for r in results or [] if (r.get("reward") or 0) > 0]
    findings = [f"attack '{r.get('attack')}' scored {r.get('reward')}" for r in broke]
    details = {"attacks": [trial_view(r) for r in results or []]}
    if replay:
        rv = call(replay, ctx.task, gateway_url=ctx.args.gateway)
        details["replay"] = {"verdict": rv["verdict"], "cheats": rv["details"].get("cheats"),
                             "findings": rv["findings"]}
        if rv["details"].get("cheats"):  # "no saved cheats" is fine; a saved cheat that still scores is not
            findings += [f for f in rv["findings"] if "no saved cheats" not in f]
    passed = bool(results) and not findings
    return {"verdict": "pass" if passed else "fail", "passed": passed, "findings": findings, "details": details}


def fixer(*candidates):
    """The first available fixer among (module, name) pairs, called as fn(task, findings)."""
    def fix(ctx, findings, round_):
        for module, name in candidates:
            fn = lazy(module, name)
            if fn:
                fn(ctx.task, findings)
                return True
        return False
    return fix


RUNNERS = {
    2: run_propose,
    3: run_proposal_qa,
    4: run_review,
    5: run_implement,
    6: simple("lifecycle.review", "static_checks", 6, lambda c: (c.task_dir,)),
    7: simple("lifecycle.gates", "noop", 7, lambda c: (c.task,)),
    8: simple("lifecycle.gates", "baseline", 8, lambda c: (c.task,), lambda c: {"trials": c.trials()}),
    9: simple("lifecycle.review", "implementation_qa", 9, lambda c: (c.task_dir,),
              lambda c: {"lean": c.settings["lean_review"]}),
    10: simple("lifecycle.review", "checkin_review", 10, lambda c: (c.task_dir, c.history()),
               lambda c: {"lean": c.settings["lean_review"]}),
    11: roles("explorer", 11),
    12: run_calibrate,
    13: run_redteam,
    14: roles("rollout", 14, judge=("agents", "difficulty_check")),
    15: simple("lifecycle.gates", "reference", 15, lambda c: (c.task,), lambda c: {"trials": c.trials()}),
    16: simple("lifecycle.review", "final_signoff", 16, lambda c: (c.task_dir, c.history(), c.acceptance)),
}
TASK_FIXER = fixer(("cut", "fix"), ("lifecycle.review", "fix_from_findings"))
# Who fixes a failed or returned stage, and which stages run again after the fix.
FIXERS = {
    3: (fix_proposal, [3]),
    4: (fix_proposal, [3, 4]),
    6: (TASK_FIXER, [6]),
    9: (TASK_FIXER, [6, 9]),
    10: (TASK_FIXER, [6, 9, 10]),
    13: (fixer(("lifecycle.review", "fix_from_findings")), [13]),
}


def record(ctx, stage, verdict, round_, wall):
    verdict = {**verdict, "profile": ctx.profile, "wall_seconds": round(wall)}
    if ctx.profile != "full":
        verdict["counts_as_accepted"] = False
    entry = ledger.append(ctx.task, stage, STAGES[stage], verdict, round_)
    model = (verdict.get("agent") or {}).get("model") if isinstance(verdict.get("agent"), dict) else None
    log.info("stage %d %s (round %d, %s): %s in %ds%s%s", stage, STAGES[stage], round_, ctx.profile,
             verdict["verdict"], wall, f" [{model}]" if model else "",
             "" if verdict["passed"] else " | " + "; ".join(str(f) for f in verdict.get("findings", []))[:400])
    return entry


def run_stage(ctx, stage, round_):
    if ctx.args.skip_phone and stage in NEEDS_PHONE:
        return {"verdict": "skipped", "passed": False, "findings": ["skipped: --skip-phone"], "details": {"skipped": True}}
    if stage in NEEDS_PHONE and not ctx.args.gateway:
        # Phone stages only run against a gateway named explicitly; the local Razr gateways are retired.
        return {"verdict": "skipped", "passed": False,
                "findings": ["skipped: no --gateway given (phone stages need an explicit device gateway)"],
                "details": {"skipped": True}}
    try:
        return RUNNERS[stage](ctx, round_)
    except Exception as exc:  # a crash is a failed stage, recorded with the error
        log.exception("stage %d crashed", stage)
        return {"verdict": "fail", "passed": False, "findings": [f"stage crashed: {type(exc).__name__}: {exc}"[:500]],
                "details": {"crashed": True}}


def run(ctx):
    stage = ctx.args.from_stage
    rounds, last_findings, broken = {}, {}, []
    max_rounds, keep_going = ctx.settings["max_rounds"], ctx.settings["keep_going"]

    def give_up(why):
        """Stop the run, or under keep_going record the break and move to the next stage."""
        nonlocal stage
        broken.append(stage)
        if keep_going:
            log.info("%s; continuing (%s profile keeps going)", why, ctx.profile)
            stage += 1
            return False
        log.info("%s; stopping", why)
        return True

    while stage <= ctx.args.to_stage:
        round_ = rounds.get(stage, 0) + 1
        rounds[stage] = round_
        t0 = time.time()
        verdict = run_stage(ctx, stage, round_)
        record(ctx, stage, verdict, round_, time.time() - t0)
        if verdict["passed"] or verdict["verdict"] == "skipped":
            stage += 1
            continue
        details = verdict.get("details", {})
        if verdict["verdict"] == "reject":
            if give_up(f"task rejected at stage {stage}"):
                return False
            continue
        if details.get("not_implemented") or details.get("crashed"):
            if give_up(f"stage {stage} could not run"):
                return False
            continue
        findings = sorted(str(f) for f in verdict.get("findings", []))
        fix = FIXERS.get(stage)
        if not fix or round_ > max_rounds:
            if give_up(f"stage {stage} did not pass after {round_} round(s)"):
                return False
            continue
        if last_findings.get(stage) == findings:  # the last fix changed nothing the stage cares about
            if give_up(f"stage {stage} returned the same findings after a fix (no progress)"):
                return False
            continue
        last_findings[stage] = findings
        fix_fn, rerun = fix
        log.info("stage %d returned to its author with %d finding(s)", stage, len(findings))
        if not fix_fn(ctx, verdict.get("findings", []), round_ + 1):
            if give_up(f"no fixer available for stage {stage}"):
                return False
            continue
        stage = rerun[0]
        for s in rerun:
            rounds[s] = max(rounds.get(s, 0), round_)
    if broken:
        log.info("stages that did not pass: %s", ", ".join(f"{b} {STAGES[b]}" for b in broken))
    return not broken


def status(task=None, profile=None):
    tasks = [task] if task else sorted(p.stem for p in ledger.LEDGER.glob("*.jsonl"))
    for t in tasks:
        seen = ledger.latest(t, profile)
        print(f"\n{t}  (next stage: {ledger.current_stage(t, profile=profile)}{', profile ' + profile if profile else ''})")
        for stage, name in STAGES.items():
            e = seen.get(stage)
            if e is None:
                print(f"  {stage:>2} {name:<18} -")
                continue
            first = (e.get("findings") or [""])[0]
            model = (e.get("agent") or {}).get("model") if isinstance(e.get("agent"), dict) else ""
            print(f"  {stage:>2} {name:<18} {e['verdict']:<8} r{e['round']} {e.get('profile', 'full'):<5} "
                  f"{e.get('wall_seconds', e.get('seconds', '')):>5}s {model or '':<22} {str(first)[:70]}")


def new_spec_task(args):
    """Set up a task built from a spec: the starter is a given commit (default: the clone's HEAD); there is no
    reference commit and no oracle yet. Calibration (stage 12) supplies the target and solution/reference/."""
    starter = subprocess.run(["git", "-C", str(ADRENO_LLMS), "rev-parse", args.starter or "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    work = RUNS / "tasks" / args.task
    work.mkdir(parents=True, exist_ok=True)
    pack_file = ensure_pack(args.port, starter)
    spec = {"kind": "spec", "port": args.port, "base": starter, "reference": None, "title": args.title or args.task,
            "metric": args.metric, "correctness_rule": args.correctness, "device": args.device, "phone": args.device,
            "valid_token_sources": [v.strip() for v in args.valid_token_sources.split(",")],
            "target_source": "calibration", "notes": args.notes or ""}
    (work / "spec.json").write_text(json.dumps(spec, indent=1))
    print(f"{args.task}: spec task on {args.port} starting from {starter[:10]}; spec at {work / 'spec.json'}; "
          f"pack at {pack_file}")


def new_task(args):
    """Set up runs/tasks/<task>/ from a mined candidate: spec.json, the base-commit pack, the port-only oracle."""
    if args.spec:
        return new_spec_task(args)
    if not args.candidate:
        raise SystemExit("--candidate is required unless --spec is given")
    cand = find_candidate(args.port, args.candidate)
    reference = subprocess.run(["git", "-C", str(ADRENO_LLMS), "rev-parse", cand["commit"]], capture_output=True,
                               text=True, check=True).stdout.strip()
    base = args.base or parent_of(reference)
    base = subprocess.run(["git", "-C", str(ADRENO_LLMS), "rev-parse", base], capture_output=True, text=True,
                          check=True).stdout.strip()
    work = RUNS / "tasks" / args.task
    work.mkdir(parents=True, exist_ok=True)
    ensure_pack(args.port, base)
    files = cand.get("speedup_files") if args.oracle_scope == "speedup" else None
    oracle = port_oracle_diff(args.port, base, reference, files)
    if not oracle.strip():
        raise SystemExit(f"no change to src/models/{args.port} between {base[:10]} and {reference[:10]}")
    target = work / "oracle.diff"
    if target.exists() and target.read_text() != oracle and not args.replace_oracle:
        (work / "oracle.factory.diff").write_text(oracle)
        note = "kept the existing oracle.diff (differs from ours; ours is oracle.factory.diff)"
    else:
        target.write_text(oracle)
        note = "wrote oracle.diff"
    spec_path = work / "spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    spec.update({"port": args.port, "base": base, "reference": reference, "candidate": cand["commit"],
                 "oracle_scope": args.oracle_scope, "oracle_files": files or f"src/models/{args.port} (minus docs)",
                 "title": cand["title"], "claimed_speedup": cand["claimed_speedup"], "metric": cand["metric"],
                 "phone": "Motorola Razr 2020 (Snapdragon 765G, Adreno 620)"})
    spec_path.write_text(json.dumps(spec, indent=1))
    files = oracle.count("diff --git ")
    print(f"{args.task}: base {base[:10]} -> reference {reference[:10]}; {note} ({files} files, "
          f"{oracle.count(chr(10))} lines); spec at {spec_path}; pack at runs/mine/{args.port}/pack-{base[:10]}.txt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--port", required=True)
    r.add_argument("--candidate", help="mined commit (prefix ok); not needed for spec tasks")
    r.add_argument("--from", dest="from_stage", type=int, default=2)
    r.add_argument("--to", dest="to_stage", type=int, default=16)
    r.add_argument("--profile", choices=sorted(PROFILES), default="full")
    r.add_argument("--skip-phone", action="store_true")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--proposal", help="start from this proposal file (proposal.json format) instead of the task's own")
    r.add_argument("--gateway", default=os.environ.get("TF_GATEWAY_URL"),
                   help="device gateway URL for every phone stage (or TF_GATEWAY_URL); phone stages are skipped without one")
    r.add_argument("--gateway-keys", default=os.environ.get("TF_GATEWAY_KEYS_DIR", str(DEFAULT_KEYS)),
                   help="directory with gateway-hmac, gateway-verifier-key, gateway-factory-key, optional gateway-token")
    n = sub.add_parser("new-task", help="set up a task from a mined candidate")
    n.add_argument("--task", required=True)
    n.add_argument("--port", required=True)
    n.add_argument("--candidate", help="mined commit (tasks built from a speedup commit)")
    n.add_argument("--spec", action="store_true", help="a task built from a spec: no reference commit, calibration sets it")
    n.add_argument("--starter", help="spec tasks: starter commit (default: HEAD of src/adreno-llms)")
    n.add_argument("--title")
    n.add_argument("--metric", help="spec tasks: the metric and how it is measured")
    n.add_argument("--correctness", help="spec tasks: the correctness rule")
    n.add_argument("--device", help="spec tasks: the phone the task is graded on")
    n.add_argument("--valid-token-sources", default="base", help="spec tasks: comma list (default: base)")
    n.add_argument("--notes", help="spec tasks: anything the proposer and the gateway need to know")
    n.add_argument("--base", help="base commit (default: the candidate's parent)")
    n.add_argument("--oracle-scope", choices=["speedup", "port"], default="speedup",
                   help="speedup: only the mined speedup files (default); port: the whole port minus docs")
    n.add_argument("--replace-oracle", action="store_true", help="overwrite an existing oracle.diff")
    s = sub.add_parser("status")
    s.add_argument("--task")
    s.add_argument("--profile", choices=sorted(PROFILES))
    q = sub.add_parser("qa-test", help="plant one flaw per copy of the approved proposal; QA or review must catch each")
    q.add_argument("--task", required=True)
    args = ap.parse_args()
    if args.cmd == "status":
        return status(args.task, args.profile)
    if args.cmd == "new-task":
        return new_task(args)
    if args.cmd == "qa-test":
        from lifecycle.proposal import mutation_test
        report = mutation_test(args.task)
        print(f"catch rate {report['catch_rate']} ({report['seconds']}s, tokens in {report['tokens']['input']} "
              f"out {report['tokens']['output']})")
        c = report["control"]
        print(f"  control (unmutated): QA={c['qa_verdict']} review={c['review_verdict']} clean={c['clean']} "
              f"{(c['qa_findings'] + c['review_blocking'])[:2]}")
        for row in report["flaws"]:
            print(f"  {row['flaw']:<24} caught={row['caught']!s:<5} by={','.join(row['caught_by']) or '-':<10} "
                  f"QA={row['qa_verdict']:<5} review={row['review_verdict']:<8} {row['evidence'][:110]}")
        return
    model, patched = apply_profile(args.profile)
    if args.gateway:
        present = apply_gateway(args.gateway, args.gateway_keys)
        log.info("phone stages use gateway %s (key files present: %s)", args.gateway, ", ".join(present) or "none")
    ctx = Context(args)
    log.info("run %s (%s, %s %s): stages %d-%d, profile %s (authoring model %s in %s)%s", args.task, args.port,
             "spec starter" if ctx.spec.get("kind") == "spec" else "candidate", ctx.candidate["commit"][:10], args.from_stage, args.to_stage, args.profile, model, ", ".join(patched),
             " skipping phone stages" if args.skip_phone else "")
    t0 = time.time()
    ok = run(ctx)
    log.info("run %s finished in %ds: %s", args.task, time.time() - t0, "all stages passed" if ok else "stopped")
    status(args.task, args.profile)


if __name__ == "__main__":
    main()
