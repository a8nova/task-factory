"""Cut: Argon turns one mined commit into a task statement, scope rules and a red-team attack list; the factory
wraps them in the fixed, locked Harbor harness (harness/).

    uv run python pack.py smollm2-135m-instruct --commit <base>
    uv run python cut.py smollm2-image2d-decode --sync     # pull public calibration data from the gateway first
    uv run python cut.py smollm2-image2d-decode            # calls Argon, then assembles
    uv run python cut.py smollm2-rope-smoke --model gemini-3.8-flash   # a faster model for smoke tasks
    uv run python cut.py smollm2-image2d-decode --reuse    # assembles from the saved cut.json

Reads runs/tasks/<task>/spec.json (port, base, reference, numbers measured on the phone, development prompts
and their expected tokens). Writes the Harbor task to tasks/<task>/ and Argon's raw answer to
runs/tasks/<task>/cut.json. Progress: tail -f logs/cut.log
"""
import argparse
import json
import urllib.request
import os
import shutil
import subprocess
import time
from pathlib import Path

from common import ARGON, ROOT, RUNS, gemini_client, get_logger

log = get_logger("cut")
HARNESS = ROOT / "harness"

PROMPT = """You are turning a real optimization commit into an RL training task for coding agents.

The agent starts from the BASE commit of one adreno-llms model port (an OpenCL LLM engine for Adreno phone
GPUs). Its job is to make fp16 greedy decode faster on a real phone (Motorola Razr 2020, Adreno 620) while
producing exactly the reference implementation's output tokens. The merged commit below is the reference
solution; the agent never sees it.

How the task works (fixed; describe it accurately, do not change it):
- Workspace: the repository at the base commit is in /workspace/repo (a git checkout). The agent cannot
  compile or run anything locally and has no network access except the device gateway.
- Deliverable: /workspace/submission/patch.diff (the output of `git add -N . && git diff --binary {base}`
  run in /workspace/repo) and /workspace/submission/summary.md (what was changed and why). Only this folder
  reaches the grader.
- Validation: /workspace/validation/val.sh checks /workspace/submission/patch.diff (or `val.sh --repo` checks
  the current changes in /workspace/repo). The gateway applies the patch to the base commit, builds it with
  the Android NDK, runs it on the phone on {n_dev} development prompts at generation lengths {lengths}, and
  reports per prompt whether every run produced exactly the expected tokens, the decode speed, and the score.
  The development prompts and the reference's tokens for them are in /workspace/task/expected_tokens.json
  (shorter runs must equal the first N of those tokens). One call takes about {try_minutes} minutes; the phone
  runs one build at a time, so validate deliberately.
- Time: /workspace/timer.sh prints the time left in the budget of {timeout_min} minutes. The agent should
  leave time to write and validate the final submission.
- Grading: the same build and runs on hidden prompts of the same kind, which the agent never sees.
  1. Valid only if, on every hidden prompt and at every generation length (around {lengths}, each raised by
     a random 0-{jitter} tokens per grading call), the greedy output tokens exactly equal the tokens of one
     trusted implementation: {sources}. The binary must exit cleanly, the patch may only change the allowed
     files, the time-vs-length points must lie on a straight line (R^2 >= 0.98), the shortest run must not be
     anomalously slow compared with the base, and the binary's own printed speed must not exceed the
     measured speed by more than 25%. The unchanged base is valid but not faster, so it scores 0.
  2. Speed: decode tokens/s is the slope of a line fitted through (generated tokens, seconds) for those runs,
     timed by the phone's own clock from outside the binary. Speedup = candidate / base. Base is
     {base_tps:.2f} tok/s and the reference reaches {ref_tps:.2f} tok/s ({target:.3f}x).
  3. Reward = 0 if not valid; otherwise min(1, max(0, (speedup - 1) / ({target:.3f} - 1))). So 1.25x gives
     {r125:.2f}, 1.40x gives {r140:.2f}, and {target:.3f}x or faster gives 1.0.
- The binary's own printed timings are not used for the score. Only fp16 weights exist on the phone.

Because validity is exact token equality with the reference, the agent must reproduce the reference's
numerics (reduction order, accumulation precision, fused operations, build options) closely enough that
greedy decoding yields the same tokens over 136 steps. The instruction therefore has to specify the target
design precisely enough to make the task solvable, like a senior engineer's design ticket: what to
restructure, which data layouts and kernel shapes, what must be fused, how reductions are ordered, which
compiler options, and what the host code must stop doing per token. Work these out from the reference diff.
Describe the design; do not paste reference code.

Write instruction_md with exactly these sections, in this order:
  # <title>
  ## Task          what to build, the deliverable files and where they go, how to produce patch.diff
  ## Constraints   files in scope, files that must not change, fixed assets, no network, fp16 only
  ## Success Criteria   validity rules, how speed is measured, the reward formula with the numbers above
  ## Design specification   the target design in enough detail to reproduce the reference's numerics
  ## Validation and time   how to use val.sh (both forms) and timer.sh, and a time-budget notice
Never mention hidden prompts' content, never include reference code.

Also write:
- in_scope_files and off_limits_files: path prefixes relative to the repo root. Off-limits must cover
  everything that affects measurement or correctness checks but is not part of the optimization (timing
  code, benchmark/run scripts, reference fixtures, golden files, weights, build scripts, other models).
- red_team_attacks: concrete ways an agent could get reward without doing the real work on THIS task, given
  the grading above, each with what stops it, or "NOT STOPPED" if nothing does.
- difficulty, agent_timeout_sec (at most 21600), tags, a short readme_md for people browsing the task, and
  reference_summary_md: a summary.md describing the reference solution's changes (for solution/reference).

Candidate notes from the mining step:
{candidate}

Reference solution (the merged commit, limited to this port; the agent never sees this):
{oracle}

Base commit code and history of the port:
{pack}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "instruction_md": {"type": "string"},
        "readme_md": {"type": "string"},
        "reference_summary_md": {"type": "string"},
        "in_scope_files": {"type": "array", "items": {"type": "string"}},
        "off_limits_files": {"type": "array", "items": {"type": "string"}},
        "red_team_attacks": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "how": {"type": "string"}, "what_stops_it": {"type": "string"}},
            "required": ["name", "how", "what_stops_it"]}},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "agent_timeout_sec": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["instruction_md", "readme_md", "reference_summary_md", "in_scope_files", "off_limits_files",
                 "red_team_attacks", "difficulty", "agent_timeout_sec", "tags"],
}


def reward_at(speedup, target):
    return max(0.0, min(1.0, (speedup - 1.0) / (target - 1.0)))


def assemble(task, spec, cut):
    """Fill the fixed harness with Argon's statement. Security-relevant files come only from harness/."""
    out = ROOT / "tasks" / task
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(HARNESS, out)
    (out / "instruction.md").write_text(cut["instruction_md"])
    (out / "README.md").write_text(cut["readme_md"])
    shutil.copy(RUNS / "tasks" / task / "oracle.diff", out / "solution" / "reference" / "patch.diff")
    (out / "solution" / "reference" / "summary.md").write_text(cut["reference_summary_md"])
    (out / "environment" / "workspace" / "task" / "expected_tokens.json").write_text(
        json.dumps(spec["dev_expected"], indent=1))
    timeout = min(int(cut["agent_timeout_sec"]), 21600)
    values = {"TASK": task, "BASE": spec["base"], "DIFFICULTY": cut["difficulty"], "TAGS": json.dumps(cut["tags"]),
              "AGENT_TIMEOUT": str(timeout), "SOURCE_COMMIT": spec["reference"], "PORT": spec["port"]}
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".toml", ".sh", ".py", ""} | {".md"}:
            text = path.read_text()
            for key, value in values.items():
                text = text.replace("{{%s}}" % key, value)
            path.write_text(text)
    leftover = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and "{{" in p.read_text(errors="ignore")]
    if leftover:
        raise SystemExit("unfilled placeholders in %s" % leftover)
    return out


SYNC = r"""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/task-factory/v2"))
os.environ["TF_STATE_DIR"] = os.path.expanduser("~/task-factory/v2")
import gateway
task, base, ref = gateway.calibrate(sys.argv[1])
hid = task["hidden_prompts"]
med = gateway.median
print(json.dumps({
    "port": task["port"], "base": task["base"], "reference": task["reference"],
    "target_speedup": task["target_speedup"], "valid_token_sources": task.get("valid_token_sources"),
    "grade_lengths": task.get("grade_lengths", list(gateway.GRADE_LENGTHS)), "jitter": task.get("jitter", 12),
    "base_tps": round(med([gateway.decode_of(base["prompts"][p])[0] for p in hid]), 3),
    "reference_tps": round(med([gateway.decode_of(ref["prompts"][p])[0] for p in hid]), 3),
    "dev_expected": [{"prompt": p, "reference_tokens": ref["prompts"][p]["tokens"]} for p in task["dev_prompts"]],
    "weaker_check": task.get("prompt_selection", {}).get("weaker_check")}))
"""


def sync_from_gateway(task, work, timeout_min):
    """Pull the public part of the gateway's calibration; hidden prompts never leave the gateway. With
    TF_GATEWAY_URL set this asks the cloud gateway's /public endpoint (factory key); otherwise the Mac mini."""
    url = os.environ.get("TF_GATEWAY_URL")
    if url:
        key = (Path.home() / ".config" / "task-factory" / "gateway-factory-key").read_text().strip()
        req = urllib.request.Request(url.rstrip("/") + "/public", data=json.dumps({"task": task}).encode(),
                                     headers={"X-TF-Factory": key, "Content-Type": "application/json"})
        spec = json.load(urllib.request.urlopen(req, timeout=3600))["result"]
    else:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "macmini", "python3", "-", task], input=SYNC,
                             capture_output=True, text=True, check=True).stdout
        spec = json.loads(out.strip().splitlines()[-1])
    spec["timeout_min"] = timeout_min
    work.mkdir(parents=True, exist_ok=True)
    (work / "spec.json").write_text(json.dumps(spec, indent=1))
    log.info("synced %s: base %s, reference %s tok/s, target %s, %d dev prompts", task, spec["base_tps"],
             spec["reference_tps"], spec["target_speedup"], len(spec["dev_expected"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--reuse", action="store_true", help="assemble from the saved cut.json without calling Argon")
    ap.add_argument("--model", default=ARGON, help="authoring model (smoke tasks can use gemini-3.8-flash)")
    ap.add_argument("--sync", action="store_true", help="refresh spec.json from the gateway's calibration first")
    ap.add_argument("--timeout-min", type=int, default=120)
    args = ap.parse_args()
    work = RUNS / "tasks" / args.task
    if args.sync:
        sync_from_gateway(args.task, work, args.timeout_min)
    spec = json.loads((work / "spec.json").read_text())
    if args.reuse:
        cut = json.loads((work / "cut.json").read_text())
    else:
        pack = (RUNS / "mine" / spec["port"] / f"pack-{spec['base'][:10]}.txt").read_text()
        cand = next(c for c in json.loads((RUNS / "mine" / spec["port"] / "candidates.json").read_text())["candidates"]
                    if spec["reference"].startswith(c["commit"][:8]) or c["commit"].startswith(spec["reference"][:8]))
        target = spec["target_speedup"]
        prompt = PROMPT.format(base=spec["base"], n_dev=len(spec["dev_expected"]), timeout_min=spec["timeout_min"],
                               lengths="/".join(str(n) for n in spec["grade_lengths"]), jitter=spec["jitter"],
                               sources=" or ".join("the " + x for x in spec["valid_token_sources"]),
                               try_minutes=spec.get("try_minutes", "1-3"),
                               base_tps=spec["base_tps"], ref_tps=spec["reference_tps"], target=target,
                               r125=reward_at(1.25, target), r140=reward_at(1.40, target),
                               candidate=json.dumps(cand, indent=1), oracle=(work / "oracle.diff").read_text(),
                               pack=pack)
        client = gemini_client()
        t0 = time.time()
        it = client.interactions.create(
            model=args.model, input=prompt, background=True,
            generation_config={"thinking_level": "high", "max_output_tokens": 131072},
            response_format={"type": "text", "mime_type": "application/json", "schema": SCHEMA})
        log.info("started %s on %s (%d chars): interaction %s", args.model, args.task, len(prompt), it.id)
        while it.status in ("in_progress", "queued"):
            time.sleep(15)
            it = client.interactions.get(id=it.id)
            log.info("  %s after %.0fs", it.status, time.time() - t0)
        if it.status != "completed":
            raise SystemExit(f"{it.status}: {getattr(it, 'error', '')}")
        cut = json.loads(it.output_text)
        cut["_usage"] = {"model": args.model, "input": it.usage.total_input_tokens, "output": it.usage.total_output_tokens,
                         "seconds": round(time.time() - t0), "interaction": it.id}
        (work / "cut.json").write_text(json.dumps(cut, indent=1))
        log.info("done in %.0fs: tokens in %s out %s; %d attacks, %d off-limits paths, difficulty %s",
                 time.time() - t0, it.usage.total_input_tokens, it.usage.total_output_tokens,
                 len(cut["red_team_attacks"]), len(cut["off_limits_files"]), cut["difficulty"])
    out = assemble(args.task, spec, cut)
    log.info("Harbor task written to %s", out)
    for a in cut["red_team_attacks"]:
        log.info("  attack: %s | stopped by: %s", a["name"][:70], a["what_stops_it"][:90])


if __name__ == "__main__":
    main()
