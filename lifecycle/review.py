"""Review stages of the task factory (CONTRACT.md stages 6, 9, 10, 13-fix and 16).

    uv run python -m lifecycle.review static smollm2-image2d-decode
    uv run python -m lifecycle.review qa smollm2-image2d-decode
    uv run python -m lifecycle.review checkin smollm2-image2d-decode
    uv run python -m lifecycle.review signoff smollm2-image2d-decode [--no-summary]
    uv run python -m lifecycle.review fix smollm2-image2d-decode findings.json [--dry-run]

static_checks is deterministic. The reviewers are Argon in FRESH interactions (no previous_interaction_id),
so a review never sees the author's reasoning. Every function returns the verdict shape in CONTRACT.md.
Progress: tail -f logs/review.log
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import tomllib
from pathlib import Path

from common import ARGON, ROOT, RUNS, gemini_client, get_logger

log = get_logger("review")
KEYS_DIR = Path.home() / ".config" / "task-factory"
ACCEPTANCE = ROOT / "private" / "acceptance.toml"

REQUIRED = [
    "instruction.md", "task.toml", "README.md",
    "environment/Dockerfile",
    "environment/workspace/validation/val.sh",
    "environment/workspace/timer.sh",
    "environment/workspace/task/expected_tokens.json",
    "solution/baseline.sh", "solution/solve.sh",
    "solution/reference/patch.diff", "solution/reference/summary.md", "solution/reference.sh",
    "tests/Dockerfile", "tests/test.sh", "tests/grade_client.py",
]
EXECUTABLE = ["environment/workspace/validation/val.sh", "environment/workspace/timer.sh", "solution/baseline.sh",
              "solution/solve.sh", "solution/reference.sh", "tests/test.sh"]
TEXT_SUFFIXES = {".md", ".toml", ".sh", ".py", ".json", ".txt", ".diff", ""}
# Ledger stage numbers from CONTRACT.md, plus the names other parts may log them under.
STAGES = {6: "static_checks", 7: "noop", 8: "baseline", 9: "implementation_qa", 10: "checkin_review",
          11: "explore", 12: "calibrate", 13: "redteam", 14: "rollouts", 15: "reference", 16: "final_signoff"}
ALIASES = {"static": 6, "staticchecks": 6, "noop": 7, "nooprun": 7, "baseline": 8, "qa": 9,
           "implementationqa": 9, "checkin": 10, "checkinreview": 10, "explore": 11, "explorer": 11,
           "calibrate": 12, "calibration": 12, "calibrator": 12, "redteam": 13, "redteamfix": 13,
           "rollout": 14, "rollouts": 14, "reference": 15, "finalsignoff": 16, "signoff": 16}


# ---------------------------------------------------------------- shared helpers

def verdict(stage, decision, passed, findings, details, t0, agent=None):
    return {"stage": stage, "verdict": decision, "passed": passed, "findings": findings, "details": details,
            "seconds": round(time.time() - t0, 1), "agent": agent or {}}


def task_name(task_dir):
    return Path(task_dir).name


def hidden_spec(task):
    """Read-only copy of the hidden spec from the Mac mini (None if unreachable). Never logged."""
    # Gateway v2 keeps its state in ~/task-factory/v2 (TF_STATE_DIR); v1 used ~/task-factory directly.
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "macmini",
                           "cat ~/task-factory/v2/tasks/%s.json 2>/dev/null || cat ~/task-factory/tasks/%s.json"
                           % (task, task)], capture_output=True, text=True, timeout=30)
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except ValueError:
        return None


def acceptance():
    return tomllib.loads(ACCEPTANCE.read_text()) if ACCEPTANCE.exists() else {}


def package_files(task_dir):
    """Every file of the package as (relative path, text or None for binary)."""
    out = []
    for p in sorted(Path(task_dir).rglob("*")):
        if not p.is_file():
            continue
        data = p.read_bytes()
        text = None if b"\0" in data[:4096] else data.decode("utf-8", "replace")
        out.append((str(p.relative_to(task_dir)), text))
    return out


def find_line(text, needle):
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return None


def secrets_on_disk():
    """Contents of the factory's key files, used only to search for leaks; never printed or logged."""
    out = {}
    for p in KEYS_DIR.glob("*") if KEYS_DIR.exists() else []:
        value = p.read_text().strip()
        if len(value) >= 16:
            out[p.name] = value
    return out


def argon_json(prompt, schema, tag, thinking="high", max_output=65536):
    """One FRESH Argon interaction (no previous_interaction_id) returning parsed JSON plus usage."""
    client = gemini_client()
    t0 = time.time()
    it = client.interactions.create(
        model=ARGON, input=prompt, background=True,
        generation_config={"thinking_level": thinking, "max_output_tokens": max_output},
        response_format={"type": "text", "mime_type": "application/json", "schema": schema})
    log.info("%s: started %s (%d chars) %s", tag, ARGON, len(prompt), it.id)
    while it.status in ("in_progress", "queued"):
        time.sleep(10)
        it = client.interactions.get(id=it.id)
    if it.status != "completed":
        raise RuntimeError("%s: interaction %s ended %s: %s" % (tag, it.id, it.status, getattr(it, "error", "")))
    usage = {"model": ARGON, "interaction": it.id, "input_tokens": it.usage.total_input_tokens,
             "output_tokens": it.usage.total_output_tokens, "seconds": round(time.time() - t0)}
    log.info("%s: done in %ds, tokens in %s out %s", tag, usage["seconds"], usage["input_tokens"],
             usage["output_tokens"])
    return json.loads(it.output_text), usage


def package_text(task_dir, skip=()):
    parts = []
    for rel, text in package_files(task_dir):
        if rel in skip:
            continue
        parts.append("=== FILE %s ===\n%s" % (rel, text if text is not None else "(binary file)"))
    return "\n\n".join(parts)


def stage_of(entry):
    """Stage number of a ledger entry, whether it was logged by number or by name."""
    for key in ("stage", "stage_name"):
        value = entry.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            norm = re.sub(r"[^a-z]", "", value.lower())
            if norm in ALIASES:
                return ALIASES[norm]
    return None


def latest_by_stage(ledger_history):
    out = {}
    for entry in ledger_history or []:
        n = stage_of(entry)
        if n is not None:
            out[n] = entry
    return out


# ---------------------------------------------------------------- stage 6: static checks

def static_checks(task_dir):
    t0 = time.time()
    task_dir = Path(task_dir)
    task = task_name(task_dir)
    findings = []

    def add(check, problem, file=None, line=None, severity="error"):
        findings.append({"check": check, "severity": severity, "file": file, "line": line, "problem": problem})

    for rel in REQUIRED:
        if not (task_dir / rel).is_file():
            add("required_file", "missing required file", rel,
                severity="warning" if rel == "README.md" else "error")
    for rel in EXECUTABLE:
        p = task_dir / rel
        if p.is_file() and not os.access(p, os.X_OK):
            add("executable", "script is not executable (chmod +x)", rel)

    files = package_files(task_dir)
    for rel, text in files:
        if text and "{{" in text and "}}" in text:
            add("placeholder", "unfilled template placeholder", rel, find_line(text, "{{"))

    # task.toml
    toml_path = task_dir / "task.toml"
    conf = {}
    if toml_path.is_file():
        toml_text = toml_path.read_text()
        try:
            conf = tomllib.loads(toml_text)
        except tomllib.TOMLDecodeError as exc:
            add("task_toml", "task.toml does not parse: %s" % exc, "task.toml")
        if conf:
            if conf.get("schema_version") != "1.4":
                add("task_toml", "schema_version is %r, expected \"1.4\"" % conf.get("schema_version"), "task.toml",
                    find_line(toml_text, "schema_version"))
            arts = [a if isinstance(a, str) else a.get("source") for a in conf.get("artifacts", [])]
            if arts != ["/workspace/submission"]:
                add("task_toml", "artifacts is %r, expected [\"/workspace/submission\"] so only the submission "
                    "reaches the verifier" % arts, "task.toml", find_line(toml_text, "artifacts"))
            ver = conf.get("verifier", {})
            if ver.get("environment_mode") != "separate":
                add("task_toml", "verifier.environment_mode is %r, expected \"separate\""
                    % ver.get("environment_mode"), "task.toml", find_line(toml_text, "environment_mode"))
            limits = acceptance().get("limits", {})
            agent_t = conf.get("agent", {}).get("timeout_sec")
            if agent_t is None:
                add("task_toml", "agent.timeout_sec not set", "task.toml")
            elif limits.get("agent_timeout_sec_max") and agent_t > limits["agent_timeout_sec_max"]:
                add("task_toml", "agent.timeout_sec %s exceeds the accepted maximum %s"
                    % (agent_t, limits["agent_timeout_sec_max"]), "task.toml", find_line(toml_text, "[agent]"))
            ver_t = ver.get("timeout_sec")
            if ver_t is None:
                add("task_toml", "verifier.timeout_sec not set", "task.toml")
            elif limits.get("verifier_timeout_sec_max") and ver_t > limits["verifier_timeout_sec_max"]:
                add("task_toml", "verifier.timeout_sec %s exceeds the accepted maximum %s"
                    % (ver_t, limits["verifier_timeout_sec_max"]), "task.toml", find_line(toml_text, "[verifier]"))
            for section in ("environment", "verifier", "agent"):
                if conf.get(section, {}).get("network_mode") == "public":
                    add("task_toml", "%s.network_mode is \"public\"; the shipped task must not have open internet"
                        % section, "task.toml", find_line(toml_text, "network_mode"))
            if conf.get("environment", {}).get("network_mode") is None:
                add("task_toml", "environment.network_mode not set (Harbor defaults to public)", "task.toml")

    # instruction.md
    inst_path = task_dir / "instruction.md"
    if inst_path.is_file():
        inst = inst_path.read_text()
        heads = [h.strip().lower() for h in re.findall(r"^#{1,4}\s*(.+?)\s*$", inst, re.M)]
        for section in ("task", "constraints", "success criteria"):
            if not any(h == section or h.startswith(section + " ") or h.startswith(section + ":")
                       or h.rstrip(":") == section for h in heads):
                add("instruction", "missing a \"%s\" section heading" % section.title(), "instruction.md")
        if not re.search(r"time[- ]budget|timer\.sh|time limit|remaining time|time remaining", inst, re.I):
            add("instruction", "no time-budget notice (mention the budget and /workspace/timer.sh)", "instruction.md")
        if "val.sh" not in inst:
            add("instruction", "does not document /workspace/validation/val.sh as the validation entrypoint",
                "instruction.md")
        if "/workspace/submission" not in inst:
            add("instruction", "does not say what goes in /workspace/submission/", "instruction.md")

    # leaks into the agent's environment
    spec = hidden_spec(task)
    if spec is None:
        add("hidden_spec", "could not read the hidden spec from the Mac mini; leak checks for hidden prompts "
            "were skipped", severity="warning")
    env_files = [(rel, text) for rel, text in files if rel.startswith("environment/")]
    hidden = (spec or {}).get("hidden_prompts", [])
    for prompt in hidden:
        for rel, text in env_files:
            if text and prompt in text:
                add("leak_hidden_prompt", "a hidden prompt appears in the agent's environment", rel,
                    find_line(text, prompt))
    for key in ("reference",):
        sha = (spec or {}).get(key) or conf.get("metadata", {}).get("source_commit")
        if sha:
            for rel, text in env_files:
                if text and sha[:10] in text:
                    add("leak_reference_commit", "the reference commit id appears in the agent's environment "
                        "(it could be fetched or looked up)", rel, find_line(text, sha[:10]))
    ref_patch = task_dir / "solution" / "reference" / "patch.diff"
    if not ref_patch.is_file():
        ref_patch = task_dir / "solution" / "oracle.diff"  # v1 layout
    if ref_patch.is_file():
        added = {l[1:].strip() for l in ref_patch.read_text().splitlines()
                 if l.startswith("+") and not l.startswith("+++") and len(l[1:].strip()) >= 40}
        for rel, text in env_files:
            if not text:
                continue
            hits = [l for l in added if l in text]
            if hits:
                add("leak_reference_patch", "%d lines of the reference solution appear in the agent's environment"
                    % len(hits), rel, find_line(text, hits[0]))
        if ref_patch.name == "patch.diff" and not ref_patch.read_text().lstrip().startswith("diff --git"):
            add("reference", "solution/reference/patch.diff is empty or not a git diff", str(ref_patch.relative_to(task_dir)))
        if spec and spec.get("off_limits"):
            touched = sorted({l.split(" b/", 1)[1] for l in ref_patch.read_text().splitlines()
                              if l.startswith("diff --git a/")})
            bad = [f for f in touched if any(f == p or f.startswith(p.rstrip("/") + "/") for p in spec["off_limits"])]
            if bad:
                add("reference", "the reference solution edits off-limits files, so it can't pass: %s" % bad[:5],
                    str(ref_patch.relative_to(task_dir)))
    sol_hashes = {hashlib.sha256((task_dir / rel).read_bytes()).hexdigest(): rel
                  for rel, _ in files if rel.startswith("solution/")}
    for rel, _ in env_files:
        h = hashlib.sha256((task_dir / rel).read_bytes()).hexdigest()
        if h in sol_hashes:
            add("leak_solution_file", "identical to %s" % sol_hashes[h], rel)
    dockerfile = task_dir / "environment" / "Dockerfile"
    if dockerfile.is_file():
        dtext = dockerfile.read_text()
        if re.search(r"(COPY|ADD)\s+[^\n]*(\.\./|solution|tests)", dtext):
            add("leak_dockerfile", "environment Dockerfile copies solution or test files", "environment/Dockerfile",
                find_line(dtext, "solution") or find_line(dtext, "tests"))
    keys = secrets_on_disk()
    for rel, text in files:
        if not text:
            continue
        for name, value in keys.items():
            if value in text:
                add("leak_secret", "contains the value of %s (rotate it)" % name, rel, find_line(text, value))
        if re.search(r"AIza[0-9A-Za-z_\-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----", text):
            add("leak_secret", "looks like an API key or private key", rel)

    # verifier image and solution
    tdock = task_dir / "tests" / "Dockerfile"
    if tdock.is_file() and not re.search(r"^\s*COPY\s+.+\s+/tests/?\s*$", tdock.read_text(), re.M):
        add("verifier", "tests/Dockerfile must COPY the tests into /tests (Harbor doesn't upload them into a "
            "dedicated verifier image)", "tests/Dockerfile")
    tsh = task_dir / "tests" / "test.sh"
    if tsh.is_file() and "/workspace/submission" not in tsh.read_text():
        add("verifier", "tests/test.sh doesn't read /workspace/submission", "tests/test.sh")
    exp = task_dir / "environment" / "workspace" / "task" / "expected_tokens.json"
    if not exp.is_file():
        exp = task_dir / "environment" / "task" / "expected_tokens.json"  # v1 layout
    if exp.is_file():
        try:
            data = json.loads(exp.read_text())
            def tokens_of(v):
                if isinstance(v, list):
                    return v
                return next((v[k] for k in ("reference_tokens", "tokens", "expected") if isinstance(v.get(k), list)), [])
            if isinstance(data, list) and data and all(isinstance(t, int) for t in data):
                flat = data  # v1: one flat list
            elif isinstance(data, list):
                flat = [t for item in data for t in tokens_of(item)]  # v2: [{prompt, reference_tokens}]
            else:
                flat = [t for v in data.values() for t in tokens_of(v)]
            if not flat or not all(isinstance(t, int) for t in flat):
                add("expected_tokens", "expected_tokens.json holds no integer token lists",
                    str(exp.relative_to(task_dir)))
        except (ValueError, AttributeError):
            add("expected_tokens", "expected_tokens.json is not valid JSON in the expected shape",
                str(exp.relative_to(task_dir)))
    if spec is not None:
        for field in ("dev_prompts", "hidden_prompts", "target_speedup", "off_limits"):
            if not spec.get(field):
                add("hidden_spec", "hidden spec has no %s" % field, "~/task-factory/tasks/%s.json (Mac mini)" % task,
                    severity="error" if field in ("hidden_prompts", "off_limits") else "warning")

    errors = [f for f in findings if f["severity"] == "error"]
    passed = not errors
    log.info("static_checks %s: %s, %d errors, %d warnings", task, "pass" if passed else "fail", len(errors),
             len(findings) - len(errors))
    return verdict("static_checks", "pass" if passed else "fail", passed, findings,
                   {"files_checked": len(files), "errors": len(errors), "warnings": len(findings) - len(errors)}, t0)


# ---------------------------------------------------------------- stage 9: implementation QA

QA_CHECKLIST = """Review this RL task package before anyone spends agent rollouts on it. Check:
1. The instruction, the agent's validation script and the hidden verifier agree: same deliverable files and
   paths, same correctness rule, same metric and the same way reward is computed from it.
2. The starter (base) code runs exactly as the instruction describes, and the unchanged starter submitted
   as-is is valid but scores about 0.
3. Every path, command, file name and number the instruction mentions exists and is right.
4. Validation shows the agent nothing hidden: no hidden prompts, hidden expected outputs, reference code,
   or reference results beyond what the instruction intentionally gives.
5. The verifier only uses what is in /workspace/submission/, runs untrusted code without the power to
   change trusted evaluation code or scores, and tells infrastructure failures (phone, adb, network,
   gateway) apart from submission failures: an infrastructure failure writes no reward at all.
6. Validity, raw metrics and reward are reported separately; an invalid submission can never earn reward;
   a valid slow one keeps a low but valid score.
7. The reference solution really meets every rule and would score 1.
8. The task is solvable by a strong agent in the time budget with only what it is given, and the
   instruction specifies enough for the correctness rule to be reachable.
9. No secrets, no leaks of the solution through file contents, git history, commit ids, or network access.
10. Anything else that would make rollouts waste money or reward the wrong behaviour.
For every problem give severity (critical: rollouts would be wasted or the reward is wrong; major: likely
to mislead agents or reviewers; minor; nit), the file, the location, the problem and a concrete fix.
Report only real problems you can point to in the files below."""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "solvable": {"type": "boolean"},
        "solvability_notes": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["critical", "major", "minor", "nit"]},
            "checklist_item": {"type": "integer"},
            "file": {"type": "string"}, "location": {"type": "string"},
            "problem": {"type": "string"}, "suggested_fix": {"type": "string"}},
            "required": ["severity", "checklist_item", "file", "location", "problem", "suggested_fix"]}},
    },
    "required": ["summary", "solvable", "solvability_notes", "findings"],
}


def trusted_side_text(task):
    """The gateway code and the hidden spec, so the reviewer can judge the verifier that grades on the phone."""
    parts = []
    for rel in ("CONTRACT.md", "device/gateway.py", "device/devrun.py"):
        p = ROOT / rel
        if p.is_file():
            parts.append("=== TRUSTED FILE %s ===\n%s" % (rel, p.read_text()))
    spec = hidden_spec(task)
    if spec is not None:
        parts.append("=== HIDDEN TASK SPEC (Mac mini only; never shown to agents) ===\n%s" % json.dumps(spec, indent=1))
    return "\n\n".join(parts)


def lean_side_text(task):
    """Smoke-profile context for reviewers: CONTRACT.md and the reference (oracle) diff, nothing else."""
    parts = []
    for label, p in (("CONTRACT.md", ROOT / "CONTRACT.md"), ("oracle.diff", RUNS / "tasks" / task / "oracle.diff")):
        if p.is_file():
            parts.append("=== TRUSTED FILE %s ===\n%s" % (label, p.read_text()))
    return "\n\n".join(parts)


def base_code_text(task_dir):
    """The starter code at the base commit, from the Mine/Cut pack, if one exists."""
    try:
        conf = tomllib.loads((Path(task_dir) / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    meta = conf.get("metadata", {})
    port, base = meta.get("port"), meta.get("base_commit", "")
    pack = RUNS / "mine" / str(port) / ("pack-%s.txt" % base[:10])
    return pack.read_text() if port and pack.is_file() else ""


def implementation_qa(task_dir, include_base_code=True, lean=False):
    """lean=True (smoke profile): package + CONTRACT.md + oracle diff only, no base-commit pack or gateway code."""
    t0 = time.time()
    task_dir = Path(task_dir)
    task = task_name(task_dir)
    base = base_code_text(task_dir) if include_base_code and not lean else ""
    prompt = "\n\n".join(filter(None, [
        QA_CHECKLIST,
        "## The task package (what ships)\n\n" + package_text(task_dir),
        ("## The contract and the reference diff\n\n" + lean_side_text(task)) if lean else
        "## The trusted grading side (gateway on the device host)\n\n" + trusted_side_text(task),
        ("## Starter code at the base commit (context for checks 2, 3 and 8)\n\n" + base) if base else "",
    ]))
    result, usage = argon_json(prompt, QA_SCHEMA, "implementation_qa %s" % task)
    counts = {s: sum(1 for f in result["findings"] if f["severity"] == s) for s in ("critical", "major", "minor", "nit")}
    passed = counts["critical"] == 0 and counts["major"] == 0
    out = RUNS / "tasks" / task
    out.mkdir(parents=True, exist_ok=True)
    (out / "implementation_qa.json").write_text(json.dumps({"result": result, "usage": usage}, indent=1))
    log.info("implementation_qa %s: %s, findings %s, solvable=%s", task, "pass" if passed else "fail", counts,
             result["solvable"])
    return verdict("implementation_qa", "pass" if passed else "fail", passed, result["findings"],
                   {"summary": result["summary"], "solvable": result["solvable"],
                    "solvability_notes": result["solvability_notes"], "counts": counts,
                    "report": str(out / "implementation_qa.json")}, t0, usage)


# ---------------------------------------------------------------- stage 10: check-in review

CHECKIN = """You are the check-in reviewer for an RL task, deciding whether it may go on to exploration, calibration
and rollouts, which cost real money. You did not write it and have not seen its author's reasoning.
Read the package, the QA reviewer's findings and how each one was fixed or justified. Decide:
- approve: ready as is;
- approve_with_comments: ready, but list comments; mark any that must be fixed before rollouts;
- reject: not fit to continue; say why.
A justification only counts if it is specific and correct given the files. Comments must point at files."""

CHECKIN_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "approve_with_comments", "reject"]},
        "summary": {"type": "string"},
        "comments": {"type": "array", "items": {"type": "object", "properties": {
            "file": {"type": "string"}, "comment": {"type": "string"},
            "must_fix_before_rollouts": {"type": "boolean"}},
            "required": ["file", "comment", "must_fix_before_rollouts"]}},
    },
    "required": ["decision", "summary", "comments"],
}


def qa_trail(ledger_history):
    """QA findings and everything logged after them (fixes, justifications, re-reviews)."""
    trail, seen_qa = [], False
    for entry in ledger_history or []:
        n = stage_of(entry)
        if n == 9:
            seen_qa = True
        if seen_qa and n in (6, 9, 13) or (seen_qa and "fix" in str(entry.get("stage_name", "")).lower()):
            trail.append({k: entry.get(k) for k in ("ts", "stage", "stage_name", "round", "verdict", "passed",
                                                     "findings", "details")})
    return trail


def checkin_review(task_dir, ledger_history, lean=False):
    t0 = time.time()
    task_dir = Path(task_dir)
    task = task_name(task_dir)
    trail = qa_trail(ledger_history)
    prompt = "\n\n".join([
        CHECKIN,
        "## QA findings and their resolutions (from the factory ledger)\n\n" + json.dumps(trail, indent=1, default=str),
        "## The task package\n\n" + package_text(task_dir),
        ("## The contract and the reference diff\n\n" + lean_side_text(task)) if lean else
        "## The trusted grading side\n\n" + trusted_side_text(task),
    ])
    result, usage = argon_json(prompt, CHECKIN_SCHEMA, "checkin_review %s" % task)
    must_fix = [c for c in result["comments"] if c["must_fix_before_rollouts"]]
    if result["decision"] == "reject":
        decision, passed = "reject", False
    elif must_fix:
        decision, passed = "return", False
    else:
        decision, passed = "approve", True
    log.info("checkin_review %s: %s (%s), %d comments, %d must-fix", task, decision, result["decision"],
             len(result["comments"]), len(must_fix))
    return verdict("checkin_review", decision, passed, result["comments"],
                   {"decision": result["decision"], "summary": result["summary"], "must_fix": len(must_fix)},
                   t0, usage)


# ---------------------------------------------------------------- stage 16: final sign-off

def _rewards(entry):
    d = (entry or {}).get("details") or {}
    for key in ("rewards", "trial_rewards", "target_rewards"):
        if isinstance(d.get(key), list):
            return [float(r) for r in d[key] if r is not None]
    trials = d.get("trials")
    if isinstance(trials, list):
        return [float(t.get("reward", 0)) for t in trials if isinstance(t, dict) and t.get("reward") is not None]
    return []


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _pvar(xs):
    if not xs:
        return None
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def signoff_criteria(ledger_history, acc):
    last = latest_by_stage(ledger_history)
    roll = acc.get("rollouts", {})
    crit = []

    def c(name, met, evidence):
        crit.append({"criterion": name, "met": bool(met), "evidence": evidence})

    def passed(n):
        return bool(last.get(n, {}).get("passed"))

    c("Static checks pass", passed(6), last.get(6, {}).get("verdict", "not run"))
    noop = last.get(7)
    noop_r = ((noop or {}).get("details") or {}).get("reward")
    c("No-op earns no credit", passed(7) and (noop_r is None or float(noop_r) == 0.0),
      "reward %s" % noop_r if noop else "not run")
    base = last.get(8)
    base_r = _rewards(base) or ([float(base["details"]["mean_reward"])] if base and
                                (base.get("details") or {}).get("mean_reward") is not None else [])
    c("Baseline scores about 0", passed(8) and base_r and _mean(base_r) <= 0.05,
      "mean %.3f over %d trials" % (_mean(base_r), len(base_r)) if base_r else "not run")
    qa = last.get(9)
    c("QA findings resolved (latest QA passes or check-in approved them)",
      passed(9) or passed(10), "latest QA %s, check-in %s" % ((qa or {}).get("verdict", "not run"),
                                                             last.get(10, {}).get("verdict", "not run")))
    c("Check-in approved", passed(10), last.get(10, {}).get("verdict", "not run"))
    cal = last.get(12)
    cal_d = (cal or {}).get("details") or {}
    unresolved = cal_d.get("unresolved_hacks") or cal_d.get("hacks_unresolved")
    c("Calibration valid with no unresolved reward hacking", passed(12) and not unresolved,
      "target %s, unresolved hacks %s" % (cal_d.get("target_speedup"), unresolved or 0) if cal else "not run")
    red = last.get(13)
    c("Every red-team attack scores 0 (or the grader was fixed and the attacks rerun)", passed(13),
      "best attack reward %s" % ((red or {}).get("details") or {}).get("best_reward") if red else "not run")
    rol = last.get(14)
    r = _rewards(rol)
    rd = (rol or {}).get("details") or {}
    target_r = rd.get("target_rewards") if isinstance(rd.get("target_rewards"), list) else r
    target_r = [float(x) for x in target_r or []]
    need_n = roll.get("target_model_trials", 0)
    mean, var = _mean(target_r), _pvar(target_r)
    ok = (rol is not None and len(target_r) >= need_n and mean is not None and var is not None
          and mean <= roll.get("target_mean_max", 1.0) and var > roll.get("target_variance_min", 0.0))
    c("Rollouts inside the difficulty window", ok,
      "%d target trials, mean %s, variance %s (window: mean <= %s, variance > %s, >= %s trials)"
      % (len(target_r), None if mean is None else round(mean, 3), None if var is None else round(var, 4),
         roll.get("target_mean_max"), roll.get("target_variance_min"), need_n) if rol else "not run")
    strong_need = roll.get("strong_model_trials", [])
    strong_got = rd.get("strong_trials") if isinstance(rd.get("strong_trials"), list) else []
    c("Strong-model rollouts present", len(strong_got) >= len(strong_need) and
      all((g or 0) >= n for g, n in zip(strong_got, strong_need)),
      "%s trials per strong model (need %s)" % (strong_got, strong_need) if rol else "not run")
    ref = last.get(15)
    ref_r = _rewards(ref) or ([float(ref["details"]["mean_reward"])] if ref and
                              (ref.get("details") or {}).get("mean_reward") is not None else [])
    c("Reference scores about 1", passed(15) and ref_r and _mean(ref_r) >= 0.95,
      "mean %.3f over %d trials" % (_mean(ref_r), len(ref_r)) if ref_r else "not run")
    for name, got, need in (("Baseline", base_r, acc.get("baseline", {}).get("trials", 0)),
                            ("Reference", ref_r, acc.get("reference", {}).get("trials", 0))):
        c("%s trial count meets the acceptance config" % name, len(got) >= need,
          "%d trials (need %d)" % (len(got), need))
    return crit


SIGNOFF_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "risks": {"type": "array", "items": {"type": "string"}},
                   "headline_numbers": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "risks", "headline_numbers"],
}


def final_signoff(task_dir, ledger_history, acceptance_cfg=None, summarize=True):
    t0 = time.time()
    task_dir = Path(task_dir)
    task = task_name(task_dir)
    acc = acceptance_cfg if acceptance_cfg is not None else acceptance()
    crit = signoff_criteria(ledger_history, acc)
    passed = all(x["met"] for x in crit)
    summary, usage = {"summary": "", "risks": [], "headline_numbers": []}, None
    if summarize:
        prompt = ("You are signing off an RL task for delivery. Summarize for a buyer, in plain words, what the task "
                  "asks, how it is verified, the key measured numbers, and the remaining risks. Use only the facts "
                  "below; do not invent numbers.\n\nAcceptance criteria and evidence:\n" + json.dumps(crit, indent=1)
                  + "\n\nLedger:\n" + json.dumps(ledger_history, indent=1, default=str)[:400000]
                  + "\n\nInstruction:\n" + (task_dir / "instruction.md").read_text())
        summary, usage = argon_json(prompt, SIGNOFF_SCHEMA, "final_signoff %s" % task, thinking="medium")
    out = RUNS / "tasks" / task
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Report card: %s" % task, "",
             "**Decision:** %s" % ("signed off" if passed else "not signed off"), "",
             "| Criterion | Met | Evidence |", "|---|---|---|"]
    lines += ["| %s | %s | %s |" % (x["criterion"], "yes" if x["met"] else "**no**", x["evidence"]) for x in crit]
    if summary["summary"]:
        lines += ["", "## Summary", "", summary["summary"]]
    if summary["headline_numbers"]:
        lines += ["", "## Numbers", ""] + ["- %s" % n for n in summary["headline_numbers"]]
    if summary["risks"]:
        lines += ["", "## Risks", ""] + ["- %s" % r for r in summary["risks"]]
    lines += ["", "## Ledger", "", "| Time | Stage | Verdict |", "|---|---|---|"]
    for e in ledger_history or []:
        n = stage_of(e)
        lines.append("| %s | %s | %s |" % (e.get("ts", ""), STAGES.get(n, e.get("stage_name") or e.get("stage")),
                                           e.get("verdict")))
    (out / "report_card.md").write_text("\n".join(lines) + "\n")
    log.info("final_signoff %s: %s (%d of %d criteria met) -> %s", task, "signed off" if passed else "not signed off",
             sum(x["met"] for x in crit), len(crit), out / "report_card.md")
    return verdict("final_signoff", "approve" if passed else "fail", passed,
                   [x for x in crit if not x["met"]],
                   {"criteria": crit, "report_card": str(out / "report_card.md"), **summary}, t0, usage)


# ---------------------------------------------------------------- stage 13: fix from findings

EDITABLE = ("instruction.md", "README.md", "task.toml", "environment/workspace/validation/")
SPEC_FIELDS = ("dev_prompts", "hidden_prompts", "off_limits")

FIX = """You maintain an RL task package. Reviewers or red-team agents reported the findings below. Propose the
smallest concrete edits that fix them. You may edit only:
- instruction.md, README.md;
- task.toml, but only its [metadata] table (other sections are fixed harness settings);
- files under environment/workspace/validation/;
- the hidden spec on the device host, only its fields dev_prompts, hidden_prompts and off_limits
  (give the full new value of a field).
For text edits use mode "replace_text" with an `old` string copied exactly from the file that occurs once,
or mode "full_file" with the whole new file. If a finding can't be fixed within these limits, list it in
`unaddressed` with the reason (for example: it needs a change to the gateway or the harness)."""

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "mode": {"type": "string", "enum": ["replace_text", "full_file"]},
            "old": {"type": "string"}, "new": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["path", "mode", "new", "reason"]}},
        "spec_changes": {"type": "array", "items": {"type": "object", "properties": {
            "field": {"type": "string", "enum": list(SPEC_FIELDS)},
            "value_json": {"type": "string", "description": "the field's full new value, as JSON"},
            "reason": {"type": "string"}}, "required": ["field", "value_json", "reason"]}},
        "unaddressed": {"type": "array", "items": {"type": "object", "properties": {
            "finding": {"type": "string"}, "reason": {"type": "string"}}, "required": ["finding", "reason"]}},
    },
    "required": ["summary", "edits", "spec_changes", "unaddressed"],
}


def _apply_edit(task_dir, edit):
    rel = edit["path"].lstrip("/")
    target = (task_dir / rel).resolve()
    if task_dir.resolve() not in target.parents:
        return "path is outside the task folder"
    if not (rel in EDITABLE or any(rel.startswith(p) for p in EDITABLE if p.endswith("/"))):
        return "path is not editable by the fixer"
    before = target.read_text() if target.is_file() else ""
    if edit["mode"] == "replace_text":
        old = edit.get("old") or ""
        if not old or before.count(old) != 1:
            return "old text occurs %d times (needs exactly 1)" % (before.count(old) if old else 0)
        after = before.replace(old, edit["new"])
    else:
        after = edit["new"]
    if rel == "task.toml":
        try:
            a, b = tomllib.loads(before), tomllib.loads(after)
        except tomllib.TOMLDecodeError as exc:
            return "task.toml would not parse: %s" % exc
        a.pop("metadata", None), b.pop("metadata", None)
        if a != b:
            return "edit changes task.toml outside [metadata]"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(after)
    if rel.endswith(".sh"):
        target.chmod(0o755)
    return None


def _write_spec(task, spec):
    body = json.dumps(spec, indent=1)
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "macmini",
                           "cat > ~/task-factory/tasks/%s.json.tmp && mv ~/task-factory/tasks/%s.json.tmp "
                           "~/task-factory/tasks/%s.json" % (task, task, task)],
                          input=body, capture_output=True, text=True, timeout=30)
    return None if done.returncode == 0 else "could not write the hidden spec: %s" % done.stderr[-300:]


def fix_from_findings(task, findings, dry_run=False):
    t0 = time.time()
    task_dir = ROOT / "tasks" / task
    spec = hidden_spec(task)
    editable_text = "\n\n".join("=== FILE %s ===\n%s" % (rel, text) for rel, text in package_files(task_dir)
                                if text is not None and (rel in EDITABLE or rel.startswith(EDITABLE[-1])))
    spec_view = {k: (spec or {}).get(k) for k in SPEC_FIELDS}
    prompt = "\n\n".join([FIX, "## Findings\n\n" + json.dumps(findings, indent=1, default=str),
                          "## Editable files\n\n" + editable_text,
                          "## Hidden spec fields you may change (current values)\n\n" + json.dumps(spec_view, indent=1),
                          "## The rest of the package, read-only\n\n" + package_text(task_dir, skip=[
                              rel for rel, _ in package_files(task_dir) if rel in EDITABLE or rel.startswith(EDITABLE[-1])])])
    result, usage = argon_json(prompt, FIX_SCHEMA, "fix_from_findings %s" % task)
    applied, rejected = [], []
    for edit in result["edits"]:
        err = "dry run" if dry_run else _apply_edit(task_dir, edit)
        (rejected if err else applied).append({"path": edit["path"], "mode": edit["mode"], "reason": edit["reason"],
                                               **({"rejected": err} if err else {})})
    spec_applied = []
    if result["spec_changes"] and spec is not None:
        new_spec = dict(spec)
        for ch in result["spec_changes"]:
            try:
                new_spec[ch["field"]] = json.loads(ch["value_json"])
                spec_applied.append({"field": ch["field"], "reason": ch["reason"]})
            except ValueError:
                rejected.append({"path": "hidden spec: " + ch["field"], "rejected": "value_json is not JSON"})
        if spec_applied and not dry_run:
            err = _write_spec(task, new_spec)
            if err:
                rejected += [{"path": "hidden spec: " + s["field"], "rejected": err} for s in spec_applied]
                spec_applied = []
    elif result["spec_changes"]:
        rejected.append({"path": "hidden spec", "rejected": "could not read the hidden spec"})
    out = RUNS / "tasks" / task
    out.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "findings": len(findings), "summary": result["summary"],
             "applied": applied, "spec_fields_changed": [s["field"] for s in spec_applied], "rejected": rejected,
             "unaddressed": result["unaddressed"], "dry_run": dry_run, "usage": usage}
    with (out / "fixes.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    passed = bool(applied or spec_applied) and not rejected and not result["unaddressed"]
    log.info("fix_from_findings %s: %d edits applied, %d spec fields changed, %d rejected, %d unaddressed%s",
             task, len(applied), len(spec_applied), len(rejected), len(result["unaddressed"]),
             " (dry run)" if dry_run else "")
    return verdict("fix_from_findings", "pass" if passed else "fail", passed,
                   rejected + [{"unaddressed": u["finding"], "reason": u["reason"]} for u in result["unaddressed"]],
                   {"summary": result["summary"], "applied": applied,
                    "spec_fields_changed": [s["field"] for s in spec_applied], "changelog": str(out / "fixes.jsonl")},
                   t0, usage)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["static", "qa", "checkin", "signoff", "fix"])
    ap.add_argument("task")
    ap.add_argument("findings", nargs="?", help="fix: JSON file with a list of findings")
    ap.add_argument("--no-summary", action="store_true", help="signoff: skip the Argon summary")
    ap.add_argument("--dry-run", action="store_true", help="fix: propose edits without applying them")
    args = ap.parse_args()
    task_dir = ROOT / "tasks" / args.task
    ledger_file = RUNS / "ledger" / ("%s.jsonl" % args.task)
    history = ([json.loads(l) for l in ledger_file.read_text().splitlines() if l.strip()]
               if ledger_file.exists() else [])
    if args.stage == "static":
        v = static_checks(task_dir)
    elif args.stage == "qa":
        v = implementation_qa(task_dir)
    elif args.stage == "checkin":
        v = checkin_review(task_dir, history)
    elif args.stage == "signoff":
        v = final_signoff(task_dir, history, summarize=not args.no_summary)
    else:
        v = fix_from_findings(args.task, json.loads(Path(args.findings).read_text()), dry_run=args.dry_run)
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main()
