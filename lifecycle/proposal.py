"""Stages 2-4: the Proposer writes a task proposal, Proposal QA checks it fast, and a Proposal Reviewer in a
fresh session approves, returns or rejects it. The Proposer revises in its own session.

Progress: tail -f logs/proposal.log
"""
import copy
import json
import re
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor

from common import ARGON, FLASH, FLASH_LITE, ROOT, RUNS, gemini_client, get_logger

log = get_logger("proposal")
ACCEPTANCE = ROOT / "private" / "acceptance.toml"


def acceptance():
    return tomllib.loads(ACCEPTANCE.read_text()) if ACCEPTANCE.exists() else {}


def interact(model, prompt, schema, thinking="high", previous=None, max_output=65536, background=True):
    """One Interactions API call with JSON output. Returns (parsed JSON, agent metadata)."""
    client = gemini_client()
    kwargs = {"model": model, "input": prompt,
              "generation_config": {"thinking_level": thinking, "max_output_tokens": max_output},
              "response_format": {"type": "text", "mime_type": "application/json", "schema": schema}}
    if previous:
        kwargs["previous_interaction_id"] = previous
    if background:
        kwargs["background"] = True
    t0 = time.time()
    it = client.interactions.create(**kwargs)
    while it.status in ("in_progress", "queued"):
        time.sleep(10)
        it = client.interactions.get(id=it.id)
    if it.status != "completed":
        raise RuntimeError(f"{model} interaction {it.id} ended {it.status}: {getattr(it, 'error', '')}")
    usage = it.usage
    meta = {"model": model, "interaction": it.id, "input_tokens": usage.total_input_tokens,
            "output_tokens": usage.total_output_tokens, "seconds": round(time.time() - t0)}
    log.info("%s %s: %ss, tokens in %s out %s", model, it.id[:24], meta["seconds"], meta["input_tokens"],
             meta["output_tokens"])
    return json.loads(it.output_text), meta


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "slug": {"type": "string"},
        "problem_statement": {"type": "string"},
        "starter_method": {"type": "string", "description": "what the agent starts from (the base commit) and how it runs"},
        "deliverable": {"type": "string", "description": "exactly what the agent submits and where"},
        "metric": {"type": "object", "properties": {
            "name": {"type": "string"}, "unit": {"type": "string"}, "how_measured": {"type": "string"},
            "baseline_value": {"type": "number"}, "target_value": {"type": "number"}},
            "required": ["name", "unit", "how_measured", "baseline_value", "target_value"]},
        "correctness_rule": {"type": "string"},
        "reward": {"type": "string", "description": "how metric and correctness turn into a 0-1 reward"},
        "why_hard": {"type": "string"},
        "cheat_resistance": {"type": "array", "items": {"type": "object", "properties": {
            "cheat": {"type": "string"}, "defense": {"type": "string"}}, "required": ["cheat", "defense"]}},
        "resources": {"type": "object", "properties": {
            "hardware": {"type": "string"}, "software": {"type": "string"}, "network": {"type": "string"}},
            "required": ["hardware", "software", "network"]},
        "agent_time_budget_sec": {"type": "integer"},
        "reference_solution": {"type": "string"},
        "baseline_expectation": {"type": "string", "description": "why the unchanged starter scores about 0"},
        "expected_difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "slug", "problem_statement", "starter_method", "deliverable", "metric", "correctness_rule",
                 "reward", "why_hard", "cheat_resistance", "resources", "agent_time_budget_sec",
                 "reference_solution", "baseline_expectation", "expected_difficulty", "open_questions"],
}

PROPOSE_PROMPT = """You are the Proposer in a task factory that turns real engineering history into RL training
tasks for coding agents. Write a task proposal for the candidate commit below. A reviewer who has never seen
your reasoning will approve, return or reject it, so make every claim specific and checkable.

The task family: the agent starts from the BASE commit of one adreno-llms model port (an OpenCL LLM engine
for Adreno phone GPUs) and must make fp16 greedy decode faster on a real phone without changing the output.
The merged commit is the reference solution; the agent never sees it.

Fixed facts about how tasks in this factory are run and graded (state them, don't change them):
- Grading runs on a real phone (Motorola Razr 2020, Snapdragon 765G, Adreno 620) through a device gateway.
  The gateway builds the agent's patch with the Android NDK, deploys it, and times decode from OUTSIDE the
  binary (a long generation minus a short one, timed by the phone's uptime clock). The binary's own printed
  timings are ignored.
- Correctness is exact equality of output token IDs with the reference implementation, on a fixed input
  and on hidden prompts the agent never sees. The agent gets development prompts with their expected tokens.
- Reward is 0 if not valid, else clip((speedup - 1) / (target_speedup - 1), 0, 1); the unchanged starter
  must score about 0 and the reference about 1.
- The agent submits /workspace/submission/patch.diff (a git diff against the base) and summary.md; only that
  folder reaches a separate verifier. The agent has no internet; it validates on the phone through a
  validation script and has a timer.
- Agent time budget at most {max_agent} seconds.

Measured on the phone so far (null if not measured yet):
{measured}

Candidate notes from the mining step:
{candidate}

Base commit code and history of the port:
{pack}
"""

SPEC_PROPOSE_PROMPT = """You are the Proposer in a task factory that turns real engineering code into RL training tasks
for coding agents. Write a task proposal from the task spec below. A reviewer who has never seen your reasoning
will approve, return or reject it, so make every claim specific and checkable.

This task is built from a spec, not from a past speedup commit: the agent starts from the STARTER commit (the
port's current code) of one adreno-llms model port (an OpenCL inference engine for Adreno phone GPUs) and must
make it faster on a real phone without changing its output. There is no reference commit. The target metric
and the reference solution come from calibration: strong agents attempt the task with long budgets, and the
best valid submission becomes both the target (reward 1) and solution/reference/.

Fixed facts about how tasks in this factory are run and graded (state them, don't change them):
- Grading runs on a real phone in Google's Device Run fleet ({device}) through a device gateway. The gateway
  builds the agent's patch with the Android NDK, deploys it, and times the run from OUTSIDE the binary (runs
  of different lengths, timed by the phone's own clock; the per-unit time is the slope). The binary's own
  printed timings are ignored.
- Correctness: {correctness}. The agent gets development inputs with their expected outputs; hidden inputs
  are never shown to it.
- Reward is 0 if not valid, else clip((speedup - 1) / (target_speedup - 1), 0, 1) with the target from
  calibration; the unchanged starter must score about 0 and the reference about 1.
- The agent submits /workspace/submission/patch.diff (a git diff against the starter) and summary.md; only that
  folder reaches a separate verifier. The agent has no internet; it validates on the phone through a
  validation script and has a timer.
- Agent time budget at most {max_agent} seconds.

Task spec:
{spec}

Starter commit code and history of the port:
{pack}
"""

QA_CHECKS_SPEC_REFERENCE = ("reference: the proposal says the target and the reference solution come from "
                            "calibration (the best valid calibrator submission), and how calibration is run")

REVISE_PROMPT = """The proposal you wrote was checked. Below is its current text, which may have been edited
since you wrote it, and the findings against it. Revise it to address every finding, or explain in
open_questions why a finding does not apply. Return the complete revised proposal in the same JSON format.

Current proposal:
{proposal}

Findings:
{findings}
"""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {"type": "array", "items": {"type": "object", "properties": {
            "check": {"type": "string"},
            "evidence_quote": {"type": "string", "description": "verbatim text copied from the proposal"},
            "passed": {"type": "boolean"},
            "finding": {"type": "string"}},
            "required": ["check", "evidence_quote", "passed", "finding"]}},
    },
    "required": ["checks"],
}

QA_CHECKS = [
    "metric: one measurable number with a unit, measured by the grader outside the agent's code (not a number the agent's binary reports about itself), with baseline and target values",
    "correctness: a deterministic pass/fail rule that does not depend on judgment",
    "baseline: the unchanged starter would score about 0 under the stated reward",
    "reference: a known solution exists that would score about 1",
    "difficulty: solvable within the time budget by a strong agent, but not trivial",
    "secrecy: the statement reveals no hidden inputs, expected hidden outputs, or the reference code",
    "deliverable: exactly what to submit and where is stated",
    "cheats: the obvious cheats are named and each has a defense in the grading setup",
    "resources: hardware, software and network needs are stated and available offline to the agent",
]

QA_PROMPT = """You are Proposal QA in a task factory. A proposal that slips through here costs hours of phone
time and agent rollouts later, so be strict. Run each check below on the task proposal.

For every check:
- evidence_quote: copy, word for word, the sentence or phrase from the proposal that the check depends on.
  If the proposal says nothing that satisfies the check, write "" and fail the check.
- passed: true only if the quoted text itself satisfies the check. Do not assume details the proposal
  doesn't state. A metric the binary prints about itself is not a measurement taken outside it. A reward the
  unchanged starter scores well on does not leave room to improve. Anything that reveals hidden grading
  inputs or expected hidden outputs fails secrecy.
- finding: one sentence on what is missing or wrong, quoting the problem; if it passes, say what satisfies it.

Checks:
{checks}

Proposal:
{proposal}
"""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "return", "reject"]},
        "summary": {"type": "string"},
        "comments": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["blocking", "suggestion"]}, "comment": {"type": "string"}},
            "required": ["severity", "comment"]}},
    },
    "required": ["verdict", "summary", "comments"],
}

REVIEW_PROMPT = """You are the Proposal Reviewer in a task factory that produces RL training tasks for coding
agents. You have not seen how this proposal was written. Decide whether it should go to implementation.

Approve only if the task is useful for training (a real engineering skill, a measurable result), fair
(solvable from what the agent is given within the budget), hard to cheat (every obvious shortcut is blocked
by the grading setup), and gradeable without judgment. Check in particular, and quote the proposal when
you raise one: is there a deterministic correctness rule; is the metric measured by the grader outside the
agent's code rather than reported by the agent's own binary; does the statement reveal hidden grading inputs
or their expected outputs; would the unchanged starter score near 0 under the stated reward; is the time
budget within {max_agent} seconds. Any of these failing is blocking. Return the proposal with blocking
comments if it can be fixed; reject it if the idea itself cannot work.

Proposal:
{proposal}

Automated QA findings on this proposal:
{qa}

Measured facts from the phone (null if not measured yet):
{measured}

The reference solution (the merged commit for this port; the agent never sees it), for judging feasibility:
{oracle}
"""


def _measured(spec):
    keys = ("phone", "device", "kind", "metric", "correctness_rule", "base_tps", "reference_tps",
            "reference_speedup", "needed_speedup", "target_speedup", "target_source")
    return json.dumps({k: spec.get(k) for k in keys}, indent=1)


def propose(candidate, spec):
    """Stage 2. Returns {"proposal", "agent"}; the agent's interaction id is kept for revisions.
    Spec tasks (spec["kind"] == "spec") have no mined candidate and no reference commit."""
    port = spec["port"]
    pack = (RUNS / "mine" / port / f"pack-{spec['base'][:10]}.txt").read_text()
    max_agent = acceptance().get("limits", {}).get("agent_timeout_sec_max", 21600)
    if spec.get("kind") == "spec":
        public = {k: v for k, v in spec.items() if k not in ("hidden_prompts",)}
        prompt = SPEC_PROPOSE_PROMPT.format(device=spec.get("device", spec.get("phone", "")),
                                            correctness=spec.get("correctness_rule", ""), max_agent=max_agent,
                                            spec=json.dumps(public, indent=1), pack=pack)
    else:
        prompt = PROPOSE_PROMPT.format(max_agent=max_agent, measured=_measured(spec),
                                       candidate=json.dumps(candidate, indent=1), pack=pack)
    log.info("propose: %s on %s (%d chars)", (candidate or {}).get("commit", "spec")[:10], port, len(prompt))
    proposal, meta = interact(ARGON, prompt, PROPOSAL_SCHEMA)
    return {"proposal": proposal, "agent": meta}


def revise(proposed, findings):
    """Back to the Proposer, in its own session, with the findings to address."""
    text = "\n".join(f"- {f}" for f in findings)
    proposal, meta = interact(ARGON, REVISE_PROMPT.format(proposal=json.dumps(proposed["proposal"], indent=1),
                                                          findings=text), PROPOSAL_SCHEMA,
                              previous=proposed["agent"]["interaction"])
    return {"proposal": proposal, "agent": meta}


def _static_findings(proposal):
    """Checks that need no model: limits from the private acceptance config and required content."""
    findings = []
    limit = acceptance().get("limits", {}).get("agent_timeout_sec_max")
    if limit and proposal.get("agent_time_budget_sec", 0) > limit:
        findings.append(f"time budget {proposal['agent_time_budget_sec']}s is over the {limit}s limit")
    if proposal.get("agent_time_budget_sec", 0) <= 0:
        findings.append("no agent time budget")
    m = proposal.get("metric", {})
    if not (m.get("target_value", 0) > m.get("baseline_value", 0) > 0):
        findings.append("metric needs a positive baseline value and a higher target value")
    if not proposal.get("cheat_resistance"):
        findings.append("no cheats or defenses listed")
    return findings


def _flat(value):
    """All text in the proposal, whitespace-normalized, for checking quotes."""
    if isinstance(value, dict):
        return " ".join(_flat(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flat(v) for v in value)
    return str(value)


def _norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def proposal_qa(proposed, spec=None):
    """Stage 3. Code checks for limits; Flash for the judgment checks, each backed by a verbatim quote.
    A pass without a quote that really appears in the proposal counts as a failed check."""
    proposal = proposed["proposal"] if "proposal" in proposed else proposed
    t0 = time.time()
    static = _static_findings(proposal)
    checks = [QA_CHECKS_SPEC_REFERENCE if (spec or {}).get("kind") == "spec" and c.startswith("reference:") else c
              for c in QA_CHECKS]
    result, meta = interact(FLASH, QA_PROMPT.format(checks="\n".join(f"- {c}" for c in checks),
                                                    proposal=json.dumps(proposal, indent=1)),
                            QA_SCHEMA, thinking="medium", max_output=16384, background=False)
    corpus = _norm(_flat(proposal))
    for c in result["checks"]:
        quote = _norm(c.get("evidence_quote", "").strip(" \"'`"))
        c["quote_found"] = len(quote) >= 12 and quote in corpus
        if c["passed"] and not c["quote_found"]:
            c["passed"] = False
            c["finding"] = "no verbatim evidence quoted from the proposal (was: %s)" % c["finding"]
    failed = [f"{c['check']}: {c['finding']}" for c in result["checks"] if not c["passed"]]
    findings = static + failed
    return {"stage": "proposal_qa", "verdict": "pass" if not findings else "fail", "passed": not findings,
            "findings": findings, "details": {"checks": result["checks"], "static": static},
            "seconds": round(time.time() - t0), "agent": meta}


def review_proposal(proposed, qa=None, spec=None, oracle=None):
    """Stage 4. A fresh Argon session: never the Proposer's interaction."""
    proposal = proposed["proposal"] if "proposal" in proposed else proposed
    spec = spec or {}
    if spec.get("kind") == "spec":
        oracle = oracle or ("(none yet: this is a spec task; the target and the reference solution come from "
                            "calibration, the best valid calibrator submission)")
    oracle = oracle or "(not available)"
    qa_text = json.dumps(qa["details"]["checks"] if qa else [], indent=1)
    t0 = time.time()
    max_agent = acceptance().get("limits", {}).get("agent_timeout_sec_max", 21600)
    result, meta = interact(ARGON, REVIEW_PROMPT.format(proposal=json.dumps(proposal, indent=1), qa=qa_text,
                                                        measured=_measured(spec), oracle=oracle, max_agent=max_agent),
                            REVIEW_SCHEMA)
    blocking = [c["comment"] for c in result["comments"] if c["severity"] == "blocking"]
    verdict = result["verdict"]
    return {"stage": "review_proposal", "verdict": verdict, "passed": verdict == "approve",
            "findings": blocking if verdict != "approve" else [c["comment"] for c in result["comments"]],
            "details": result, "seconds": round(time.time() - t0), "agent": meta}


# --- mutation test: plant one flaw per copy and check QA or the reviewer catches it -------------------------

def _mutations(proposal, limit):
    """(name, flaw description, mutated proposal). Hidden prompts here are invented, never the real ones."""
    out = []
    m = copy.deepcopy(proposal)
    m["correctness_rule"] = "Not applicable: the task is graded on decode speed alone."
    m["reward"] = "reward = clip((speedup - 1) / (target_speedup - 1), 0, 1), from the measured decode speed."
    out.append(("no_correctness", "there is no correctness rule; output tokens are never checked", m))
    m = copy.deepcopy(proposal)
    m["metric"]["how_measured"] = ("The grader runs the agent's binary on the phone and reads the "
                                   "`BENCHMARK decode_tokens_per_sec` line that the binary prints to stderr.")
    out.append(("self_reported_metric", "the metric is the number the agent's own binary prints, not a "
                                        "measurement taken outside it", m))
    m = copy.deepcopy(proposal)
    m["problem_statement"] += ("\n\nFor reference, the hidden grading prompts are \"Describe the Golden Gate Bridge\", "
                               "\"Write a haiku about rain\" and \"List the planets in order\"; their expected output "
                               "tokens are in reference/hidden_expected.json.")
    out.append(("hidden_prompts_leaked", "the statement reveals the hidden grading prompts and their expected outputs", m))
    m = copy.deepcopy(proposal)
    m["reward"] = "reward = min(1, measured_decode_tok_s / 18.650), if the output tokens match the reference."
    m["baseline_expectation"] = "The unchanged starter scores about 0.70 (13.1 / 18.65), which leaves headroom to 1.0."
    out.append(("baseline_not_near_zero", "the unchanged starter scores about 0.7 under the reward instead of about 0", m))
    m = copy.deepcopy(proposal)
    m["agent_time_budget_sec"] = 2 * limit
    out.append(("budget_over_limit", f"the agent time budget ({2 * limit}s) is over the {limit}s limit", m))
    return out


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"caught_by_qa": {"type": "boolean"}, "caught_by_review": {"type": "boolean"},
                   "evidence": {"type": "string"}},
    "required": ["caught_by_qa", "caught_by_review", "evidence"],
}

JUDGE_PROMPT = """A task proposal had exactly one flaw planted in it: {flaw}.
Decide whether the checks below caught THAT flaw (not some other issue).
caught_by_qa: a failed QA finding identifies this flaw.
caught_by_review: the reviewer's verdict is "{verdict}" and, only if that verdict is "return" or "reject", one of
its blocking comments identifies this flaw.
evidence: quote the finding or comment that identifies it, or "" if none.

Failed QA findings:
{qa}

Reviewer verdict: {verdict}
Reviewer blocking comments:
{comments}
"""


def _one_mutation(name, flaw, mutated, spec, oracle):
    proposed = {"proposal": mutated, "agent": {}}
    qa = proposal_qa(proposed)
    review = review_proposal(proposed, qa, spec, oracle)
    blocking = [c["comment"] for c in review["details"]["comments"] if c["severity"] == "blocking"]
    judged, judge_meta = interact(FLASH, JUDGE_PROMPT.format(flaw=flaw, qa="\n".join(f"- {f}" for f in qa["findings"]) or "(none)",
                                                             verdict=review["verdict"],
                                                             comments="\n".join(f"- {c}" for c in blocking) or "(none)"),
                                  JUDGE_SCHEMA, thinking="low", max_output=4096, background=False)
    caught_review = judged["caught_by_review"] and review["verdict"] in ("return", "reject")
    caught = (judged["caught_by_qa"] and not qa["passed"]) or caught_review
    row = {"flaw": name, "description": flaw, "caught": caught,
           "caught_by": [w for w, hit in (("qa", judged["caught_by_qa"] and not qa["passed"]),
                                          ("review", caught_review)) if hit],
           "evidence": judged["evidence"], "qa_verdict": qa["verdict"], "qa_findings": qa["findings"],
           "review_verdict": review["verdict"], "review_blocking": blocking,
           "tokens": {"qa": [qa["agent"]["input_tokens"], qa["agent"]["output_tokens"]],
                      "review": [review["agent"]["input_tokens"], review["agent"]["output_tokens"]],
                      "judge": [judge_meta["input_tokens"], judge_meta["output_tokens"]]},
           "mutated": mutated}
    log.info("mutation %s: caught=%s by %s | QA %s, review %s", name, caught, row["caught_by"] or "nobody",
             qa["verdict"], review["verdict"])
    return row


def mutation_test(task):
    """Plant one flaw per copy of the task's approved proposal; every flaw must be caught by QA or the reviewer.
    Writes runs/tasks/<task>/qa-mutation.json."""
    work = RUNS / "tasks" / task
    proposed = json.loads((work / "proposal.json").read_text())
    spec = json.loads((work / "spec.json").read_text()) if (work / "spec.json").exists() else {}
    oracle = (work / "oracle.diff").read_text() if (work / "oracle.diff").exists() else None
    limit = acceptance().get("limits", {}).get("agent_timeout_sec_max", 21600)
    muts = _mutations(proposed["proposal"], limit)
    t0 = time.time()

    def control():
        """The unmutated proposal: shows false alarms (it should pass QA and be approved)."""
        qa = proposal_qa(proposed)
        review = review_proposal(proposed, qa, spec, oracle)
        return {"qa_verdict": qa["verdict"], "qa_findings": qa["findings"], "review_verdict": review["verdict"],
                "review_blocking": [c["comment"] for c in review["details"]["comments"] if c["severity"] == "blocking"],
                "clean": qa["passed"] and review["passed"],
                "tokens": {"qa": [qa["agent"]["input_tokens"], qa["agent"]["output_tokens"]],
                           "review": [review["agent"]["input_tokens"], review["agent"]["output_tokens"]]}}

    with ThreadPoolExecutor(len(muts) + 1) as pool:
        ctl = pool.submit(control)
        rows = list(pool.map(lambda m: _one_mutation(*m, spec, oracle), muts))
        ctl = ctl.result()
    total_in = sum(v[0] for r in rows + [ctl] for v in r["tokens"].values())
    total_out = sum(v[1] for r in rows + [ctl] for v in r["tokens"].values())
    caught = sum(r["caught"] for r in rows)
    report = {"task": task, "catch_rate": f"{caught}/{len(rows)}", "passed": caught == len(rows) and ctl["clean"],
              "control": ctl, "seconds": round(time.time() - t0),
              "tokens": {"input": total_in, "output": total_out}, "flaws": rows}
    (work / "qa-mutation.json").write_text(json.dumps(report, indent=1))
    log.info("mutation test %s: caught %d/%d in %ds, tokens in %d out %d -> %s", task, caught, len(rows),
             report["seconds"], total_in, total_out, work / "qa-mutation.json")
    return report
