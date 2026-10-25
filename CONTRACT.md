# Task Factory contract (v2)

Every part of the factory builds against this file. Change it only together with every part that depends on it.

## Task package (Harbor, `tasks/<task>/`)

```text
tasks/<task>/
  instruction.md            # sections: Task, Constraints, Success Criteria, plus a time-budget notice
  task.toml                 # schema 1.4; artifacts = ["/workspace/submission"]; separate verifier
  README.md
  environment/
    Dockerfile              # copies workspace/ to /workspace; repo at the base commit only, in /workspace/repo
    workspace/
      validation/val.sh     # agent validation: POST $TF_GATEWAY/try with /workspace/submission/patch.diff
      timer.sh              # prints seconds left (records start on first call; budget from $TF_AGENT_TIMEOUT)
      task/expected_tokens.json   # reference tokens on the fixed input and the development prompts
  solution/
    baseline.sh             # writes an empty patch.diff + summary.md into /workspace/submission (scores ~0)
    solve.sh                # calls baseline.sh
    reference/patch.diff    # the merged commit's change for this port
    reference/summary.md
    reference.sh            # copies reference/ into /workspace/submission (scores ~1)
  tests/
    Dockerfile              # separate verifier image; includes /tests
    test.sh                 # grades /workspace/submission/patch.diff via the gateway
    grade_client.py
```

Agent deliverable: `/workspace/submission/patch.diff` (a `git diff` against the base commit, run in
`/workspace/repo`) and `/workspace/submission/summary.md`. Only that folder reaches the verifier.

## Gateway v2 (Mac mini, `device/gateway.py`)

Runs on port 8787 (v1, until the current red team finishes) and 8788 (v2). Laptop tunnels: 8799 -> 8787,
8798 -> 8788.

| Endpoint | Who | Body | Returns |
|---|---|---|---|
| `POST /try` | agents (val.sh, run_on_phone) | `{task, patch}` | `{ok, error_kind, error, decode_tok_s, speedup, inputs: [{input, tokens, expected, match}], valid, score}` on the fixed input + development prompts only |
| `POST /grade` | verifier (`X-TF-Verifier`) | `{task, patch}` | `{valid, reward, metrics: {decode_tok_s, speedup, target_speedup}, checks, flags}` or `{infra_error}` |
| `POST /calibrate` | factory (`X-TF-Factory`) | `{task}` | base + reference measured on fixed, development and hidden prompts; cached |
| `POST /measure` | factory | `{port, commit, patch?, rounds?}` | raw measurement |
| `POST /set_target` | factory | `{task, target_speedup, source}` | updates the target after calibration |

`error_kind` is `submission` (patch doesn't apply, build fails, binary crashes, off-limits file) or `infra`
(adb, phone, gateway). Infra errors never produce a reward.

Every response is `{"result": ..., "signature": HMAC-SHA256}`.

## Reward

- **valid** = patch applies and builds, the binary runs, no off-limits file is touched, and output tokens
  exactly equal the reference's on the fixed input and every prompt (hidden prompts in `/grade`).
- **reward** = 0 if not valid, else `clip((speedup - 1) / (target_speedup - 1), 0, 1)`, where speedup is
  candidate decode tok/s over base decode tok/s, both measured outside the binary on the phone.
- `target_speedup` starts at the reference's measured speedup and is replaced by the best valid result
  from calibration (stage 12).

## Hidden task spec (Mac mini only, `~/task-factory/tasks/<task>.json`)

`{port, base, reference, rounds, dev_prompts, hidden_prompts, off_limits, target_speedup, target_source}`.
Hidden prompts never leave the Mac mini.

## Lifecycle stages (`factory.py`, ledger at `runs/ledger/<task>.jsonl`)

Each stage returns a verdict: `{"stage", "verdict": "approve|return|reject|pass|fail", "passed": bool,
"findings": [...], "details": {...}, "seconds", "agent": {"model", "interaction"}}`. A stage that
returns or fails sends the task back to the author agent with the findings (at most 2 rounds), then the
stage runs again.

| # | Stage | Module | Function |
|---|---|---|---|
| 1 | Mine | `mine.py` | existing |
| 2 | Propose | `lifecycle/proposal.py` | `propose(candidate, spec) -> proposal` |
| 3 | Proposal QA | `lifecycle/proposal.py` | `proposal_qa(proposal) -> verdict` |
| 4 | Proposal review | `lifecycle/proposal.py` | `review_proposal(proposal) -> verdict` |
| 5 | Implement | `cut.py` | `cut(task)` (Argon) + `assemble` |
| 6 | Static checks | `lifecycle/review.py` | `static_checks(task_dir) -> verdict` |
| 7 | No-op | `lifecycle/gates.py` | `noop(task) -> verdict` |
| 8 | Baseline | `lifecycle/gates.py` | `baseline(task, trials) -> verdict` |
| 9 | Implementation QA | `lifecycle/review.py` | `implementation_qa(task_dir) -> verdict` |
| 10 | Check-in review | `lifecycle/review.py` | `checkin_review(task_dir, ledger) -> verdict` |
| 11 | Explore | `agents.py` | `run_role(task, "explorer", ...)` |
| 12 | Calibrate | `agents.py` | `run_role(task, "calibrator", ...)` |
| 13 | Red team + fixer | `agents.py`, `lifecycle/review.py` | `run_role(task, "redteam")`, `fix_from_findings(task, findings)` |
| 14 | Rollouts | `agents.py` | `run_role(task, "rollout", ...)` + `difficulty_check(results, acceptance)` |
| 15 | Reference | `lifecycle/gates.py` | `reference(task, trials) -> verdict` |
| 16 | Final sign-off | `lifecycle/review.py` | `final_signoff(task_dir, ledger, acceptance) -> verdict` |

Acceptance thresholds come from `private/acceptance.toml` (gitignored).

## Rules for every part

- Log every long step to `logs/<name>.log`; never print or log keys.
- Gemini calls go through `common.gemini_client()`; Argon for authoring and reviewing, Flash and
  Flash-Lite for fast checks and rollouts.
- Agents run in Antigravity sandboxes with `network: "disabled"`; hardware is reachable only through
  custom functions the orchestrator answers.
- One request reaches the phone at a time; keep phone use small during development.
