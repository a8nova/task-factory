# Task Factory

Task Factory turns real, merged speedup commits into verified Harbor
tasks for training and evaluating coding agents. Before a task ships, it is timed on real Android phones and
attacked by agents that try to cheat its grader.

The source repo is [adreno-llms](https://github.com/a8nova/adreno-llms): OpenCL ports of small models for
Qualcomm Adreno GPUs. A task gives an agent a port at the commit before a speedup, and the reward is the decode
speed it reaches on a phone with exactly the reference's output tokens.

## How a task is made

Each stage writes a verdict to `runs/ledger/<task>.jsonl`. A stage that fails or is returned goes back to
its author with the findings (at most 2 revisions). The full contract is in [CONTRACT.md](CONTRACT.md).

| Stages | What happens | Who |
|---|---|---|
| 1 Mine | Read one model port and its whole git history in one call; pick commits that can become tasks | Gemini 4 Argon |
| 2–4 Propose | Write the task proposal; check it against the rules; review it | Argon, Flash-Lite, Argon |
| 5 Implement | Write the Harbor package (instruction, scope rules, attack list) around the fixed harness in `harness/` | Argon |
| 6–10 Checks | Static checks; no-op and baseline must score about 0 on the phone; audit; check-in review | scripts, Argon |
| 11–14 Agents | Explore the task end to end; calibrate the target; red team tries to cheat and a fixer patches the grader; rollouts check the difficulty | Antigravity managed agents (Argon, Flash) |
| 15–16 Ship | The reference commit must score 1; final sign-off | phone, Argon |

Grading runs on the cloud gateway ([cloud/README.md](cloud/README.md)): a Cloud Run service that builds a patch
with the Android NDK, times it on Google's Device Run phones (Pixel 5, Adreno 620), checks every output token
and signs the result. `device/` holds the original local gateway for a phone on USB (a Razr 2020 on a Mac mini).

## Setup

1. Python 3.13 and [uv](https://docs.astral.sh/uv/): `uv sync`. On an Intel Mac this needs `cryptography<46`
   (already pinned), because newer versions have no prebuilt wheel there.
2. Gemini API key, kept outside the repo:
   ```sh
   mkdir -p ~/.config/task-factory
   printf '%s' 'YOUR_KEY' > ~/.config/task-factory/gemini-api-key
   chmod 600 ~/.config/task-factory/gemini-api-key
   ```
   One key covers `gemini-4-argon`, `antigravity-preview-09-2026`, `gemini-3.8-flash` and `gemini-3.5-flash-lite`.
3. The source repo: `git clone https://github.com/a8nova/adreno-llms src/adreno-llms`.
4. A gateway: deploy the cloud gateway (see [cloud/README.md](cloud/README.md)), or run `device/gateway.py` on a
   machine with a phone attached. Put the gateway's keys next to the Gemini key, in
   `~/.config/task-factory/gateway-agent-key` and `~/.config/task-factory/gateway-verifier-key` (mode 600).
   Each key is read the first time a call needs it.
5. Optional: `private/acceptance.toml` with your difficulty thresholds. It's not published; defaults are used
   when it's missing.

## Make a task

```sh
uv run python pack.py smollm2-135m-instruct          # one port + its history, with a token estimate
uv run python mine.py smollm2-135m-instruct          # -> runs/mine/<port>/candidates.json
uv run python factory.py new-task --task my-task --port smollm2-135m-instruct --candidate 887bd7e
uv run python factory.py run --task my-task --port smollm2-135m-instruct --candidate 887bd7e
uv run python factory.py status --task my-task
```

`factory.py run` takes `--from N --to M` to run part of the pipeline, `--profile smoke` for a fast pass with a
faster model, and `--skip-phone` to skip the phone stages. Single steps can be run on their own:

```sh
uv run python agents.py my-task --role redteam               # one agent per attack
uv run python agents.py my-task --role rollout --n 4         # rollouts + difficulty check
uv run python -m lifecycle.gates my-task reference --trials 3
uv run python -m lifecycle.review signoff my-task
```

Every long step logs to `logs/<step>.log`.

## Run a finished task

A task is a standard Harbor folder in `tasks/<task>/`: `instruction.md`, `task.toml`, `environment/` (the agent's
container), `solution/` (the reference, never shown to the agent) and `tests/` (the verifier, in its own container).

```sh
harbor run --path tasks/<task> --agent <agent>
```

## Monitor

```sh
ui/start.sh          # background; restarted if it exits. Stop with: ui/start.sh stop
```

Open http://127.0.0.1:8777. The monitor has these views:
- **Repos:** repo stats, with a split into chunks that each fit one Argon call. Add a repo from a GitHub link.
- **Tasks:** every task and how far it got. A task's page shows its Harbor package, every proposal round, and
  which agent did each step.
- **Agents** and **Rollouts:** every managed agent and every rollout, live.
- **Monitor:** one task's stages, its agents and the machines they run on.

The monitor only reads, except when you add a repo.

## Layout

| Path | What |
|---|---|
| `factory.py` | The stage machine (stages 2–16) |
| `pack.py`, `mine.py`, `cut.py` | Pack a port, mine candidates, write the task package |
| `agents.py` | Antigravity managed agents: explorer, calibrator, red team, rollouts |
| `lifecycle/` | Proposal, review, gates, ledger and trajectory stages |
| `harness/` | The fixed Harbor harness every task is built on |
| `cloud/` | The Cloud Run grading gateway and its scripts |
| `device/` | The local phone gateway and Device Run timing |
| `tasks/` | Finished Harbor task packages |
| `ui/` | The monitor |
| `reproduce.py`, `docker/` | Reproduce a llama.cpp speedup PR in Docker |

`runs/`, `logs/`, `src/`, `models/`, `data/` and `private/` are local only and gitignored.

## Security

- The Gemini key lives in `~/.config/task-factory/`, never in the repo, and is never printed or logged.
- Hidden prompts, their reference tokens and the gateway's keys stay in the gateway: in Secret Manager for the
  cloud gateway, and only on the device host for the local one. Every gateway response is signed with HMAC-SHA256.
- Agents run in Antigravity sandboxes with networking disabled. They reach the phone only through tools that
  the laptop answers, and only the laptop holds the gateway keys.
