# Task Factory: notes for agents working in this repo

## Machines
- **Laptop** (Intel i5-1038NG7, macOS): runs the orchestrator and holds the Gemini API key. Disk is tight; keep builds off it.
- **Mac mini** (`ssh macmini`, Apple M2 Pro, 216 GB free): build and device host. It has Android NDK `26.3.11579264` (r26d), `~/.nnopt/deps/opencl`, and adreno-llms at `~/Projects/adreno-llms`. The Razr 2020 (Snapdragon 765G / SM7250, Adreno 620) is on its USB and authorized. adb is at `~/Library/Android/sdk/platform-tools/adb` (not on PATH over ssh).

## Secrets
- The Gemini API key lives in `~/.config/task-factory/gemini-api-key` (mode 600), outside the repo. `common.gemini_client()` reads it. Never print, log or commit it.
- One key covers `gemini-4-argon`, `antigravity-preview-09-2026`, `gemini-3.8-flash` and `gemini-3.5-flash-lite`.

## Quirks
- Docker Desktop's CLI is at `/Applications/Docker.app/Contents/Resources/bin` and is not on PATH; `reproduce.py` adds it.
- `uv sync` on the Intel laptop needs Python 3.13 and `cryptography<46`; newer cryptography has no prebuilt wheel here and fails to build.
- The whole adreno-llms repo (17 MB of text) does not fit in one Argon call. One model port plus its git history does (about 260k–465k tokens, from `pack.py`).
- Every long-running step logs to `logs/<step>.log`.

## Monitor
- `ui/start.sh` runs it in the background (restarted if it exits; `ui/start.sh stop` stops it; log in `logs/ui.log`). It restarts itself when `ui/server.py` changes, and open pages reload when `ui/index.html` changes.
- `python3 ui/server.py` runs it in the foreground. Either way, open http://127.0.0.1:8777: every stage, every agent and what is running on each machine, live. It is read-only. It reads `runs/ledger/`, `runs/agents/`, `runs/tasks/` and `logs/*.log`, calls GET `/health` on both gateways through the tunnels, and reads the Mac mini's `devrun.log` and process list over ssh (prompt text is replaced before it reaches the page).
- Four views: **Repos** (paste a GitHub link to digest a repo: stats, languages, and a split into chunks of at most 450k code tokens, each one Argon call; clones go to `src/repos/`, digests to `runs/repos/`; logic in `ui/repos.py`), **Tasks** (every task and how far it got; accepted means stage 16 passed; click one for its Task page: the Harbor package files, every proposal round with the checker's and reviewer's verdicts, and an Agents tab: every role and step, from the ledger, `fixes.jsonl` and agent runs), **Agents** (every managed agent in every task, running ones first) and **Monitor** (one task live).
- It parses `logs/agents.log` lines (`<task>/<role>-<i> started (...)`, `<tool> #n (k diff lines) -> {json}`, `finished <status> in Ns`) and the ledger rows. Keep those formats when you change `agents.py` or `lifecycle/ledger.py`, or update `ui/server.py` with them.
