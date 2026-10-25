#!/bin/bash
# Runs in a separate container the agent never touched. Sends the agent's diff to the gateway's /grade,
# checks the gateway's signature, and writes the reward.
set -u
mkdir -p /logs/verifier
cd /app
git config --global --add safe.directory /app
git add -N . >/dev/null 2>&1 || true
git diff --binary f6e489b3326b94264dde6e78ea5c7a68c748a258 > /tmp/agent.diff
python3 /tests/grade_client.py /tmp/agent.diff
