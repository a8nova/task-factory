#!/bin/bash
# Agent validation. Sends a patch to the device gateway, which applies it to the base commit, builds it with
# the Android NDK, runs it on the Motorola Razr 2020 (Adreno 620) on the development prompts, and reports
# token checks, decode speed and the score the same rules would give. The final grade uses hidden prompts.
#
#   /workspace/validation/val.sh          check /workspace/submission/patch.diff (what gets graded)
#   /workspace/validation/val.sh --repo   check the current changes in /workspace/repo instead
#
# One call takes about 2-5 minutes; the phone runs one build at a time.
set -uo pipefail
if [ "${1:-}" = "--repo" ]; then
    cd /workspace/repo && git add -N . >/dev/null 2>&1; git diff --binary {{BASE}} > /tmp/val.diff
    PATCH=/tmp/val.diff
else
    PATCH=/workspace/submission/patch.diff
fi
exec python3 /workspace/validation/val.py "$PATCH"
