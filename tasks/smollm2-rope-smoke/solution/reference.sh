#!/bin/bash
# Reference submission: the merged commit's change for this port. It scores about 1.
set -euo pipefail
mkdir -p /workspace/submission
cp "$(dirname "$0")/reference/patch.diff" "$(dirname "$0")/reference/summary.md" /workspace/submission/
