#!/bin/bash
# Baseline submission: the unchanged starter code (an empty patch). It scores about 0.
set -euo pipefail
mkdir -p /workspace/submission
: > /workspace/submission/patch.diff
cat > /workspace/submission/summary.md <<'MD'
# Baseline
The unchanged starter code at the base commit. No optimization.
MD
