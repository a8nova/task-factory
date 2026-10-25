#!/bin/bash
# Reference solution: the merged commit's changes for this port.
set -euo pipefail
cd /app
git apply --whitespace=nowarn /solution/oracle.diff
