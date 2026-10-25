#!/bin/bash
# Harbor's oracle entrypoint: packages the baseline submission.
set -euo pipefail
bash "$(dirname "$0")/baseline.sh"
