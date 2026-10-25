#!/bin/bash
# Run your current changes on the phone: the gateway builds them with the Android NDK, deploys to the
# Razr 2020, times decode from outside the binary, and returns the 40 output tokens for the fixed input.
# Takes about 1-2 minutes. Usage: /app/task/phone_run.sh
set -euo pipefail
cd /app
git add -N . >/dev/null 2>&1 || true
git diff --binary f6e489b3326b94264dde6e78ea5c7a68c748a258 > /tmp/phone_run.diff
python3 - <<'PY'
import json, os, urllib.request
patch = open("/tmp/phone_run.diff").read()
req = urllib.request.Request(os.environ["TF_GATEWAY"] + "/try",
                             data=json.dumps({"task": os.environ["TF_TASK"], "patch": patch}).encode())
try:
    reply = json.load(urllib.request.urlopen(req, timeout=1800))
except Exception as exc:
    raise SystemExit("gateway unreachable: %s" % exc)
r = reply.get("result", reply)
if r.get("error"):
    print("BUILD OR RUN FAILED:\n" + r["error"][-4000:])
    raise SystemExit(1)
want = json.load(open("/app/task/expected_tokens.json"))
got = r.get("tokens") or []
first_diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None if len(got) == len(want) else min(len(got), len(want)))
print("decode tokens/s (measured outside the binary): %s" % r.get("decode_tok_s_external"))
print("long-run seconds: %s" % r.get("long_seconds"))
print("output tokens: %s" % got)
print("tokens match expected: %s%s" % (first_diff is None, "" if first_diff is None else " (first difference at position %d)" % first_diff))
PY
