"""Report for one validation call. Exit codes: 0 valid, 1 the submission has a problem, 2 the check itself
failed (gateway or phone), in which case there is no score."""
import json
import os
import sys
import urllib.request

path = sys.argv[1]
if not os.path.exists(path):
    print("no patch at %s: write your git diff there first (see instruction.md)" % path)
    sys.exit(1)
patch = open(path).read()
req = urllib.request.Request(os.environ["TF_GATEWAY"] + "/try",
                             data=json.dumps({"task": os.environ["TF_TASK"], "patch": patch}).encode())
try:
    r = json.load(urllib.request.urlopen(req, timeout=3000)).get("result", {})
except Exception as exc:
    print("VALIDATION ERROR (not your submission): gateway unreachable: %s" % exc)
    sys.exit(2)
if r.get("infra_error"):
    print("VALIDATION ERROR (not your submission): %s" % r["infra_error"])
    sys.exit(2)
if r.get("error"):
    print("SUBMISSION PROBLEM (%s): %s" % (r.get("error_kind"), r["error"]))
    if r.get("off_limits_touched"):
        print("off-limits files in the patch: %s" % ", ".join(r["off_limits_touched"]))
    sys.exit(1)
m = r.get("metrics", {})
print("development prompts checked: %d (the final grade uses hidden prompts of the same kind)" % len(r.get("checks", [])))
for c in r.get("checks", []):
    first = next((i for i, (a, b) in enumerate(zip(c.get("tokens", []), c.get("expected", []))) if a != b), None)
    print("  %-60s 8-token run %s, 136-token run %s, decode %s tok/s%s" % (
        repr(c["input"])[:60], "ok" if c["short_match"] else "WRONG", "ok" if c["long_match"] else "WRONG",
        c.get("decode_tok_s"), "" if first is None else " (first wrong token at position %d)" % first))
print("decode: %s tok/s, base: %s tok/s, speedup %sx, target %sx" % (
    m.get("decode_tok_s"), m.get("base_decode_tok_s"), m.get("speedup"), m.get("target_speedup")))
for f in r.get("flags", []):
    print("flag: %s" % f)
print("valid: %s   score with these rules: %s" % (r.get("valid"), r.get("score")))
sys.exit(0 if r.get("valid") else 1)
