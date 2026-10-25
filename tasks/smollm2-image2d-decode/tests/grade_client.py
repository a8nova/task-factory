"""Ask the device gateway to grade the agent's diff; trust the answer only if its signature checks out."""
import hashlib
import hmac
import json
import os
import sys
import urllib.request

patch = open(sys.argv[1]).read()
req = urllib.request.Request(os.environ["TF_GATEWAY"] + "/grade",
                             data=json.dumps({"task": os.environ["TF_TASK"], "patch": patch}).encode(),
                             headers={"X-TF-Verifier": os.environ["TF_VERIFIER_KEY"]})
reward, detail = 0.0, {}
try:
    reply = json.load(urllib.request.urlopen(req, timeout=1700))
    body = json.dumps(reply["result"], sort_keys=True, separators=(",", ":"))
    expected = hmac.new(os.environ["TF_GATEWAY_HMAC"].encode(), body.encode(), hashlib.sha256).hexdigest()
    if hmac.compare_digest(expected, reply.get("signature", "")):
        detail = reply["result"]
        reward = float(detail.get("reward", 0.0))
    else:
        detail = {"error": "gateway signature did not verify"}
except Exception as exc:
    detail = {"error": "grading failed: %s" % exc}
print(json.dumps(detail, indent=1))
json.dump({"reward": reward, "speedup": float(detail.get("speedup") or 0.0),
           "correct": 1.0 if detail.get("correct") else 0.0}, open("/logs/verifier/reward.json", "w"))
open("/logs/verifier/reward.txt", "w").write("%g\n" % reward)
