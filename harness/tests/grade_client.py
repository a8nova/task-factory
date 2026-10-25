"""Grade the submitted patch through the device gateway and write the reward.

Validity, raw metrics and reward are reported separately. A missing deliverable is a submission failure
(reward 0). A gateway, phone or signature failure is not the submission's fault: it is reported as an error
and no reward file is written, so it can't be mistaken for a zero.
"""
import hashlib
import hmac
import json
import os
import sys
import urllib.request

LOGS = "/logs/verifier"


def write(reward, valid, metrics, detail):
    json.dump(detail, open(os.path.join(LOGS, "details.json"), "w"), indent=1)
    json.dump({"reward": reward, "valid": 1.0 if valid else 0.0, "speedup": float(metrics.get("speedup") or 0.0),
               "decode_tok_s": float(metrics.get("decode_tok_s") or 0.0)},
              open(os.path.join(LOGS, "reward.json"), "w"))
    open(os.path.join(LOGS, "reward.txt"), "w").write("%g\n" % reward)
    print(json.dumps({"valid": valid, "metrics": metrics, "reward": reward}, indent=1))


def infra(message):
    print("VERIFIER INFRASTRUCTURE ERROR (no reward written): %s" % message)
    sys.exit(1)


path = sys.argv[1]
if not os.path.exists(path):
    write(0.0, False, {}, {"error": "no submission: %s is missing" % path})
    sys.exit(0)
req = urllib.request.Request(os.environ["TF_GATEWAY"] + "/grade",
                             data=json.dumps({"task": os.environ["TF_TASK"], "patch": open(path).read()}).encode(),
                             headers={"X-TF-Verifier": os.environ["TF_VERIFIER_KEY"]})
try:
    reply = json.load(urllib.request.urlopen(req, timeout=3500))
except Exception as exc:
    infra("gateway unreachable: %s" % exc)
body = json.dumps(reply.get("result"), sort_keys=True, separators=(",", ":"))
expected = hmac.new(os.environ["TF_GATEWAY_HMAC"].encode(), body.encode(), hashlib.sha256).hexdigest()
if not hmac.compare_digest(expected, reply.get("signature", "")):
    infra("gateway signature did not verify")
r = reply["result"]
if r.get("infra_error"):
    infra(r["infra_error"])
write(float(r.get("reward", 0.0)), bool(r.get("valid")), r.get("metrics") or {}, r)
