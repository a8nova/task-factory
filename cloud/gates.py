"""Calibrate a task on the cloud gateway, then grade the oracle, an empty patch and every saved cheat.

    uv run python cloud/gates.py smollm2-image2d-decode --url https://tf-gateway-....run.app

Grades run in parallel (each is its own Device Run session). Every response's signature is checked.
Results: runs/tasks/<task>/cloud-gates.json   Progress: tail -f logs/cloud-gateway.log
"""
import argparse
import hashlib
import hmac
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import RUNS, get_logger  # noqa: E402

log = get_logger("cloud-gateway")
KEYS = Path.home() / ".config" / "task-factory"


def key(name):
    return (KEYS / name).read_text().strip()


def call(url, path, body, header, keyfile, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={header: key(keyfile), "Content-Type": "application/json"})
    t0 = time.time()
    try:
        reply = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as exc:
        return {"http_error": exc.code, "body": exc.read().decode()[:1500], "seconds": round(time.time() - t0)}
    payload = json.dumps(reply["result"], sort_keys=True, separators=(",", ":"))
    ok = hmac.compare_digest(hmac.new(key("gateway-hmac").encode(), payload.encode(), hashlib.sha256).hexdigest(),
                             reply.get("signature", ""))
    return {"signature_ok": ok, "seconds": round(time.time() - t0), **reply["result"]}


def summary(r):
    keep = ("valid", "reward", "flags", "error_kind", "error", "infra_error", "http_error", "signature_ok", "seconds")
    out = {k: r.get(k) for k in keep if r.get(k) is not None}
    if r.get("metrics"):
        out["speedup"] = r["metrics"].get("speedup")
        out["decode_tok_s"] = r["metrics"].get("decode_tok_s")
        out["base_decode_tok_s"] = r["metrics"].get("base_decode_tok_s")
    if r.get("timings"):
        out["timings"] = r["timings"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    work = RUNS / "tasks" / args.task
    out = {"task": args.task, "gateway": args.url, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log.info("calibrating %s on %s", args.task, args.url)
    out["calibrate"] = call(args.url, "/calibrate", {"task": args.task}, "X-TF-Factory", "gateway-factory-key")
    log.info("calibrate: %s", {k: out["calibrate"].get(k) for k in ("device", "base_decode_tok_s", "reference_decode_tok_s",
                                                                    "target_speedup", "session_seconds", "seconds", "error", "body", "infra_error")})
    patches = {"oracle": (work / "oracle.diff").read_text(), "empty": ""}
    for c in json.loads((work / "cheats" / "index.json").read_text()):
        patches[c["id"]] = (work / "cheats" / ("%s.diff" % c["id"])).read_text()
    with ThreadPoolExecutor(len(patches)) as pool:
        futures = {name: pool.submit(call, args.url, "/grade", {"task": args.task, "patch": p}, "X-TF-Verifier",
                                     "gateway-verifier-key") for name, p in patches.items()}
        out["grades"] = {name: f.result() for name, f in futures.items()}
    for name, r in out["grades"].items():
        log.info("grade %s: %s", name, summary(r))
    out["summary"] = {name: summary(r) for name, r in out["grades"].items()}
    (work / "cloud-gates.json").write_text(json.dumps(out, indent=1))
    log.info("-> %s", work / "cloud-gates.json")


if __name__ == "__main__":
    main()
