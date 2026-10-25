"""Can an Antigravity sandbox call the cloud gateway directly? The /try key is stored with the Credentials API
and injected at the sandbox's network proxy as the X-TF-Agent header, only for the gateway's host; the sandbox
never sees the key and can reach no other host.

    uv run python cloud/reach.py --url https://tf-gateway-....run.app

Result: runs/agents/cloud-reach.json   Progress: tail -f logs/cloud-gateway.log
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import ANTIGRAVITY, FLASH, RUNS, gemini_client, get_logger  # noqa: E402

log = get_logger("cloud-gateway")
CRED_ID = "tf-gateway-agent"

PROMPT = """Run these shell commands one at a time and report each command's exact output, nothing else:
1. curl -s -m 30 -o /dev/null -w '%{{http_code}}' https://{host}/health
2. curl -s -m 60 -X POST https://{host}/try -H 'Content-Type: application/json' -d '{{"task":"reach-test-no-such-task"}}' -w ' HTTP %{{http_code}}'
3. curl -s -m 20 -o /dev/null -w '%{{http_code}}' https://github.com ; echo " exit=$?"
4. env | grep -ci 'tf-\\|x-tf\\|tf_' ; true
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    host = urlparse(args.url).netloc
    client = gemini_client()
    token = (Path.home() / ".config" / "task-factory" / "gateway-agent-key").read_text().strip()
    try:
        client.credentials.delete(id=CRED_ID)
    except Exception:
        pass
    cred = client.credentials.create(id=CRED_ID, type="bearer_token", token=token, header_name="X-TF-Agent", prefix="")
    log.info("credential %s: %s", CRED_ID, getattr(cred, "status", "created"))
    env = {"type": "remote", "network": {"allowlist": [{"domain": host, "credential": CRED_ID}]}}
    t0 = time.time()
    it = client.interactions.create(agent=ANTIGRAVITY, agent_config={"type": "antigravity", "model": FLASH},
                                    input=PROMPT.format(host=host), environment=env, tools=[{"type": "code_execution"}],
                                    background=True)
    while it.status in ("in_progress", "queued"):
        time.sleep(8)
        it = client.interactions.get(id=it.id)
    text = it.output_text or ""
    result = {"host": host, "status": it.status, "seconds": round(time.time() - t0), "interaction": it.id, "output": text}
    (RUNS / "agents").mkdir(parents=True, exist_ok=True)
    (RUNS / "agents" / "cloud-reach.json").write_text(json.dumps(result, indent=1))
    log.info("reach test %s in %ds:\n%s", it.status, result["seconds"], text[-1500:])


if __name__ == "__main__":
    main()
