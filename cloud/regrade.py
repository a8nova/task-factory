"""Grade saved agent submissions through the cloud gateway's /grade (hidden prompts), all in parallel, and write
<run dir>/regrade.json. Used when a rollout's own grading hit an infrastructure error.

    uv run python cloud/regrade.py --url https://tf-gateway-....run.app \
        runs/agents/smollm2-image2d-decode/20261025-145545-rollout rollout-1 rollout-2 rollout-4

Each agent's latest submit-N.diff is graded. Results already in regrade.json are kept unless regraded again.
Progress: tail -f logs/regrade.log
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates import call, log, summary  # noqa: E402


def latest_submission(agent_dir):
    subs = sorted(agent_dir.glob("submit-*.diff"), key=lambda p: int(p.stem.split("-")[1]))
    return subs[-1] if subs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--url", required=True)
    ap.add_argument("--task", help="defaults to the run directory's parent name")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    task = args.task or run_dir.parent.name
    jobs = {}
    for agent in args.agents:
        sub = latest_submission(run_dir / agent)
        if not sub:
            log.info("grade %s (%s) skipped: %s", task, agent, {"reason": "no submission saved"})
            continue
        jobs[agent] = sub

    def grade(agent):
        sub = jobs[agent]
        # "grade <task> (<agent> <submission>): {...}", the line format the UI monitor parses
        log.info("grade %s (%s %s) started: %s", task, agent, sub.stem, {"diff_lines": len(sub.read_text().splitlines())})
        r = call(args.url, "/grade", {"task": task, "patch": sub.read_text()}, "X-TF-Verifier", "gateway-verifier-key")
        log.info("grade %s (%s %s): %s", task, agent, sub.stem, summary(r))
        return agent, {"submission": str(sub), "reward": r.get("reward"), "valid": r.get("valid"),
                       "speedup": (r.get("metrics") or {}).get("speedup"), "flags": r.get("flags"),
                       "error_kind": r.get("error_kind"), "error": (r.get("error") or "")[:1500] or None,
                       "infra_error": r.get("infra_error"), "http_error": r.get("http_error"),
                       "scoring": r.get("scoring"), "metrics": r.get("metrics"), "timings": r.get("timings"),
                       "session": r.get("session"), "signature_ok": r.get("signature_ok"),
                       "seconds": r.get("seconds")}

    out_file = run_dir / "regrade.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    with ThreadPoolExecutor(max(1, len(jobs))) as pool:
        for agent, res in pool.map(grade, jobs):
            results[agent] = res
            out_file.write_text(json.dumps(results, indent=1))
    log.info("wrote %s", out_file)


if __name__ == "__main__":
    main()
