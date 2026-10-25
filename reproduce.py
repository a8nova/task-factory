"""Check whether a merged llama.cpp speedup PR is really faster than the commit before it, on this machine.

    python3 reproduce.py 26947 --model models/SmolLM2-135M-Instruct-Q8_0.gguf

Builds the commit before the merge (base) and the merge commit in Docker, times both with llama-bench
(runs alternate base/merge so drift hits both), and checks the outputs still match with llama-perplexity's
KL divergence. Results: runs/reproduce-<pr>-<time>.json   Progress: tail -f logs/reproduce.log
"""
import argparse
import datetime as dt
import json
import logging
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "src" / "llama.cpp"
IMAGE = "task-factory-build"
DOCKER_BIN = "/Applications/Docker.app/Contents/Resources/bin"
MIN_SPEEDUP = 1.1  # GSO's bar for a commit to become a task
LOG_FILE = ROOT / "logs" / "reproduce.log"

log = logging.getLogger("reproduce")
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)):
    handler.setFormatter(fmt)
    log.addHandler(handler)
log.setLevel(logging.INFO)
ENV = {**os.environ, "PATH": f"{DOCKER_BIN}:{os.environ['PATH']}"}


def sh(cmd, timeout=3600, check=True):
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
    if check and done.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}... exit {done.returncode}: {done.stderr[-1500:]}")
    return done


def git(*args):
    return sh(["git", "-C", str(REPO), *args]).stdout.strip()


def worktree(sha):
    path = ROOT / "src" / f"wt-{sha[:12]}"
    if not path.exists():
        git("worktree", "add", "--detach", str(path), sha)
    return path


def docker(wt, command, extra_mounts=(), timeout=3600):
    # Every worktree is mounted at /src so ccache reuses objects across commits.
    cmd = ["docker", "run", "--rm", "-v", f"{wt}:/src", "-v", "task-factory-ccache:/ccache",
           "-e", "CCACHE_BASEDIR=/src", "-v", f"{ROOT / 'models'}:/models:ro"]
    for host, inside in extra_mounts:
        cmd += ["-v", f"{host}:{inside}"]
    return sh(cmd + [IMAGE, "bash", "-c", command], timeout=timeout)


def build(wt):
    if (wt / "build" / "bin" / "llama-bench").exists():
        return
    log.info("building %s", wt.name)
    docker(wt, "cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF "
               "-DLLAMA_BUILD_TESTS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
               "> /src/build.log 2>&1 && cmake --build /src/build -j8 --target llama-bench llama-perplexity "
               ">> /src/build.log 2>&1 || (tail -40 /src/build.log; exit 1)")
    log.info("built %s", wt.name)


def bench(wt, model, args):
    out = docker(wt, f"/src/build/bin/llama-bench -m /models/{model} {args} -o json")
    return {f"{'pp' if r['n_prompt'] else 'tg'}{r['n_prompt'] or r['n_gen']}": r["samples_ts"]
            for r in json.loads(out.stdout)}


def kl_divergence(base_wt, merge_wt, model, text, work):
    flags = f"-m /models/{model} -f /data/{text.name} -c 128 --chunks 4 -t 4"
    mounts = [(text.parent, "/data"), (work, "/out")]
    docker(base_wt, f"/src/build/bin/llama-perplexity {flags} --kl-divergence-base /out/base.kld > /out/kld-base.txt 2>&1",
           mounts)
    docker(merge_wt, f"/src/build/bin/llama-perplexity {flags} --kl-divergence-base /out/base.kld --kl-divergence "
                     f"> /out/kld-merge.txt 2>&1", mounts)
    report = (work / "kld-merge.txt").read_text()
    (work / "base.kld").unlink(missing_ok=True)
    mean_kld = re.search(r"Mean\s+KLD:\s+([-\d.e]+)", report)
    same_top = re.search(r"Same top p:\s+([\d.]+)", report)
    return {"mean_kld": float(mean_kld.group(1)) if mean_kld else None,
            "same_top_token_pct": float(same_top.group(1)) if same_top else None,
            "report_tail": report[-1200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pr", type=int)
    ap.add_argument("--model", required=True, help="GGUF file inside models/")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--bench-args", default="-p 512 -n 32 -fa 1 -t 4 -r 3")
    args = ap.parse_args()
    model = Path(args.model).name

    pr = json.loads(sh(["gh", "pr", "view", str(args.pr), "--repo", "ggml-org/llama.cpp",
                        "--json", "number,title,mergeCommit"]).stdout)
    merge = pr["mergeCommit"]["oid"]
    git("fetch", "-q", "origin", merge)
    base = git("rev-parse", f"{merge}^1")
    log.info("PR #%d %s: base %s -> merge %s", args.pr, pr["title"], base[:10], merge[:10])
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    work = ROOT / "runs" / f"reproduce-{args.pr}-{stamp}"
    work.mkdir(parents=True)

    base_wt, merge_wt = worktree(base), worktree(merge)
    build(base_wt)
    build(merge_wt)

    samples = {"base": {}, "merge": {}}
    for i in range(args.rounds):
        for side, wt in (("base", base_wt), ("merge", merge_wt)):
            for test, ts in bench(wt, model, args.bench_args).items():
                samples[side].setdefault(test, []).extend(ts)
        log.info("round %d/%d done", i + 1, args.rounds)

    results = {}
    for test in samples["base"]:
        b, m = samples["base"][test], samples["merge"][test]
        speedup = statistics.mean(m) / statistics.mean(b)  # tokens/s, so higher is faster
        spread = (statistics.stdev(b) + statistics.stdev(m)) / statistics.mean(b)
        results[test] = {"base_tps": round(statistics.mean(b), 2), "base_sd": round(statistics.stdev(b), 2),
                         "merge_tps": round(statistics.mean(m), 2), "merge_sd": round(statistics.stdev(m), 2),
                         "speedup": round(speedup, 3), "noise": round(spread, 3),
                         "passes_1.1x": speedup >= MIN_SPEEDUP, "beyond_noise": speedup - 1 > spread}
        log.info("%s base %.1f±%.1f t/s, merge %.1f±%.1f t/s -> %.3fx (noise %.3f)", test,
                 statistics.mean(b), statistics.stdev(b), statistics.mean(m), statistics.stdev(m), speedup, spread)

    text = work / "text.txt"
    text.write_text((base_wt / "README.md").read_text() * 3)
    kld = kl_divergence(base_wt, merge_wt, model, text, work)
    log.info("outputs: mean KL divergence %s, same top token %s%%", kld["mean_kld"], kld["same_top_token_pct"])

    cpu = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip()
    out = {"pr": args.pr, "title": pr["title"], "base": base, "merge": merge, "machine": cpu, "model": model,
           "bench_args": args.bench_args, "rounds": args.rounds, "results": results, "kl_divergence": kld,
           "samples": samples}
    (work / "result.json").write_text(json.dumps(out, indent=1))
    log.info("result -> %s", work / "result.json")


if __name__ == "__main__":
    main()
