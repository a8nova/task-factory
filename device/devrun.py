"""Build one adreno-llms version and time it on the attached phone. Runs on the device host (the Mac mini).

    python3 devrun.py --port smollm2-135m-instruct --commit f6e489b [--patch agent.diff] --prompt "Why is the sky blue?"

The phone's own clock times every run from a runner script this tool owns, so nothing the build prints
about its own speed is trusted. For each prompt the same build generates several lengths; the caller fits
a line through (length, time) to get decode tok/s, so a delay that only fires on short runs can't inflate
the slope. Every run prints its tokens, and the caller checks those exact tokens, so a build can't be fast
on a timing run and correct on a different run. Timing only ever uses the prompts passed in (development or
hidden prompts), never the port's public fixed input, which an agent could special-case.

Safety on the build host:
- Only the files the task allows may differ from the base commit; any other modified or new file, or a
  symlink, makes the patch invalid (this also stops edits to build scripts).
- Build scripts and CMake files are forced back to the base commit before building, as defence in depth.
- The build runs under sandbox-exec, denied reads of the keys, the hidden specs, the caches and the repo's
  git history, so a planted `#include` of a secret cannot leak it through a compiler error.
- Returned build logs are capped and any 64-hex-character string (a key) is redacted.

Errors are classified: "submission" (patch, build or binary) or "infra" (adb, phone, this machine: never a
reward). One build reaches the phone at a time across every process (an flock on ~/task-factory/phone.lock).
Prints one JSON result on stdout. Progress: tail -f ~/task-factory/logs/devrun.log
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
BASE = HOME / "task-factory"
CLONE = BASE / "adreno-llms"
WORKTREES = BASE / "wt"
PHONE_LOCK = BASE / "phone.lock"
ADB = str(HOME / "Library/Android/sdk/platform-tools/adb")
UPSTREAM = "https://github.com/a8nova/adreno-llms"
DEVICE_ROOT = "/data/local/tmp/tf"
DEFAULT_LENGTHS = (8, 40, 136)   # generated tokens; the caller may pass jittered lengths
HEX64 = re.compile(r"[0-9a-f]{64}")
# Build files can never be part of an optimization; they are forced to the base commit and may not be edited.
BUILD_FILE_HINTS = ("scripts/", "CMakeLists.txt", ".cmake", "/cmake/", "toolchain")
# Reads the sandboxed build is denied, so a planted #include of these can't leak them through compiler errors.
DENY_READS = [str(HOME / ".config" / "task-factory"), str(BASE / "tasks"), str(BASE / "cache"),
              str(BASE / "v2" / "tasks"), str(BASE / "v2" / "cache"), str(CLONE / ".git")]
PORTS = {
    "smollm2-135m-instruct": {
        "binary": "SmolLM2_135M_Instruct_inference_fp16",
        "deployed": "/data/local/tmp/SmolLM2_135M_Instruct_inference",
    },
}
SAFE_PROMPT = re.compile(r"^[A-Za-z0-9 ,.:?()=]+$")

(BASE / "logs").mkdir(parents=True, exist_ok=True)
log = logging.getLogger("devrun")
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
for handler in (logging.FileHandler(BASE / "logs" / "devrun.log"), logging.StreamHandler(sys.stderr)):
    handler.setFormatter(fmt)
    log.addHandler(handler)
log.setLevel(logging.INFO)


class SubmissionError(Exception):
    """The candidate's fault: bad patch, failed build, failing binary."""


class InfraError(Exception):
    """Not the candidate's fault: adb, the phone, git on this machine."""


def scrub(text, limit=1500):
    return HEX64.sub("<redacted>", text)[-limit:]


def sh(cmd, cwd=None, timeout=1800, check=True, env=None):
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    if check and done.returncode != 0:
        raise RuntimeError("%s exit %d: %s" % (" ".join(cmd[:4]), done.returncode, (done.stdout + done.stderr)[-2000:]))
    return done


def adb(*args, timeout=600, check=True):
    try:
        return sh([ADB, *args], timeout=timeout, check=check)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        raise InfraError("adb %s: %s" % (args[0], exc))


@contextlib.contextmanager
def phone_lock():
    with open(PHONE_LOCK, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def git(path, *args, check=True):
    return sh(["git", "-C", str(path), *args], check=check)


def enforce_allowlist(path, port, allow):
    """After applying the patch, the only files that may differ from the base are the task's in-scope files.
    Anything else (including edits to build scripts, new files, or symlinks) makes the submission invalid."""
    allow = set(allow)
    status = git(path, "status", "--porcelain", "-uall", check=False).stdout.splitlines()
    bad = []
    for line in status:
        rel = line[3:].split(" -> ")[-1].strip().strip('"')
        full = path / rel
        if rel not in allow or full.is_symlink():
            bad.append(rel)
    if bad:
        raise SubmissionError("patch changes files outside the allowed set (or adds symlinks): %s" % ", ".join(bad[:20]))
    # Defence in depth: force build files back to the base even if one were ever added to the allowlist.
    build_files = [rel for rel in allow if any(h in rel for h in BUILD_FILE_HINTS)]
    if build_files:
        git(path, "checkout", "HEAD", "--", *build_files, check=False)


def worktree(commit, patch, allow):
    try:
        if not CLONE.exists():
            sh(["git", "clone", "-q", "-c", "core.autocrlf=false", UPSTREAM, str(CLONE)])
            git(CLONE, "config", "core.autocrlf", "false")
        git(CLONE, "fetch", "-q", "origin", check=False)
        sha = git(CLONE, "rev-parse", commit).stdout.strip()
    except RuntimeError as exc:
        raise InfraError("git: %s" % exc)
    tag = sha[:12] + ("-" + hashlib.sha256(patch.encode()).hexdigest()[:8] if patch else "")
    path = WORKTREES / tag
    if not path.exists():
        WORKTREES.mkdir(parents=True, exist_ok=True)
        try:
            git(CLONE, "worktree", "add", "-q", "--detach", str(path), sha)
        except RuntimeError as exc:
            raise InfraError("git worktree: %s" % exc)
        try:
            if patch:
                # The patch file lives beside the worktree, not in it, so it never shows up as a changed file.
                diff_file = WORKTREES / (tag + ".diff")
                diff_file.write_text(patch)
                done = git(path, "apply", "--whitespace=nowarn", str(diff_file), check=False)
                if done.returncode != 0:
                    raise SubmissionError("patch does not apply: %s" % scrub(done.stdout + done.stderr))
                if allow is not None:
                    enforce_allowlist(path, None, allow)
        except Exception:
            git(CLONE, "worktree", "remove", "--force", str(path), check=False)
            shutil.rmtree(path, ignore_errors=True)
            raise
    return sha, tag, path


def build(path, port):
    model = path / "src" / "models" / port
    binary = model / "build" / "fp16" / PORTS[port]["binary"]
    if binary.exists():
        return binary, 0.0
    # Denylist sandbox: the build can read everything except the secrets, specs, caches and git history.
    profile = "(version 1)(allow default)(deny file-read* " + " ".join(
        '(subpath "%s")' % p for p in DENY_READS) + ")"
    t0 = time.time()
    done = sh(["sandbox-exec", "-p", profile, "bash", "scripts/build.sh", "--release"], cwd=model, timeout=3600,
              check=False, env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:%s/.local/bin:/opt/homebrew/bin" % HOME,
                                "HOME": str(HOME), "NNOPT_DTYPE": "fp16"})
    (model / "build.log").write_text(done.stdout + done.stderr)
    if done.returncode != 0 or not binary.exists():
        text = done.stdout + done.stderr
        # Compiler errors first (that's what an agent needs to fix the patch), then the end of the log.
        errors = [l for l in text.splitlines() if "error" in l.lower() and "make" not in l.lower()][:25]
        raise SubmissionError("build failed:\n%s\n--- end of build log ---\n%s"
                              % (scrub("\n".join(errors), 2500), scrub(text, 600)))
    return binary, time.time() - t0


def deploy(path, port, tag, binary):
    cfg = PORTS[port]
    model = path / "src" / "models" / port
    d = "%s/%s/%s" % (DEVICE_ROOT, port, tag)
    adb("shell", "rm -rf %s && mkdir -p %s/kernels %s/weights" % (d, d, d))
    adb("push", str(binary), d + "/")
    kernels = sorted((model / "kernels").glob("*.cl"))
    if kernels:
        adb("push", *[str(k) for k in kernels], d + "/kernels/")
    links = ["weights/model.fp16.bin", "weights/model.fp16.meta.json", "weights/tokenizer_vocab.bin", "lib"]
    adb("shell", " && ".join("ln -s %s/%s %s/%s" % (cfg["deployed"], f, d, f) for f in links))
    return d


def device_run(d, port, n, prompt):
    """One run on a prompt, timed by the phone's clock in a runner the build can't touch. Tokens are printed
    on every run (NNOPT_DEBUG_LAYERS=1), so the timed run is also the checked run."""
    if not SAFE_PROMPT.match(prompt):
        raise ValueError("prompt has characters the device shell can't take: %r" % prompt)
    # /proc/uptime, not date: on the Razr, `date +%s%N` under load reported ~1 s for an 18 s run.
    cmd = ("cd %s && export LD_LIBRARY_PATH=%s/lib:/system/vendor/lib64:$LD_LIBRARY_PATH && "
           "u1=$(cut -d' ' -f1 /proc/uptime); NNOPT_DEBUG_LAYERS=1 ./%s '%s' %d --temperature 0 --seed 42 "
           "--top-k 1 > run.out 2>&1; rc=$?; u2=$(cut -d' ' -f1 /proc/uptime); echo GW_U1=$u1 GW_U2=$u2 GW_RC=$rc; "
           "cat run.out" % (d, d, PORTS[port]["binary"], prompt, n))
    out = adb("shell", cmd, timeout=900, check=False).stdout
    m = re.search(r"GW_U1=([\d.]+) GW_U2=([\d.]+) GW_RC=(\d+)", out)
    if not m:
        raise InfraError("the device run produced no result: %s" % out[-400:])
    rc = int(m.group(3))
    tokens = [int(t) for t in re.findall(r"Generated token:\s*(-?\d+)", out)]
    if rc != 0:
        raise SubmissionError("the binary exited with code %d: %s" % (rc, scrub(out, 800)))
    return {"seconds": float(m.group(2)) - float(m.group(1)), "tokens": tokens,
            "self_reported": {k: float(v) for k, v in re.findall(r"BENCHMARK (\w+): ([\d.eE+-]+)", out)}}


def device_state():
    try:
        raw = adb("shell", "cat /sys/class/thermal/thermal_zone0/temp /sys/class/power_supply/battery/capacity "
                           "/sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null; getprop ro.product.model",
                  check=False).stdout.split()
    except InfraError as exc:
        return {"error": str(exc)}
    return {"raw": raw}


def measure(port, commit, patch="", prompts=(), rounds=1, lengths=DEFAULT_LENGTHS, allow=None):
    """Build, deploy and time one version on each prompt at several generation lengths (tokens on every run).
    `allow` is the set of files the patch may change; None means no patch is expected (base/reference)."""
    lengths = sorted(set(lengths))
    result = {"port": port, "commit": commit, "patched": bool(patch), "lengths": lengths, "prompts": {}}
    d = None
    try:
        sha, tag, path = worktree(commit, patch, allow if patch else None)
        result.update({"commit": sha, "tag": tag})
        log.info("%s %s: building", port, tag)
        binary, build_s = build(path, port)
        result["build_seconds"] = round(build_s, 1)
        log.info("%s %s: built in %.0fs, waiting for the phone", port, tag, build_s)
        with phone_lock():
            if adb("get-state", check=False).stdout.strip() != "device":
                raise InfraError("phone not connected (adb get-state)")
            d = deploy(path, port, tag, binary)
            result["device_before"] = device_state()
            if prompts:
                # Discarded warm-up: the first launch after deploy compiles OpenCL kernels (and fills a binary
                # cache where the build has one), which must not land inside a timed run.
                device_run(d, port, 2, prompts[0])
            for prompt in prompts:
                by_len = {n: [] for n in lengths}
                for _ in range(rounds):
                    for n in lengths:
                        by_len[n].append(device_run(d, port, n, prompt))
                longest = max(lengths)
                result["prompts"][prompt] = {
                    # keys are str(length) so they survive JSON round-trips to the cache unchanged
                    "times": {str(n): [round(r["seconds"], 3) for r in by_len[n]] for n in lengths},
                    "median_times": {str(n): round(statistics.median(r["seconds"] for r in by_len[n]), 3) for n in lengths},
                    # greedy + deterministic, so the longest run's tokens contain every shorter run's as a prefix
                    "tokens": by_len[longest][0]["tokens"],
                    "per_length_tokens": {str(n): by_len[n][0]["tokens"] for n in lengths},
                    "self_reported": statistics.median(
                        [r["self_reported"].get("decode_tokens_per_sec") for r in by_len[longest]
                         if r["self_reported"].get("decode_tokens_per_sec")] or [0]) or None,
                }
                log.info("  %r: %s -> median times %s", prompt[:30], lengths,
                         list(result["prompts"][prompt]["median_times"].values()))
            result["device_after"] = device_state()
    except SubmissionError as exc:
        result.update({"error": str(exc)[:2000], "error_kind": "submission"})
        return result
    except InfraError as exc:
        result.update({"error": str(exc)[:2000], "error_kind": "infra"})
        return result
    finally:
        if d is not None:
            adb("shell", "rm -rf %s" % d, check=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, choices=sorted(PORTS))
    ap.add_argument("--commit", required=True)
    ap.add_argument("--patch")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--prompt", action="append", default=[])
    args = ap.parse_args()
    patch = Path(args.patch).read_text() if args.patch else ""
    print(json.dumps(measure(args.port, args.commit, patch, args.prompt, args.rounds)))


if __name__ == "__main__":
    main()
