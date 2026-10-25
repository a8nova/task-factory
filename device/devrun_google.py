"""Time adreno-llms builds on Google's real phones (Developer Device Platform "Device Run").

    uv run python device/devrun_google.py --port smollm2-135m-instruct \
        --builds base=f6e489b3326b,reference=e2076eeb9b20 \
        --devices redfin-30,dm1q-35,e1q-36,pa3q-35,frankel-36 [--rounds 2] [--prompt "..."]

One Device Run session runs every build on every listed phone in parallel. Each phone gets the builds
(binary + kernels, built on the Mac mini with devrun), the fp16 weights pulled once from the Razr (kept in
gs://<bucket>/tf-assets/<port>/), and a runner script this file writes. The runner times every process with
/proc/uptime around it, so nothing the build prints about its own speed is trusted, and writes parseable
lines to one file that Device Run pulls back (Device Run does not keep the executable's stdout).

Decode speed = 1 / slope of wall time against generated tokens over 8, 72 and 136 tokens; R^2 of that fit
is reported, since a build that is not linear in tokens is gaming the measurement.

Results: runs/device/google-<time>.json   Progress: tail -f logs/devicerun-google.log
"""
import argparse
import datetime as dt
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import RUNS, get_logger  # noqa: E402

log = get_logger("devicerun-google")
PROJECT = "mockloop-gcs-dadd"
BUCKET = "gs://mockloop-gcs-dadd-devicerun"
ASSETS = RUNS / "device" / "google-assets"
DEVICE_DIR = "/data/local/tmp/tf"
OUT_FILE = "/data/local/tmp/tf_out.txt"
LENGTHS = (8, 72, 136)
TOKEN_RUN = 40
PROMPT_TOKENS = 24
SAFE_PROMPT = re.compile(r"^[A-Za-z0-9 ,.:?()=]+$")
PORTS = {
    "smollm2-135m-instruct": {
        "binary": "SmolLM2_135M_Instruct_inference_fp16",
        "weights": ["model.fp16.bin", "model.fp16.meta.json", "tokenizer_vocab.bin"],
        "phone_weights": "/data/local/tmp/SmolLM2_135M_Instruct_inference/weights",
    },
}
DEVICES = {"redfin-30": "Pixel 5 (Adreno 620)", "dm1q-35": "Galaxy S23 (Adreno 740)",
           "e1q-36": "Galaxy S24 (Adreno 750)", "pa3q-35": "Galaxy S25 Ultra (Adreno 830)",
           "frankel-36": "Pixel 10 (PowerVR)"}


def sh(cmd, timeout=3600, check=True):
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and done.returncode != 0:
        raise RuntimeError("%s exit %d: %s" % (" ".join(cmd[:5]), done.returncode, (done.stdout + done.stderr)[-1500:]))
    return done


def gcs_exists(path):
    return sh(["gsutil", "-q", "stat", path], check=False).returncode == 0


def ensure_assets(port):
    """Weights and the fixed input, uploaded to GCS once per port."""
    cfg = PORTS[port]
    local = ASSETS / port / "weights"
    remote = "%s/tf-assets/%s" % (BUCKET, port)
    for name in cfg["weights"]:
        if not gcs_exists("%s/weights/%s" % (remote, name)):
            if not (local / name).exists():
                local.mkdir(parents=True, exist_ok=True)
                sh(["ssh", "-o", "BatchMode=yes", "macmini",
                    "~/Library/Android/sdk/platform-tools/adb pull -q %s/%s ~/task-factory/assets/%s/weights/ 2>/dev/null; "
                    "true" % (cfg["phone_weights"], name, port)])
                sh(["scp", "-q", "macmini:task-factory/assets/%s/weights/%s" % (port, name), str(local / name)])
            sh(["gsutil", "-q", "cp", str(local / name), "%s/weights/%s" % (remote, name)])
    fixture = "%s/test_input_ids.bin" % remote
    if not gcs_exists(fixture):  # the port's fixed input, identical at every commit of this port
        local_fix = ASSETS / port / "test_input_ids.bin"
        sh(["scp", "-q", "macmini:task-factory/adreno-llms/src/models/%s/reference/test_input_ids.bin" % port, str(local_fix)])
        sh(["gsutil", "-q", "cp", str(local_fix), fixture])
    return remote


def build_on_macmini(port, commit, patch=""):
    """Build commit (+ patch) with devrun on the Mac mini, pack binary + kernels, upload the tar to GCS."""
    patch_arg = ""
    if patch:
        local_patch = ASSETS / port / "patches" / ("%d.diff" % (abs(hash(patch)) % 10**10))
        local_patch.parent.mkdir(parents=True, exist_ok=True)
        local_patch.write_text(patch)
        sh(["scp", "-q", str(local_patch), "macmini:task-factory/assets/pending.diff"])
        patch_arg = "~/task-factory/assets/pending.diff"
    code = ("import devrun, json, sys, subprocess; "
            "patch = open(sys.argv[2]).read() if len(sys.argv) > 2 and sys.argv[2] else ''; "
            "sha, tag, path = devrun.worktree(sys.argv[1], patch); "
            "binary, _ = devrun.build(path, %r); "
            "model = path / 'src' / 'models' / %r; "
            "tar = devrun.BASE / 'assets' / ('build-%%s.tar' %% tag); "
            "subprocess.run(['tar', 'cf', str(tar), '-C', str(binary.parent), binary.name, '-C', str(model), 'kernels'], check=True); "
            "print(json.dumps({'sha': sha, 'tag': tag, 'tar': str(tar)}))" % (port, port))
    out = sh(["ssh", "-o", "BatchMode=yes", "macmini",
              "cd ~/task-factory && python3 -c %s %s %s" % (shlex.quote(code), commit, patch_arg)], timeout=3600)
    info = json.loads(out.stdout.strip().splitlines()[-1])
    remote = "%s/tf-assets/%s/builds/%s.tar" % (BUCKET, port, info["tag"])
    if not gcs_exists(remote):
        local = ASSETS / port / "builds" / ("%s.tar" % info["tag"])
        local.parent.mkdir(parents=True, exist_ok=True)
        sh(["scp", "-q", "macmini:%s" % info["tar"].replace(str(Path.home()) + "/", ""), str(local)])
        sh(["gsutil", "-q", "cp", str(local), remote])
        local.unlink()  # the laptop disk is small; GCS keeps the copy
    info["gcs_tar"] = remote
    return info


def runner_script(port, labels, prompts, rounds, lengths=LENGTHS):
    binary = PORTS[port]["binary"]
    lines = ["#!/system/bin/sh", "exec > %s 2>&1" % OUT_FILE, "R=%s" % DEVICE_DIR,
             "export LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64",
             'echo "TF_DEVICE model=$(getprop ro.product.model | tr \' \' _) soc=$(getprop ro.soc.model) '
             'board=$(getprop ro.board.platform) gpu=$(cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null) '
             'egl=$(getprop ro.hardware.egl) sdk=$(getprop ro.build.version.sdk)"',
             # run LABEL KIND INPUT N DEBUG ROUND [PROMPT]: one timed process, results on one TF line
             "run() {",
             "  cd $R/$1 || { echo \"TF_FAIL build=$1 reason=no_build_dir\"; return; }",
             "  if [ -n \"$7\" ]; then ARGS=\"$7\"; FIX=\"\"; else ARGS=ref; FIX=\"--token-ids ../test_input_ids.bin\"; fi",
             "  u1=$(cut -d' ' -f1 /proc/uptime)",
             "  NNOPT_DEBUG_LAYERS=$5 ./%s \"$ARGS\" $4 $FIX --temperature 0 --seed 42 --top-k 1 > run.out 2>&1" % binary,
             "  rc=$?",
             "  u2=$(cut -d' ' -f1 /proc/uptime)",
             "  toks=$(grep 'Generated token:' run.out | sed 's/.*Generated token: *//' | tr '\\n' ',')",
             "  bench=$(grep '^BENCHMARK' run.out | sed 's/^BENCHMARK //;s/ //g' | tr '\\n' ';')",
             "  echo \"TF build=$1 kind=$2 input=$3 n=$4 round=$6 rc=$rc u1=$u1 u2=$u2 tokens=$toks bench=$bench\"",
             "  if [ $rc -ne 0 ]; then tail -6 run.out | sed 's/^/TF_TAIL build='\"$1\"' /'; fi",
             "}"]
    for label in labels:
        lines += ["mkdir -p $R/%s && cd $R/%s && tar xf $R/%s.tar && ln -sf ../weights weights" % (label, label, label)]
    for label in labels:  # token runs; the first one also warms each build's kernel cache
        lines.append("run %s tokens fixed %d 1 0" % (label, TOKEN_RUN))
        for i, p in enumerate(prompts):
            lines.append("run %s tokens p%d %d 1 0 '%s'" % (label, i, PROMPT_TOKENS, p))
    for r in range(rounds):
        for n in lengths:
            for label in labels:  # alternate builds so drift and heat hit both
                lines.append("run %s time fixed %d 0 %d" % (label, n, r))
    lines.append('echo "TF_DONE"')
    return "\n".join(lines) + "\n"


def fit(points):
    """Least squares seconds = a + b * tokens; returns tokens/s (1/b) and R^2."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return (1.0 / b if b > 0 else None), (1 - ss_res / ss_tot if ss_tot else None), a


def parse(text, labels, prompts, lengths=LENGTHS):
    dev = {}
    m = re.search(r"TF_DEVICE (.*)", text)
    if m:
        dev = dict(kv.split("=", 1) for kv in re.findall(r"(\w+=\S*)", m.group(1)))
    builds = {}
    for label in labels:
        res = {"tokens": [], "prompt_tokens": {}, "token_run_rc": None, "timing": [], "self_reported": [],
               "failures": [l for l in re.findall(r"TF_TAIL build=%s (.*)" % re.escape(label), text)]}
        for line in re.findall(r"^TF build=%s (.*)$" % re.escape(label), text, re.M):
            kv = dict(re.findall(r"(\w+)=(\S*)", line))
            toks = [int(t) for t in kv.get("tokens", "").split(",") if t.strip().lstrip("-").isdigit()]
            rc = int(kv["rc"]) if kv.get("rc", "").isdigit() else None
            seconds = float(kv["u2"]) - float(kv["u1"]) if kv.get("u1") and kv.get("u2") else None
            bench = dict(b.split(":", 1) for b in kv.get("bench", "").split(";") if ":" in b)
            if kv["kind"] == "tokens":
                if kv["input"] == "fixed":
                    res["tokens"], res["token_run_rc"] = toks, rc
                else:
                    res["prompt_tokens"][prompts[int(kv["input"][1:])]] = toks
            else:
                res["timing"].append({"n": int(kv["n"]), "round": int(kv["round"]), "seconds": seconds, "rc": rc})
                if "decode_tokens_per_sec" in bench:
                    res["self_reported"].append(float(bench["decode_tokens_per_sec"]))
        ok = [t for t in res["timing"] if t["rc"] == 0 and t["seconds"] is not None]
        if len({t["n"] for t in ok}) >= 2:
            tps, r2, startup = fit([(t["n"], t["seconds"]) for t in ok])
            res.update({"decode_tok_s_external": round(tps, 3) if tps else None,
                        "linearity_r2": round(r2, 5) if r2 is not None else None,
                        "startup_seconds": round(startup, 2),
                        "short_seconds": [round(t["seconds"], 3) for t in ok if t["n"] == min(lengths)],
                        "long_seconds": [round(t["seconds"], 3) for t in ok if t["n"] == max(lengths)]})
        else:
            res["decode_tok_s_external"] = None
            res["error"] = "timing runs failed: %s" % ("; ".join(res["failures"][-3:]) or "no successful timing runs")
        if res["self_reported"]:
            res["decode_tok_s_self_reported"] = round(statistics.median(res["self_reported"]), 3)
        if res["token_run_rc"] not in (0, None) or (res["token_run_rc"] == 0 and not res["tokens"]):
            res["error"] = res.get("error") or "token run failed: %s" % "; ".join(res["failures"][-3:])
        builds[label] = res
    return {"device": dev, "builds": builds, "complete": "TF_DONE" in text}


def measure_google(port, builds, devices, prompts=(), rounds=2, timeout_min=20, lengths=LENGTHS):
    """builds: {label: {"gcs_tar": gs://... tar of binary + kernels/}} (see build_on_macmini).
    Returns {"session", "devices": {device_id: {"device": {...}, "builds": {label: result}, "complete"}},
    "device_minutes_estimate"}. On fast phones (Adreno 8xx, ~110 tok/s) 136 tokens is ~1 s, smaller than
    launch noise, so pass longer lengths there, e.g. (32, 256, 512). Each result: tokens, prompt_tokens, token_run_rc, timing, decode_tok_s_external,
    linearity_r2, startup_seconds, short_seconds, long_seconds, decode_tok_s_self_reported, error."""
    for p in prompts:
        if not SAFE_PROMPT.match(p):
            raise ValueError("prompt has characters the device shell can't take: %r" % p)
    labels = list(builds)
    assets = ensure_assets(port)
    work = RUNS / "device" / "google-sessions"
    work.mkdir(parents=True, exist_ok=True)
    script = work / ("runner-%s.sh" % dt.datetime.now().strftime("%H%M%S"))
    script.write_text(runner_script(port, labels, list(prompts), rounds, lengths))
    script.chmod(0o755)
    push = ["%s/weights/%s=%s/weights/%s" % (assets, w, DEVICE_DIR, w) for w in PORTS[port]["weights"]]
    push += ["%s=%s/%s.tar" % (b["gcs_tar"], DEVICE_DIR, label) for label, b in builds.items()]
    push.append("%s/tf-assets/%s/test_input_ids.bin=%s/test_input_ids.bin" % (BUCKET, port, DEVICE_DIR))
    cmd = ["gcloud", "beta", "device-run", "sessions", "submit", "android-executable", "--project", PROJECT,
           "--executable", str(script), "--executable-timeout=%dm" % timeout_min,
           "--other-files-to-push=" + ",".join(push), "--paths-to-pull=" + OUT_FILE, "--async"]
    for d in devices:
        cmd += ["--device", d]
    t0 = time.time()
    out = sh(cmd, timeout=900)
    session = re.search(r"session-[0-9a-f-]+", out.stdout + out.stderr).group(0)
    log.info("session %s: %d builds x %d devices, waiting", session, len(labels), len(devices))
    sh(["gcloud", "beta", "device-run", "sessions", "wait", session, "--project", PROJECT], timeout=60 * (timeout_min + 30),
       check=False)
    log.info("session %s finished after %.0fs", session, time.time() - t0)
    files = sh(["gsutil", "ls", "-r", "%s/automation/sessions/%s/**" % (BUCKET, session)], check=False).stdout.split()
    result = {"session": session, "devices": {}, "seconds": round(time.time() - t0)}
    for i, d in enumerate(devices):
        outs = [f for f in files if "/job-%03d/" % i in f and f.endswith("tf_out.txt")]
        text = sh(["gsutil", "cat", outs[0]]).stdout if outs else ""
        parsed = parse(text, labels, list(prompts), lengths) if text else {"device": {}, "builds": {}, "complete": False,
                                                                     "error": "no output pulled back"}
        parsed["device_id"] = d
        result["devices"][d] = parsed
    report = sh(["gcloud", "beta", "device-run", "sessions", "describe", session, "--project", PROJECT, "--full"],
                check=False).stdout
    starts = re.findall(r"startTime: '([^']+)'", report)
    ends = re.findall(r"endTime: '([^']+)'", report)
    try:
        parse_t = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        job_minutes = sum((parse_t(e) - parse_t(s)).total_seconds() for s, e in zip(starts[1:], ends[1:])) / 60
        result["device_minutes_estimate"] = round(job_minutes / 2, 1)  # each job appears with its execution
    except (ValueError, IndexError):
        result["device_minutes_estimate"] = None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, choices=sorted(PORTS))
    ap.add_argument("--builds", required=True, help="label=commit[,label=commit...]")
    ap.add_argument("--devices", required=True, help="comma-separated Device Run ids")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--timeout-min", type=int, default=20)
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)), help="generated-token counts for timing runs")
    args = ap.parse_args()
    builds = {}
    for item in args.builds.split(","):
        label, commit = item.split("=", 1)
        builds[label] = build_on_macmini(args.port, commit)
        log.info("build %s = %s -> %s", label, builds[label]["tag"], builds[label]["gcs_tar"])
    result = measure_google(args.port, builds, args.devices.split(","), args.prompt, args.rounds, args.timeout_min,
                            tuple(int(n) for n in args.lengths.split(",")))
    result.update({"port": args.port, "builds": builds})
    out = RUNS / "device" / ("google-%s.json" % dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.write_text(json.dumps(result, indent=1))
    for d, r in result["devices"].items():
        row = {k: (v.get("decode_tok_s_external"), v.get("linearity_r2"), v.get("error")) for k, v in r["builds"].items()}
        log.info("%s %s: %s", d, r["device"].get("gpu") or r["device"].get("model"), row)
    log.info("result -> %s", out)


if __name__ == "__main__":
    main()
