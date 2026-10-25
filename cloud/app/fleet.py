"""Time builds on Google's Device Run phones in one session, from a runner script the gateway owns.

Every timed run prints its tokens, and every one is checked (all rounds at a length must agree), so the timed
run is also the checked run. Builds that share prompts alternate inside the session in ABBA order, so a slow
phone or heat hits base and candidate alike. Timing uses /proc/uptime around each process; nothing the build
prints about its own speed is trusted. One session can carry several tasks' builds, each with its own prompts.
"""
import re
import statistics
import subprocess
import time
import uuid

PROJECT = "mockloop-gcs-dadd"
BUCKET = "gs://mockloop-gcs-dadd-devicerun"
ASSETS = BUCKET + "/tf-assets"
DEVICE_DIR = "/data/local/tmp/tf"
OUT_FILE = "/data/local/tmp/tf_out.txt"
SAFE_PROMPT = re.compile(r"^[A-Za-z0-9 ,.:?()=]+$")
SAFE_ENV_KEY = re.compile(r"^(NNOPT|BONSAI)_[A-Z0-9_]+$")
SAFE_ENV_VALUE = re.compile(r"^[A-Za-z0-9_.-]*$")
SAFE_ARG = re.compile(r"^--[a-z][a-z0-9-]*$")
SAFE_WEIGHT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_HF_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
WEIGHTS = ["model.fp16.bin", "model.fp16.meta.json", "tokenizer_vocab.bin"]
HF_BASE = "https://huggingface.co/a8nova/adreno-llms-weights/resolve/main"
# The weight mirror's folder per port (scripts/fetch_weights.sh hf_subdir_for); others use the port name.
HF_SUBDIR = {"qwen2-5-0-5b": "qwen2-5-0-5b-instruct", "lfm2-5-350m": "lfm2-5-350m-instruct",
             "openelm-270m": "openelm-270m-instruct"}


def weight_files(port, extra=(), override=None):
    """[(file name on the phone under weights/, path in the public mirror)]: the port's fp16 weights plus any
    extra bundle (e.g. model.q4.bin), or a task's own list (e.g. Bonsai's .nnb and tokenizer.json)."""
    if override:
        files = sorted(override.items())
    else:
        files = [(w, "%s/%s" % (HF_SUBDIR.get(port, port), w)) for w in WEIGHTS + sorted(set(extra))]
    for name, src in files:
        if not SAFE_WEIGHT.match(name) or not SAFE_HF_PATH.match(src) or ".." in src.split("/"):
            raise ValueError("weight file %s (%s) is not allowed" % (name, src))
    return files


def ensure_assets(port, files):
    """Copy each weight file once from the public mirror into Cloud Storage."""
    for name, src in files:
        dst = "%s/%s/weights/%s" % (ASSETS, port, name)
        if sh(["gsutil", "-q", "stat", dst], timeout=120).returncode == 0:
            continue
        url = "%s/%s" % (HF_BASE, src)
        done = subprocess.run("curl -sfL --retry 3 '%s' | gsutil -q cp - '%s'" % (url, dst), shell=True,
                              capture_output=True, text=True, timeout=3600)
        if done.returncode != 0 or sh(["gsutil", "-q", "stat", dst], timeout=120).returncode != 0:
            raise FleetError("could not copy %s weights %s from the mirror" % (port, name))


class FleetError(Exception):
    """Device Run or the phone failed: an infrastructure error, never a reward."""


def sh(cmd, timeout=3600, inp=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=inp)


def runner(port, builds, lengths, rounds, binaries, envs=None, args=None, warmups=1, texts=None, gen_tokens=None,
           cli=None):
    """builds: {label: [prompts]} (labels sharing a prompt list are paired); binaries: {label: file name};
    envs: {label: {NNOPT_*: value}} set for that build's runs only (e.g. NNOPT_QUANT=q4); args: {label: [flags]}
    appended to that build's command line (e.g. --ignore-eos); warmups: discarded warm-up rounds per build.
    Prefill timing: texts {label: {prompt key: {length: prompt text}}} runs a different prompt per length, each
    generating gen_tokens tokens. cli: "bonsai" for Bonsai's `<nnb> <tokenizer> gen <prompt> <n>` command line."""
    envs, args, texts = envs or {}, args or {}, texts or {}
    lines = ["#!/system/bin/sh", "exec > %s 2>&1" % OUT_FILE, "R=%s" % DEVICE_DIR,
             "export LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64",
             'echo "TF_DEVICE model=$(getprop ro.product.model | tr \' \' _) soc=$(getprop ro.soc.model) '
             'gpu=$(cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null) sdk=$(getprop ro.build.version.sdk)"',
             "run() {",  # run LABEL PROMPT_INDEX N ROUND PROMPT [TOKENS_TO_GENERATE, default N]
             "  cd $R/$1 || { echo \"TF_FAIL build=$1 reason=no_build_dir\"; return; }",
             "  B=$(cat $R/$1/.binary); M=${6:-$3}",
             "  u1=$(cut -d' ' -f1 /proc/uptime)",
             "  ( [ -f .env ] && . ./.env; export LD_LIBRARY_PATH=$R/$1:/vendor/lib64:/system/vendor/lib64 NNOPT_DEBUG_LAYERS=1; "
             "if [ \"$(cat .cli 2>/dev/null)\" = bonsai ]; then "
             "./$B model/$BONSAI_NNB model/tokenizer.json gen \"$5\" $M $(cat .args 2>/dev/null) > run.out 2>&1; "
             "else ./$B \"$5\" $M --temperature 0 --seed 42 --top-k 1 $(cat .args 2>/dev/null) > run.out 2>&1; fi )",
             "  rc=$?",
             "  u2=$(cut -d' ' -f1 /proc/uptime)",
             "  toks=$(grep 'Generated token:' run.out | sed 's/.*Generated token: *//' | tr '\\n' ',')",
             "  [ -z \"$toks\" ] && toks=$(grep '^output_ids:' run.out | head -1 | sed 's/^output_ids: *//' | tr ' ' ',')",
             "  pn=$(grep '^prompt_ids:' run.out | head -1 | wc -w)",
             "  bench=$(grep '^BENCHMARK decode_tokens_per_sec' run.out | sed 's/.*: *//' | head -1)",
             "  pbench=$(grep '^BENCHMARK prefill_tokens_per_sec' run.out | sed 's/.*: *//' | head -1)",
             "  echo \"TF build=$1 p=$2 n=$3 round=$4 rc=$rc u1=$u1 u2=$u2 tokens=$toks bench=$bench pbench=$pbench pn=$pn\"",
             "  if [ \"$2\" = w ] && [ \"$4\" = 0 ]; then echo \"TF_GPU build=$1 cl=$(grep -m1 -i -E '^opencl: |adreno' run.out "
             "| head -c 120 | tr -c 'A-Za-z0-9().-' '_')\"; fi",
             "  if [ $rc -ne 0 ]; then tail -4 run.out | sed 's/^/TF_TAIL build='\"$1\"' /'; fi",
             "}"]
    labels = list(builds)
    for label in labels:
        lines.append("mkdir -p $R/%s && cd $R/%s && tar xf $R/%s.tar && ln -sf ../weights weights && ln -sf ../weights model "
                     "&& echo %s > .binary" % (label, label, label, binaries[label]))
        if cli:
            if cli != "bonsai":
                raise ValueError("unknown command line style %s" % cli)
            lines.append("echo %s > $R/%s/.cli" % (cli, label))
        for k, v in sorted((envs.get(label) or {}).items()):
            if not (SAFE_ENV_KEY.match(k) and SAFE_ENV_VALUE.match(str(v))):
                raise ValueError("run env %s=%s is not allowed" % (k, v))
            lines.append("echo 'export %s=%s' >> $R/%s/.env" % (k, v, label))
        flags = list(args.get(label) or [])
        if not all(SAFE_ARG.match(str(a)) for a in flags):
            raise ValueError("run args %s are not allowed" % flags)
        if flags:
            lines.append("echo '%s' > $R/%s/.args" % (" ".join(flags), label))
    def prompt(label, key, n):
        text = texts[label][key][str(n)] if label in texts else key
        if not SAFE_PROMPT.match(text):
            raise ValueError("prompt has characters the device shell can't take")
        return text

    # discarded warm-ups: the first launches compile kernels, fill a binary cache and pull weights into memory
    for w in range(max(1, int(warmups))):
        for label in labels:
            n = min(lengths)
            lines.append("run %s w %d %d '%s' %d" % (label, n, w, prompt(label, builds[label][0], n),
                                                     2 if w == 0 else (gen_tokens or n)))
    step = 0
    for i in range(max(len(p) for p in builds.values())):
        for r in range(rounds):
            for n in lengths:
                order = labels if step % 2 == 0 else list(reversed(labels))  # ABBA
                step += 1
                for label in order:
                    if i < len(builds[label]):
                        lines.append("run %s %d %d %d '%s' %d" % (label, i, n, r, prompt(label, builds[label][i], n),
                                                                  gen_tokens if label in texts else n))
    lines.append('echo "TF_DONE"')
    return "\n".join(lines) + "\n"


def parse(text, builds, lengths):
    out = {"device": {}, "builds": {}, "complete": "TF_DONE" in text}
    m = re.search(r"TF_DEVICE (.*)", text)
    if m:
        out["device"] = dict(kv.split("=", 1) for kv in re.findall(r"(\w+=\S*)", m.group(1)))
    m = re.search(r"^TF_GPU build=\S+ cl=(\S*)", text, re.M)  # the OpenCL device name the binary printed
    if m and m.group(1).strip("_"):
        out["device"]["opencl_device"] = m.group(1).strip("_").replace("_", " ")
    for label, prompts in builds.items():
        runs = {}
        failures = re.findall(r"TF_TAIL build=%s (.*)" % re.escape(label), text)
        bad_rc = False
        for line in re.findall(r"^TF build=%s (p=\d+ .*)$" % re.escape(label), text, re.M):
            kv = dict(re.findall(r"(\w+)=(\S*)", line))
            rc = int(kv["rc"]) if kv.get("rc", "").isdigit() else None
            if rc != 0:
                bad_rc = True
            toks = [int(t) for t in kv.get("tokens", "").split(",") if t.strip().lstrip("-").isdigit()]
            secs = float(kv["u2"]) - float(kv["u1"]) if kv.get("u1") and kv.get("u2") else None
            bench = float(kv["bench"]) if re.match(r"^[\d.eE+-]+$", kv.get("bench", "")) else None
            pbench = float(kv["pbench"]) if re.match(r"^[\d.eE+-]+$", kv.get("pbench", "")) else None
            pn = int(kv["pn"]) - 1 if kv.get("pn", "").isdigit() and int(kv["pn"]) > 0 else None  # minus the label
            runs.setdefault(int(kv["p"]), {}).setdefault(int(kv["n"]), []).append((secs, toks, bench, pbench, pn))
        res = {"prompts": {}}
        if bad_rc:
            res.update({"error": "the binary failed on the phone: %s" % "; ".join(failures[-3:])[:800],
                        "error_kind": "submission"})
        for i, p in enumerate(prompts):
            by_len = runs.get(i, {})
            if sorted(by_len) != sorted(lengths):
                res.setdefault("error", "missing runs for a prompt on the phone")
                res.setdefault("error_kind", "submission" if out["complete"] else "infra")
                continue
            longest = max(lengths)
            res["prompts"][p] = {
                "times": {str(n): [round(r[0], 3) for r in by_len[n]] for n in lengths},
                "median_times": {str(n): round(statistics.median(r[0] for r in by_len[n]), 3) for n in lengths},
                "tokens": by_len[longest][0][1],
                "per_length_tokens": {str(n): by_len[n][0][1] for n in lengths},
                # every timed run is checked: all rounds at one length must print the same tokens
                "tokens_consistent": all(all(r[1] == by_len[n][0][1] for r in by_len[n]) for n in lengths),
                "self_reported": statistics.median([r[2] for r in by_len[longest] if r[2]] or [0]) or None,
                "self_reported_prefill": statistics.median([r[3] for r in by_len[longest] if r[3]] or [0]) or None,
                # prompt length in tokens as the binary tokenized it (trusted only from trusted builds)
                "prompt_tokens": {str(n): by_len[n][0][4] for n in lengths},
            }
        out["builds"][label] = res
    return out


def measure(port, builds, lengths, rounds=1, device="redfin-30", timeout_min=40, extra_weights=(), warmups=1,
            weights=None, gen_tokens=None, cli=None, fallback_devices=()):
    """builds: {label: {"tar": gs:// tar, "prompts": [...], "binary": name, "env": {...}, "args": [...],
    "texts": {prompt key: {length: text}} for prefill timing}}. One Device Run session on one phone model (the
    first of device + fallback_devices that starts a session). extra_weights: weight files beyond fp16 to push
    (e.g. model.q4.bin); weights: a task's own {file name: mirror path} instead of the port's fp16 set.
    Returns parse() output plus {"session", "seconds", "device_id"}."""
    files = weight_files(port, extra_weights, weights)
    plan = {label: list(b["prompts"]) for label, b in builds.items()}
    texts = {label: b["texts"] for label, b in builds.items() if b.get("texts")}
    for label, ps in plan.items():
        for p in ps:
            if label not in texts and not SAFE_PROMPT.match(p):
                raise ValueError("prompt has characters the device shell can't take")
    lengths = sorted(set(int(n) for n in lengths))
    # The runner goes to our own prefix and the bucket is named explicitly, so gcloud never needs project-wide
    # bucket listing (the service account only has rights on this one bucket).
    runner_gcs = "%s/tf-gateway/runners/%s.sh" % (BUCKET, uuid.uuid4().hex)
    ensure_assets(port, files)
    up = sh(["gsutil", "-q", "cp", "-", runner_gcs], timeout=300,
            inp=runner(port, plan, lengths, rounds, {label: b["binary"] for label, b in builds.items()},
                       {label: b.get("env") for label, b in builds.items()},
                       {label: b.get("args") for label, b in builds.items()}, warmups, texts, gen_tokens, cli))
    if up.returncode != 0:
        raise FleetError("could not upload the runner: %s" % up.stderr[-400:])
    push = ["%s/%s/weights/%s=%s/weights/%s" % (ASSETS, port, w, DEVICE_DIR, w) for w, _ in files]
    push += ["%s=%s/%s.tar" % (b["tar"], DEVICE_DIR, label) for label, b in builds.items()]
    t0 = time.time()
    tried = []
    for dev in [device] + [d for d in fallback_devices if d != device]:
        cmd = ["gcloud", "beta", "device-run", "sessions", "submit", "android-executable", "--project", PROJECT,
               "--device", dev, "--executable", runner_gcs, "--bucket-name", BUCKET.replace("gs://", ""),
               "--executable-timeout=%dm" % timeout_min,
               "--other-files-to-push=" + ",".join(push), "--paths-to-pull=" + OUT_FILE, "--async"]
        done = sh(cmd, timeout=900)
        m = re.search(r"session-[0-9a-f-]+", done.stdout + done.stderr)
        if m:
            device = dev
            break
        tried.append("%s: %s" % (dev, (done.stdout + done.stderr)[-300:]))
    if not m:
        raise FleetError("Device Run session did not start: %s" % " | ".join(tried)[-900:])
    session = m.group(0)
    sh(["gcloud", "beta", "device-run", "sessions", "wait", session, "--project", PROJECT], timeout=60 * (timeout_min + 30))
    files = sh(["gsutil", "ls", "-r", "%s/automation/sessions/%s/**" % (BUCKET, session)]).stdout.split()
    if not any(f.endswith("tf_out.txt") for f in files):
        files = sh(["gsutil", "ls", "-r", "%s/**/%s/**" % (BUCKET, session)]).stdout.split()
    outs = [f for f in files if f.endswith("tf_out.txt")]
    if not outs:
        raise FleetError("no output came back from the phone (session %s)" % session)
    text = sh(["gsutil", "cat", outs[0]]).stdout
    result = parse(text, plan, lengths)
    result.update({"session": session, "seconds": round(time.time() - t0), "device_id": device})
    if not result["complete"] and not any(b.get("error_kind") == "submission" for b in result["builds"].values()):
        raise FleetError("the phone run did not finish (session %s)" % session)
    return result
