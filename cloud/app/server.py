"""Task Factory cloud gateway: the trusted grader on Cloud Run, timing builds on Google's Device Run phones.

    GET  /health                          no auth; service state only
    POST /try        {task, patch}        X-TF-Agent:    base + patch on the development prompts
    POST /grade      {task, patch}        X-TF-Verifier: the hidden check on the hidden prompts
    POST /calibrate  {task | tasks, force?}   X-TF-Factory: base and reference on the fleet; several tasks of one
                                              port share one Device Run session
    POST /set_target {task, target_speedup, source}   X-TF-Factory
    POST /measure    {port, commit, patch?, prompts, lengths?}   X-TF-Factory: any commit

An optional X-TF-Caller header (an agent id) allows one request in flight per caller; tasks never block
each other. Rules as in gateway v2 (device/gateway.py): valid only if, at every generated length and in
every round, the candidate's tokens exactly match a trusted implementation's tokens (valid_token_sources);
decode speed is the slope of a line through several lengths, timed on the phone's clock, with a residual
check against that line; base and candidate are timed in the same Device Run session (ABBA); reward is
clip((speedup - 1) / (target - 1), 0, 1) when valid, else 0; infrastructure failures never produce a reward;
every response is signed (HMAC-SHA256). Keys and hidden task specs come from Secret Manager and stay in this
process's memory.
"""
import hashlib
import hmac
import json
import os
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import builder
import fleet

PROJECT = "mockloop-gcs-dadd"
CACHE = "%s/%s/cache" % (builder.BUCKET, builder.PREFIX)
TIMER_TOLERANCE = 1.25
RESIDUAL_FRACTION, RESIDUAL_SECONDS = 0.12, 0.5
MIN_R2 = 0.98
CAL_LENGTHS = (8, 40, 150)
GRADE_LENGTHS = (8, 40, 136)
MIN_HIDDEN, MIN_DEV, MAX_HIDDEN, MAX_DEV = 3, 2, 3, 2
AUTH = {"/try": ("X-TF-Agent", "tf-gateway-agent-key"), "/grade": ("X-TF-Verifier", "tf-gateway-verifier-key"),
        "/calibrate": ("X-TF-Factory", "tf-gateway-factory-key"), "/set_target": ("X-TF-Factory", "tf-gateway-factory-key"),
        "/measure": ("X-TF-Factory", "tf-gateway-factory-key"), "/public": ("X-TF-Factory", "tf-gateway-factory-key")}
_secrets, _tasks, _cache = {}, {}, {}
_build_locks = {}
TASK_TTL = 30
_task_locks, _in_flight = {}, set()
_guard = threading.Lock()
state = {"served": 0, "running": 0, "started": time.time()}


def log(msg, *args):
    print(time.strftime("%H:%M:%S") + " " + (msg % args if args else msg), flush=True)


def secret(name):
    if name not in _secrets:
        done = subprocess.run(["gcloud", "secrets", "versions", "access", "latest", "--secret", name,
                               "--project", PROJECT], capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise KeyError("secret %s unavailable" % name)
        _secrets[name] = done.stdout.strip()
    return _secrets[name]


def load_task(name):
    # Re-read the spec every TASK_TTL seconds: Cloud Run runs several instances, and a calibration or
    # /set_target on one of them must reach the others.
    if name not in _tasks or time.time() - _tasks[name][0] > TASK_TTL:
        _secrets.pop("tf-task-%s" % name, None)
        try:
            _tasks[name] = (time.time(), json.loads(secret("tf-task-%s" % name)))
        except KeyError:
            raise KeyError("unknown task %r" % name)
    return _tasks[name][1]


def save_task(name, task):
    done = subprocess.run(["gcloud", "secrets", "versions", "add", "tf-task-%s" % name, "--data-file=-",
                           "--project", PROJECT], input=json.dumps(task), capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError("could not save the task spec")
    _tasks[name] = (time.time(), task)
    _secrets.pop("tf-task-%s" % name, None)


def cache_get(key):
    # calibrations can be redone on another instance, so they expire like task specs
    if key not in _cache or (key.startswith("cal-") and time.time() - _cache[key][0] > TASK_TTL):
        done = subprocess.run(["gsutil", "-q", "cat", "%s/%s.json" % (CACHE, key)], capture_output=True, text=True)
        if done.returncode != 0:
            return None
        _cache[key] = (time.time(), json.loads(done.stdout))
    return _cache[key][1]


def cache_put(key, value):
    subprocess.run(["gsutil", "-q", "cp", "-", "%s/%s.json" % (CACHE, key)], input=json.dumps(value),
                   capture_output=True, text=True, check=True)
    _cache[key] = (time.time(), value)


def signed(result):
    body = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return {"result": result,
            "signature": hmac.new(secret("tf-gateway-hmac").encode(), body.encode(), hashlib.sha256).hexdigest()}


def median(xs):
    xs = [x for x in xs if x]
    return statistics.median(xs) if xs else None


def linfit(points):
    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    if sxx == 0 or sxy <= 0:
        return None, 0.0
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in points)
    ss_tot = sum((y - my) ** 2 for _, y in points)
    return 1.0 / b, (1 - ss_res / ss_tot if ss_tot > 0 else 1.0)


TOKENS_PER_WORD = 1.3  # prefill prompts are cut by words; their real token counts come from the trusted base


def prefill(task):
    return task.get("metric") == "prefill_tok_s"


def xs_of(pr, xref=None):
    """The x of each timed length: generated tokens (decode), or the prompt's token count as the trusted base
    build tokenized it (prefill: xref is the base's runs of the same prompt in the same session)."""
    if xref is None:
        return {n: int(n) for n in pr["median_times"]}
    return {n: (xref.get("prompt_tokens") or {}).get(n) for n in pr["median_times"]}


def decode_of(pr, xref=None):
    """(tokens per second, R^2) from the time-vs-length fit: decode tok/s, or prefill tok/s when xref is given."""
    xs = xs_of(pr, xref)
    if any(x is None for x in xs.values()):
        return None, 0.0
    return linfit([(xs[n], t) for n, t in pr["median_times"].items()])


def prompt_texts(task, key, lengths):
    """Prefill prompts: the first words of one passage of the pool, one cut per length (about that many tokens)."""
    words = task["prompt_pool"][int(key.split()[-1]) - 1].split()
    out = {}
    for n in lengths:
        k = max(1, int(round(int(n) / TOKENS_PER_WORD)))
        out[str(n)] = " ".join((words * (k // len(words) + 1))[:k])
    return out


def prompt_keys(task):
    if prefill(task):
        return ["passage %d" % (i + 1) for i in range(len(task["prompt_pool"]))]
    return list(task.get("prompt_pool") or (task.get("hidden_prompts", []) + task.get("dev_prompts", [])))


def worst_residual(pr, xref=None):
    """The run farthest from the fitted time-vs-length line. A uniform startup cost shifts every point equally
    and is fine; a delay that only fires at some lengths shows up here."""
    xs = xs_of(pr, xref)
    pts = sorted((xs[n], t) for n, t in pr["median_times"].items())
    mx, my = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / max(sum((x - mx) ** 2 for x, _ in pts), 1e-9)
    x, y, r = max(((x, y, y - (my + slope * (x - mx))) for x, y in pts), key=lambda v: abs(v[2]))
    return {"length": x, "seconds": y, "residual": round(r, 3),
            "anomaly": abs(r) > max(RESIDUAL_FRACTION * y, RESIDUAL_SECONDS)}


def trusted_build(port, commit):
    """Base and reference builds are made once and kept in Cloud Storage (one build at a time per commit)."""
    key = "build2-%s-%s" % (port, commit[:12])
    with _build_locks.setdefault(key, threading.Lock()):
        info = cache_get(key)
        if not info or "binary" not in info:
            info = builder.build_version(port, commit)
            cache_put(key, info)
    return info


def stable(pr, task=None):
    """Every round agreed and every run produced all its tokens (n for decode, gen_tokens for prefill)."""
    want = (lambda n: int(task.get("gen_tokens", 8))) if task and prefill(task) else int
    return pr.get("tokens_consistent", True) and all(len(v) == want(n) for n, v in pr["per_length_tokens"].items())


def select_prompts(task, base, ref, prompts):
    """Keep prompts whose base and reference tokens differ (unchanged base code then fails correctness on
    them) and that produced every token; split into hidden and development sets. A spec task has no reference,
    so every complete prompt counts."""
    ok = lambda p: stable(ref["prompts"][p], task) and stable(base["prompts"][p], task)  # noqa: E731
    weaker = None
    if not task.get("reference"):
        kept = [p for p in prompts if ok(p)]
        weaker = "spec task: no reference, so prompts are not checked for telling base and reference apart"
    else:
        kept = [p for p in prompts if base["prompts"][p]["tokens"] != ref["prompts"][p]["tokens"] and ok(p)]
        need = task.get("min_hidden", MIN_HIDDEN) + task.get("min_dev", MIN_DEV)
        if len(kept) < need:
            if task.get("require_discriminating_prompts", True):
                raise RuntimeError("only %d of %d prompts tell base and reference apart on the fleet" % (len(kept), len(prompts)))
            weaker = "only %d prompts tell base and reference apart; using every complete prompt" % len(kept)
            # the ones that do tell them apart go first, so they land among the hidden prompts
            kept = kept + [p for p in prompts if p not in kept and ok(p)]
    n_dev = min(task.get("max_dev", MAX_DEV), max(1, len(kept) - task.get("min_hidden", 1)))
    dev = kept[-n_dev:]
    hidden = kept[:-n_dev][:task.get("max_hidden", MAX_HIDDEN)]
    return hidden, dev, kept, weaker


def calibrate_many(names, force=False, batch=False):
    """Calibrate several tasks on every device in their specs. Default: one Device Run session per task and
    device, all in parallel (sessions run concurrently on separate phones). batch=True puts the tasks of one
    port into one session per device (saves a provisioning minute and phones; the slowest task sets the pace).
    Prompt selection and the target come from each task's primary device (devices[0]); every other device gets
    a device check: its own base and reference speed and whether its tokens match the primary's."""
    done, todo = {}, []
    for name in names:
        task = load_task(name)
        cal = None if force else cache_get("cal-%s-%s" % (name, task.get("devices", ["redfin-30"])[0]))
        if cal and "target_speedup" not in task:
            # spec re-registered after calibration: restore what calibration set
            task.update({"hidden_prompts": cal["hidden_prompts"], "dev_prompts": cal["dev_prompts"],
                         "calibrated_on": task.get("devices", ["redfin-30"])[0],
                         "target_speedup": round(cal["reference_decode"] / cal["base_decode"], 3),
                         "target_source": "reference"})
            save_task(name, task)
        if cal:
            done[name] = (task, cal)
        else:
            todo.append(name)
    jobs = {}
    for name in todo:
        t = load_task(name)
        for device in t.get("devices", ["redfin-30"]):
            key = (t["port"], device) + (() if batch else (name,))
            jobs.setdefault(key, []).append(name)
    raws = {}
    with ThreadPoolExecutor(max(1, len(jobs))) as pool:
        for result in pool.map(lambda kv: _measure_group(kv[0][0], kv[0][1], kv[1]), jobs.items()):
            raws.update(result)
    for name in todo:
        done[name] = _finalize(name, raws)
    return done


def run_env(task, side):
    """Environment for one side's runs on the phone (side: base, reference or candidate), e.g. NNOPT_QUANT=q4."""
    env = task.get("run_env") or {}
    return dict(env.get("all") or {}, **(env.get(side) or {}))


def run_args(task):
    """Extra command-line flags for every side's runs (e.g. --ignore-eos so every prompt decodes to full length)."""
    return list(task.get("run_args") or [])


def extra_weights(task):
    """Weight bundles beyond fp16 the task's runs need on the phone, as file names under weights/."""
    return [w.split("/")[-1] for w in task.get("extra_weights", [])]


def sides(task):
    return ("base", "reference") if task.get("reference") else ("base",)


def run_opts(task):
    """fleet.measure options that come from the task (the same for every build in its sessions)."""
    return {"weights": task.get("weights"), "cli": task.get("cli"), "fallback_devices": task.get("fallback_devices", ()),
            "gen_tokens": int(task.get("gen_tokens", 8)) if prefill(task) else None, "warmups": task.get("warmups", 1)}


def build_entry(task, b, prompts, side, lengths):
    entry = {"tar": b["gcs_tar"], "binary": b["binary"], "prompts": prompts, "env": run_env(task, side),
             "args": run_args(task)}
    if prefill(task):
        entry["texts"] = {k: prompt_texts(task, k, lengths) for k in prompts}
    return entry


def _measure_group(port, device, names):
    builds, lengths = {}, set()
    for name in names:
        task = load_task(name)
        cal_lengths = tuple(task.get("cal_lengths", CAL_LENGTHS))
        grade_max = max(task.get("grade_lengths", GRADE_LENGTHS)) + task.get("jitter", 12)
        if grade_max > max(cal_lengths) and not prefill(task):
            raise RuntimeError("%s: grade lengths reach %d tokens but calibration records %d" % (name, grade_max, max(cal_lengths)))
        lengths.update(cal_lengths)
    opts = run_opts(load_task(names[0]))  # a session's builds share one command line and weight set
    for name in names:
        task = load_task(name)
        if run_opts(task) != opts:
            raise RuntimeError("tasks %s can't share a Device Run session" % ", ".join(names))
        for side in sides(task):
            b = trusted_build(port, task[side])
            builds["%s--%s" % (name, side)] = build_entry(task, b, prompt_keys(task), side, sorted(lengths))
    rounds = max(load_task(n).get("cal_rounds", 1) for n in names)
    extra = sorted({w for n in names for w in extra_weights(load_task(n))})
    m = fleet.measure(port, builds, sorted(lengths), rounds, device, extra_weights=extra, **opts)
    out = {}
    for name in names:
        task = load_task(name)
        base = m["builds"]["%s--base" % name]
        raw = {"device": m.get("device_id", device), "device_info": m["device"], "session": m["session"],
               "seconds": m["seconds"], "base": base, "at": time.time(),
               # a spec task has no reference: the base stands in, so the shapes below stay the same
               "reference": m["builds"]["%s--reference" % name] if task.get("reference") else base}
        cache_put("cal-raw-%s-%s" % (name, device), raw)  # raw first: changing selection never re-runs phones
        out[(name, device)] = raw
    return out


def _finalize(name, raws):
    task = load_task(name)
    devices = task.get("devices", ["redfin-30"])
    primary = raws[(name, devices[0])]
    base, ref = primary["base"], primary["reference"]
    for side, r in (("base", base), ("reference", ref)):
        if r.get("error"):
            raise RuntimeError("%s %s did not run on %s: %s" % (name, side, devices[0], r["error"]))
    hidden, dev, kept, weaker = select_prompts(task, base, ref, list(base["prompts"]))
    xref = (lambda r, p: r["base"]["prompts"][p]) if prefill(task) else (lambda r, p: None)  # noqa: E731
    ref_d = median([decode_of(ref["prompts"][p], xref(primary, p))[0] for p in hidden])
    base_d = median([decode_of(base["prompts"][p], xref(primary, p))[0] for p in hidden])
    if not hidden or not dev or not ref_d or not base_d:
        # never cache or save a calibration that can't grade: the last good one stays in place
        raise RuntimeError("%s: calibration on %s left %d hidden and %d development prompts with usable timings"
                           % (name, devices[0], len(hidden), len(dev)))
    checks = {}
    for device in devices:
        r = raws[(name, device)]
        if r["base"].get("error") or r["reference"].get("error"):
            checks[device] = {"gpu": r["device_info"].get("gpu"), "error": r["base"].get("error") or r["reference"].get("error")}
            continue
        bd = median([decode_of(r["base"]["prompts"][p], xref(r, p))[0] for p in hidden])
        rd = median([decode_of(r["reference"]["prompts"][p], xref(r, p))[0] for p in hidden])
        checks[device] = {"gpu": r["device_info"].get("gpu"), "model": r["device_info"].get("model"),
                          "soc": r["device_info"].get("soc"), "opencl_device": r["device_info"].get("opencl_device"),
                          "device_id": r.get("device"),
                          "base_decode_tok_s": round(bd, 3) if bd else None, "reference_decode_tok_s": round(rd, 3) if rd else None,
                          "speedup": round(rd / bd, 3) if bd and rd else None,
                          "tokens_match_primary": all(r["reference"]["prompts"][p]["tokens"] == ref["prompts"][p]["tokens"] for p in hidden),
                          "session": r["session"], "session_seconds": r["seconds"]}
    cal = dict(primary, hidden_prompts=hidden, dev_prompts=dev, kept=kept, weaker_check=weaker,
               base_decode=base_d, reference_decode=ref_d, device_checks=checks,
               metric="prefill_tok_s" if prefill(task) else "decode_tok_s")
    cache_put("cal-%s-%s" % (name, devices[0]), cal)
    task.pop("prompt_selection", None)  # numbers from other hardware never carry over
    task.update({"hidden_prompts": hidden, "dev_prompts": dev, "calibrated_on": devices[0]})
    if task.get("target_source", "reference") == "reference":
        task.update({"target_speedup": round(ref_d / base_d, 3), "target_source": "reference"})
    save_task(name, task)
    log("calibrated %s on %s: kept %d prompts (%d hidden, %d dev)%s; base %.2f, reference %.2f tok/s; target %s; devices %s",
        name, devices[0], len(kept), len(hidden), len(dev), " WEAKER: " + weaker if weaker else "", base_d, ref_d,
        task["target_speedup"], {d: c.get("speedup") for d, c in checks.items()})
    return task, cal


def calibrate(name, force=False):
    with _task_locks.setdefault(name, threading.Lock()):
        return calibrate_many([name], force)[name]


def touched(patch):
    return sorted({l.split(" b/", 1)[1] for l in patch.splitlines() if l.startswith("diff --git a/")})


def tokens_match(cand_prompt, source_prompt, task=None):
    if task and prefill(task):  # a different prompt per length: every length's tokens must match exactly
        want = source_prompt["per_length_tokens"]
        return cand_prompt.get("tokens_consistent", True) and all(
            toks == want.get(n) and len(toks) == int(task.get("gen_tokens", 8))
            for n, toks in cand_prompt["per_length_tokens"].items())
    full = source_prompt["tokens"]
    return cand_prompt.get("tokens_consistent", True) and all(
        toks == full[:int(n)] and len(toks) == int(n) for n, toks in cand_prompt["per_length_tokens"].items())


def scoring(task, base_decode, cand_decode, speedup, which):
    if prefill(task):
        return {"metric": "prefill tokens/s",
                "measured": "outside the binary, by a time-vs-prompt-length linear fit on the phone's clock, each run "
                            "generating %d tokens; prompt lengths are the base build's token counts" % int(task.get("gen_tokens", 8)),
                "aggregate": "median across the %s prompts" % ("hidden" if which == "hidden" else "development"),
                "mapping": "reward = clip((speedup - 1) / (target - 1), 0, 1) when valid, else 0",
                "baseline": "the base commit, timed in the same Device Run session",
                "baseline_prefill_tok_s": round(base_decode, 3) if base_decode else None,
                "candidate_prefill_tok_s": round(cand_decode, 3) if cand_decode else None,
                "speedup": round(speedup, 3), "target_speedup": task["target_speedup"],
                "target_source": task.get("target_source")}
    return {"metric": "decode tokens/s", "measured": "outside the binary, by a time-vs-tokens linear fit on the phone's clock",
            "aggregate": "median across the %s prompts" % ("hidden" if which == "hidden" else "development"),
            "mapping": "reward = clip((speedup - 1) / (target - 1), 0, 1) when valid, else 0",
            "baseline": "the base commit, timed in the same Device Run session",
            "baseline_decode_tok_s": round(base_decode, 3) if base_decode else None,
            "candidate_decode_tok_s": round(cand_decode, 3) if cand_decode else None,
            "speedup": round(speedup, 3), "target_speedup": task["target_speedup"],
            "target_source": task.get("target_source")}


class TaskStateError(Exception):
    """The task's stored spec or calibration is incomplete: an infrastructure error, never a reward."""


def evaluate(name, patch, which):
    task, cal = calibrate(name)
    missing = [k for k in ("target_speedup", "port", "base", "in_scope") if k not in task]
    missing += [k for k in ("hidden_prompts", "dev_prompts", "device", "reference", "base") if k not in cal]
    if missing:
        raise TaskStateError("task %s is missing %s in its stored spec or calibration" % (name, ", ".join(missing)))
    target = task["target_speedup"]
    blocked = [f for f in touched(patch) if any(f == p or f.startswith(p.rstrip("/") + "/") for p in task.get("off_limits", []))]
    if blocked:
        return {"task": name, "valid": False, "reward": 0.0, "error_kind": "submission", "off_limits_touched": blocked,
                "error": "patch edits off-limits files", "target_speedup": target}
    prompts = cal["hidden_prompts"] if which == "hidden" else cal["dev_prompts"]
    jitter = task.get("jitter", 12)
    lengths = [n + (random.randint(0, jitter) if which == "hidden" else 0) for n in task.get("grade_lengths", GRADE_LENGTHS)]
    t0 = time.time()
    try:
        cand = (builder.build_version(task["port"], task["base"], patch, set(task["in_scope"])) if patch
                else trusted_build(task["port"], task["base"]))
    except builder.SubmissionError as exc:
        return {"task": name, "valid": False, "reward": 0.0, "error_kind": "submission", "error": str(exc)[:3000],
                "flags": ["submission did not build"], "target_speedup": target}
    build_s = round(time.time() - t0, 1)
    base_b = trusted_build(task["port"], task["base"])
    opts = run_opts(task)
    opts["fallback_devices"] = ()  # grade on the phone model the task was calibrated on
    m = fleet.measure(task["port"], {"base": build_entry(task, base_b, prompts, "base", lengths),
                                     "cand": build_entry(task, cand, prompts, "candidate", lengths)},
                      lengths, task.get("grade_rounds", 1), cal["device"], extra_weights=extra_weights(task), **opts)
    base_m, cand_m = m["builds"]["base"], m["builds"]["cand"]
    if base_m.get("error"):
        raise fleet.FleetError("the base build failed on the phone: %s" % base_m["error"])
    if cand_m.get("error"):
        return {"task": name, "valid": False, "reward": 0.0, "error_kind": cand_m.get("error_kind", "submission"),
                "error": cand_m["error"], "flags": ["submission did not run on the phone"], "target_speedup": target,
                "session": m["session"]}
    sources = {s: cal[s] for s in task.get("valid_token_sources", ["reference", "base"])}
    checks, flags, cand_ds, base_ds = [], [], [], []
    valid = True
    for i, p in enumerate(prompts, 1):
        cp = cand_m["prompts"][p]
        xref = base_m["prompts"][p] if prefill(task) else None  # prompt token counts from the trusted base run
        if prefill(task):
            # the base ran these exact prompts in this session: that run, not calibration, is the token source
            matched = ["base"] if tokens_match(cp, base_m["prompts"][p], task) else []
        else:
            matched = [s for s, data in sources.items() if tokens_match(cp, data["prompts"][p], task)]
        cand_d, r2 = decode_of(cp, xref)
        base_d, _ = decode_of(base_m["prompts"][p], xref)
        worst = worst_residual(cp, xref)
        ok = bool(matched) and r2 >= MIN_R2 and not worst["anomaly"]
        valid = valid and ok
        label = p if which == "dev" else "hidden prompt %d" % i
        check = {"input": label, "matches": matched, "r2": round(r2, 4), "tokens_consistent": cp.get("tokens_consistent"),
                 "decode_tok_s": round(cand_d, 3) if cand_d else None, "base_decode_tok_s": round(base_d, 3) if base_d else None,
                 "worst_residual": worst, "ok": ok}
        if which == "dev":
            check["expected_reference_tokens"] = (base_m if prefill(task) else cal["reference"])["prompts"][p]["per_length_tokens"]
            check["tokens"] = cp["per_length_tokens"]
            if prefill(task):
                check["prompt_tokens"] = xref.get("prompt_tokens")
        if not matched:
            flags.append("%s: tokens match no trusted implementation" % label)
        if r2 < MIN_R2:
            flags.append("%s: time-vs-length not linear (R^2 %.3f)" % (label, r2))
        if worst["anomaly"]:
            flags.append("%s: the %d-token run took %.2fs, %.2fs off the time-vs-length line"
                         % (label, worst["length"], worst["seconds"], worst["residual"]))
        checks.append(check)
        cand_ds.append(cand_d)
        base_ds.append(base_d)
    cand_decode, base_decode = median(cand_ds), median(base_ds)
    claimed = median([pr["self_reported_prefill" if prefill(task) else "self_reported"] for pr in cand_m["prompts"].values()])
    if claimed and cand_decode and claimed > TIMER_TOLERANCE * cand_decode:
        flags.append("binary claims %.1f tok/s but the phone measured %.1f" % (claimed, cand_decode))
        valid = False
    speedup = cand_decode / base_decode if cand_decode and base_decode else 0.0
    reward = max(0.0, min(1.0, (speedup - 1.0) / (target - 1.0))) if valid and target > 1.0 else 0.0
    return {"task": name, "inputs_kind": which, "valid": valid, "reward": round(reward, 4), "checks": checks,
            "flags": flags, "scoring": scoring(task, base_decode, cand_decode, speedup, which),
            "metrics": {("prefill_tok_s" if prefill(task) else "decode_tok_s"): round(cand_decode, 3) if cand_decode else None,
                        ("base_prefill_tok_s" if prefill(task) else "base_decode_tok_s"): round(base_decode, 3) if base_decode else None,
                        "speedup": round(speedup, 3), "target_speedup": target,
                        "self_reported_decode_tok_s": claimed, "lengths": lengths},
            "device": {"id": cal["device"], **m["device"]}, "session": m["session"],
            "timings": {"build_seconds": build_s, "device_session_seconds": m["seconds"]}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log("http " + fmt, *args)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "service": "tf-gateway", "backend": "google-device-run",
                                    "default_device": "redfin-30", **state})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._send(400, {"error": "body must be JSON"})
        if self.path not in AUTH:
            return self._send(404, {"error": "not found"})
        header, key = AUTH[self.path]
        try:
            expected = secret(key)
        except KeyError:
            return self._send(503, {"error": "gateway keys unavailable"})
        if not hmac.compare_digest(self.headers.get(header, ""), expected):
            return self._send(403, {"error": "this endpoint needs the %s header" % header})
        caller = self.headers.get("X-TF-Caller", "")
        if caller:
            with _guard:
                if caller in _in_flight:
                    return self._send(409, {"error": "caller %s already has a request in flight" % caller})
                _in_flight.add(caller)
        t0 = time.time()
        state["running"] += 1
        try:
            if self.path == "/try":
                r = evaluate(req["task"], req.get("patch", ""), "dev")
                result = {"ok": not r.get("error"), "score": r.get("reward"), **r}
            elif self.path == "/grade":
                result = evaluate(req["task"], req.get("patch", ""), "hidden")
            elif self.path == "/calibrate":
                names = req.get("tasks") or [req["task"]]
                done = calibrate_many(names, force=bool(req.get("force")), batch=bool(req.get("batch")))
                result = {"tasks": {n: {"device": c["device"], "target_speedup": t["target_speedup"],
                                        "target_source": t.get("target_source"),
                                        "base_decode_tok_s": round(c["base_decode"], 3),
                                        "reference_decode_tok_s": round(c["reference_decode"], 3),
                                        "hidden_prompts": len(c["hidden_prompts"]), "dev_prompts": c["dev_prompts"],
                                        "kept": len(c.get("kept", [])), "weaker_check": c.get("weaker_check"),
                                        "device_checks": c.get("device_checks"),
                                        "session": c["session"], "session_seconds": c["seconds"]}
                                    for n, (t, c) in done.items()}}
            elif self.path == "/public":
                # Only what agents may see: the development prompts with the reference's tokens, the speeds, the
                # target and how scoring works. Hidden prompts never leave the gateway.
                task, cal = calibrate(req["task"])
                result = {"port": task["port"], "base": task["base"], "reference": task["reference"],
                          "target_speedup": task["target_speedup"], "target_source": task.get("target_source"),
                          "valid_token_sources": task.get("valid_token_sources", ["reference", "base"]),
                          "grade_lengths": task.get("grade_lengths", list(GRADE_LENGTHS)), "jitter": task.get("jitter", 12),
                          "base_tps": round(cal["base_decode"], 3), "reference_tps": round(cal["reference_decode"], 3),
                          "dev_expected": [{"prompt": p, "reference_tokens": cal["reference"]["prompts"][p]["tokens"]}
                                           for p in cal["dev_prompts"]],
                          "weaker_check": cal.get("weaker_check"), "device": cal["device"],
                          "device_checks": {d: {k: c.get(k) for k in ("gpu", "base_decode_tok_s", "reference_decode_tok_s", "speedup")}
                                            for d, c in (cal.get("device_checks") or {}).items()},
                          "scoring": scoring(task, cal["base_decode"], cal["reference_decode"],
                                             cal["reference_decode"] / cal["base_decode"], "hidden")}
            elif self.path == "/set_target":
                task = load_task(req["task"])
                task.update({"target_speedup": float(req["target_speedup"]), "target_source": req.get("source", "manual")})
                save_task(req["task"], task)
                result = {"task": req["task"], "target_speedup": task["target_speedup"], "target_source": task["target_source"]}
            else:
                b = builder.build_version(req["port"], req["commit"], req.get("patch", ""), req.get("allow"))
                result = fleet.measure(req["port"], {"cand": {"tar": b["gcs_tar"], "binary": b["binary"], "prompts": req["prompts"]}},
                                       req.get("lengths", GRADE_LENGTHS), 1, req.get("device", "redfin-30"))
        except (fleet.FleetError, builder.InfraError, TaskStateError) as exc:
            result = {"task": req.get("task"), "error_kind": "infra", "infra_error": str(exc)[:1500]}
        except Exception as exc:
            if self.path in ("/try", "/grade") and "task" in req:
                # a grader fault (missing field, phone output we can't parse, ...) is the gateway's problem,
                # never the candidate's: say so explicitly instead of a bare 400
                log("%s failed: %r", self.path, exc)
                result = {"task": req.get("task"), "error_kind": "infra",
                          "infra_error": "grader error: %s: %s" % (type(exc).__name__, str(exc)[:1500])}
            elif isinstance(exc, (KeyError, ValueError, RuntimeError)):
                log("%s failed: %s", self.path, exc)
                return self._send(400, {"error": str(exc)[:2000]})
            else:
                raise
        finally:
            state["running"] -= 1
            state["served"] += 1
            if caller:
                with _guard:
                    _in_flight.discard(caller)
        result["gateway_seconds"] = round(time.time() - t0, 1)
        log("%s done in %.0fs: %s", self.path, time.time() - t0,
            {k: result.get(k) for k in ("task", "valid", "reward", "flags", "error_kind", "infra_error")})
        self._send(200, signed(result))


if __name__ == "__main__":
    builder.BUILDS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler)
    log("tf-gateway listening on %s:%d", *server.server_address)
    server.serve_forever()
