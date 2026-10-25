"""Device gateway: the only way builds reach the phone. Runs on the device host (the Mac mini).

    GATEWAY_PORT=8788 TF_STATE_DIR=~/task-factory/v2 python3 gateway.py
    ssh -N -L 8798:127.0.0.1:8788 macmini    # from the laptop

    GET  /health                       phone model, GPU, temperature, queue state
    POST /try       {task, patch}      agents: base + patch on the development prompts; tokens, speed, score
    POST /grade     {task, patch}      verifier only (X-TF-Verifier): the hidden check on the hidden prompts
    POST /calibrate {task}             factory only (X-TF-Factory): picks prompts, measures base and reference
    POST /measure   {port, commit, patch?, prompts?}   factory only: any commit
    POST /set_target {task, target_speedup, source}    factory only: target from calibration

Grading. Every timed run prints its tokens; a submission is valid only if, at every generation length, its
tokens exactly match one trusted implementation's tokens (the reference, or the base, per the task's
valid_token_sources) for the same prompt and length. So the unchanged base is valid but not faster
(reward ~0), while a build that hardcodes the public fixed input gains nothing because timing uses hidden
prompts. Decode speed is the slope of a line fitted through several generation lengths, timed on the phone's
clock, so a delay that only fires on short runs can't inflate it; a poor fit or an anomalously slow short
run makes the submission invalid. Hidden prompts, their reference tokens and the thresholds never leave this
machine; every response is signed (HMAC-SHA256).

Errors are "submission" (bad patch, failed build, failing binary, wrong tokens, off-limits edit: reward 0)
or "infra" (adb, the phone, this machine: no reward at all). Progress: tail -f ~/task-factory/logs/devrun.log
"""
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import devrun

BASE = Path.home() / "task-factory"
STATE = Path(os.environ.get("TF_STATE_DIR", str(BASE)))  # v2 runs with its own tasks/ and cache/ dirs
TASKS = STATE / "tasks"
CACHE = STATE / "cache"
KEYS = Path.home() / ".config" / "task-factory"
NEEDS_KEY = {"/measure": "gateway-factory-key", "/calibrate": "gateway-factory-key",
             "/set_target": "gateway-factory-key", "/grade": "gateway-verifier-key"}
HEADER = {"gateway-factory-key": "X-TF-Factory", "gateway-verifier-key": "X-TF-Verifier"}
TIMER_TOLERANCE = 1.25     # the binary's own decode claim may exceed our measurement by at most 25%
SHORT_ANOMALY_FRACTION, SHORT_ANOMALY_SECONDS = 0.12, 0.5  # any run this far off the fitted line is flagged
MIN_R2 = 0.98              # time-vs-length must be close to linear
CAL_LENGTHS = (8, 40, 150)   # calibration: 150 gives the full reference token sequence for prefix checks
GRADE_LENGTHS = (8, 40, 136)  # nominal; each is jittered up by 0..12 per grade call, staying <= 150
MIN_HIDDEN, MIN_DEV = 3, 2
log = devrun.log
lock = threading.Lock()
state = {"busy": None, "served": 0}


def secret(name):
    path = KEYS / name
    if not path.exists():
        KEYS.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        os.chmod(path, 0o600)
    return path.read_text().strip()


def signed(result):
    body = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return {"result": result, "signature": hmac.new(secret("gateway-hmac").encode(), body.encode(), hashlib.sha256).hexdigest()}


def task_path(name):
    return TASKS / ("%s.json" % name)


def load_task(name):
    if not task_path(name).exists():
        raise KeyError("unknown task %r" % name)
    return json.loads(task_path(name).read_text())


def save_task(name, task):
    task_path(name).write_text(json.dumps(task, indent=1))


def cache_path(name, side):
    return CACHE / ("%s-%s-v2.json" % (name, side))


def median(xs):
    xs = sorted(x for x in xs if x)
    return (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2) if xs else None


def linfit(points):
    """Least-squares fit of time = a + b*length over (length, time) points. Returns decode tok/s (1/b) and R^2."""
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    if sxx == 0 or sxy <= 0:
        return None, 0.0
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in points)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return (1.0 / b if b > 0 else None), r2


def decode_of(prompt_result):
    pts = [(int(n), t) for n, t in prompt_result["median_times"].items()]
    return linfit(pts)


def calibrate(name):
    """Measure base and reference on the prompt pool once, keep only prompts where their tokens differ, split
    the kept prompts into hidden and development sets, and set the target from the reference's speed."""
    task = load_task(name)
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (cache_path(name, "base").exists() and cache_path(name, "reference").exists()):
        pool = task["prompt_pool"]
        cal_lengths = tuple(task.get("cal_lengths", CAL_LENGTHS))
        grade_max = max(task.get("grade_lengths", GRADE_LENGTHS)) + task.get("jitter", 12)
        if grade_max > max(cal_lengths):
            raise RuntimeError("grade lengths reach %d tokens but calibration only records %d" % (grade_max, max(cal_lengths)))
        sides = {}
        for side in ("base", "reference"):
            # Raw measurements are saved before prompt selection, so changing a selection setting never
            # forces the phone runs again (delete the -raw files to re-measure).
            raw = CACHE / ("%s-%s-raw.json" % (name, side))
            if raw.exists():
                sides[side] = json.loads(raw.read_text())
                continue
            m = devrun.measure(task["port"], task[side], "", pool, task.get("cal_rounds", 1), cal_lengths)
            if m.get("error"):
                raise RuntimeError("%s did not run (%s): %s" % (side, m.get("error_kind"), m["error"][:400]))
            m["measured_at"] = time.time()
            raw.write_text(json.dumps(m))
            sides[side] = m
        kept, rejected = [], []
        for p in pool:
            b, r = sides["base"]["prompts"][p], sides["reference"]["prompts"][p]
            differs = b["tokens"] != r["tokens"]
            stable = all(len(v) == int(n) for n, v in r["per_length_tokens"].items())  # each length produced its tokens
            (kept if differs and stable else rejected).append(p)
        min_hidden, min_dev = task.get("min_hidden", MIN_HIDDEN), task.get("min_dev", MIN_DEV)
        weaker = None
        if len(kept) < min_hidden + min_dev:
            if task.get("require_discriminating_prompts", True):
                raise RuntimeError("only %d of %d prompts tell base and reference apart; add prompts to prompt_pool"
                                   % (len(kept), len(pool)))
            # Smoke tasks only: fall back to every prompt with complete runs. Timing still uses hidden prompts,
            # but unchanged base code is no longer forced to fail correctness on them.
            weaker = "only %d prompts tell base and reference apart; using non-discriminating prompts" % len(kept)
            kept = [p for p in pool if all(len(v) == int(n) for n, v in
                                          sides["reference"]["prompts"][p]["per_length_tokens"].items())]
            log.info("%s: WEAKER CHECK: %s", name, weaker)
        n_dev = min(task.get("max_dev", 3), len(kept) - min_hidden)
        task["dev_prompts"] = kept[-n_dev:]
        task["hidden_prompts"] = kept[:-n_dev][:task.get("max_hidden")]  # max_hidden keeps a /grade short
        task["prompt_selection"] = {"kept": kept, "rejected_same_tokens": rejected, "at": time.time(),
                                    "weaker_check": weaker}
        task.setdefault("valid_token_sources", ["reference", "base"])
        ref_d = median([decode_of(sides["reference"]["prompts"][p])[0] for p in task["hidden_prompts"]])
        base_d = median([decode_of(sides["base"]["prompts"][p])[0] for p in task["hidden_prompts"]])
        if "target_speedup" not in task or task.get("target_source") == "reference":
            task["target_speedup"] = round(ref_d / base_d, 3)
            task["target_source"] = "reference"
        save_task(name, task)
        for side, m in sides.items():
            cache_path(name, side).write_text(json.dumps(m))
        log.info("calibrated %s: kept %d prompts (%d hidden, %d dev), rejected %d same-token; base %.2f, "
                 "reference %.2f tok/s; target %s", name, len(kept), len(task["hidden_prompts"]),
                 len(task["dev_prompts"]), len(rejected), base_d, ref_d, task["target_speedup"])
    base = json.loads(cache_path(name, "base").read_text())
    ref = json.loads(cache_path(name, "reference").read_text())
    return load_task(name), base, ref


def touched_paths(patch):
    return sorted({line.split(" b/", 1)[1] for line in patch.splitlines() if line.startswith("diff --git a/")})


def off_limits(task, patch):
    return [f for f in touched_paths(patch)
            if any(f == p or f.startswith(p.rstrip("/") + "/") for p in task.get("off_limits", []))]


def tokens_match(cand_prompt, source_prompt):
    """Valid against one source if, at every measured length, the candidate's tokens equal that source's
    tokens truncated to the same length (greedy decode is deterministic, so shorter runs are prefixes)."""
    full = source_prompt["tokens"]
    for n, toks in cand_prompt["per_length_tokens"].items():
        if toks != full[:int(n)] or len(toks) != int(n):
            return False
    return True


def grade_prompts(task, base, ref, patch, prompts, visible, jitter):
    j = task.get("jitter", 12)
    lengths = [n + (random.randint(0, j) if jitter else 0) for n in task.get("grade_lengths", GRADE_LENGTHS)]
    cand = devrun.measure(task["port"], task["base"], patch, prompts, task.get("grade_rounds", 1), lengths,
                          allow=set(task["in_scope"]))
    if cand.get("error_kind") == "infra":
        return {"infra_error": cand["error"][:1500]}
    if cand.get("error"):
        return {"valid": False, "reward": 0.0, "error_kind": "submission", "error": cand["error"][:1500],
                "flags": ["submission did not run"]}
    sources = {s: {"base": base, "reference": ref}[s] for s in task.get("valid_token_sources", ["reference", "base"])}
    flags, checks = [], []
    base_decodes, cand_decodes = [], []
    valid = True
    for i, p in enumerate(prompts, 1):
        cp, bp = cand["prompts"][p], base["prompts"][p]
        matched = [s for s, data in sources.items() if tokens_match(cp, data["prompts"][p])]
        cand_d, r2 = decode_of(cp)
        base_d, _ = decode_of(bp)
        # A delay that only fires on some lengths pushes those points off the least-squares line; a uniform
        # startup cost (e.g. a table prewarmed at init) shifts every point equally and is fine.
        pts = sorted((int(n), t) for n, t in cp["median_times"].items())
        n_ = len(pts)
        mx, my = sum(x for x, _ in pts) / n_, sum(y for _, y in pts) / n_
        slope = sum((x - mx) * (y - my) for x, y in pts) / max(sum((x - mx) ** 2 for x, _ in pts), 1e-9)
        resid = [(x, y, y - (my + slope * (x - mx))) for x, y in pts]
        worst = max(resid, key=lambda r: abs(r[2]))
        cand_short, base_short = worst[1], worst[1] - worst[2]  # observed vs expected, for the flag text
        anomaly = abs(worst[2]) > max(SHORT_ANOMALY_FRACTION * worst[1], SHORT_ANOMALY_SECONDS)
        ok = bool(matched) and r2 >= MIN_R2 and not anomaly
        valid = valid and ok
        check = {"input": (p if visible else "hidden prompt %d" % i), "matches": matched,
                 "r2": round(r2, 4), "decode_tok_s": round(cand_d, 3) if cand_d else None,
                 "short_run_anomaly": anomaly, "ok": ok}
        if visible:
            check["expected_reference_tokens"] = ref["prompts"][p]["per_length_tokens"]
            check["tokens"] = cp["per_length_tokens"]
        if not matched:
            flags.append("%s: tokens match no trusted implementation" % check["input"])
        if r2 < MIN_R2:
            flags.append("%s: time-vs-length not linear (R^2 %.3f)" % (check["input"], r2))
        if anomaly:
            flags.append("%s: a run took %.2fs, off the time-vs-length line (expected %.2fs)"
                         % (check["input"], cand_short, base_short))
        checks.append(check)
        if cand_d:
            cand_decodes.append(cand_d)
        if base_d:
            base_decodes.append(base_d)
    claimed = median([cp["self_reported"] for cp in cand["prompts"].values()])
    cand_decode = median(cand_decodes)
    if claimed and cand_decode and claimed > TIMER_TOLERANCE * cand_decode:
        flags.append("binary claims %.1f tok/s but the phone measured %.1f" % (claimed, cand_decode))
        valid = False
    return {"valid": valid, "checks": checks, "flags": flags, "lengths": lengths,
            "cand_decode": cand_decode, "base_decode": median(base_decodes),
            "build_seconds": cand.get("build_seconds"), "device": cand.get("device_after"),
            "self_reported": claimed}


def reward_of(speedup, target, mapping, valid):
    """The scoring rule: continuous and monotone in decode tok/s, 0 at the baseline (speedup 1) and 1 at the
    per-task target. "linear" (default): equal reward steps are equal tok/s steps; "log": equal reward steps
    are equal ratios. Invalid submissions score 0. Noise is handled by the baseline trials, not a deadband."""
    if not valid or not speedup or speedup <= 0:
        return 0.0
    if mapping == "log":
        r = math.log(speedup) / math.log(target)
    else:
        r = (speedup - 1.0) / (target - 1.0)
    return max(0.0, min(1.0, r))


def scoring_block(task, which, base_decode, speedup, reward, lengths, valid):
    """What the reward means, with public numbers only (no prompt text)."""
    return {"metric": "decode_tok_s", "units": "generated tokens per second",
            "conditions": {"device": task.get("device", "Motorola Razr 2020 (Snapdragon 765G, Adreno 620)"),
                           "precision": "fp16 weights", "decoding": "greedy (top-k 1, temperature 0)",
                           "timing": "phone clock (/proc/uptime), measured outside the binary; decode tok/s is the "
                                     "slope of a least-squares line through (generated tokens, seconds)",
                           "lengths": lengths, "prompts": "%s prompts" % ("hidden" if which == "hidden" else "development")},
            "aggregation": "median across %s prompts" % ("hidden" if which == "hidden" else "development"),
            "mapping": task.get("reward_mapping", "linear"),
            "formula": ("clip(ln(speedup) / ln(target_speedup), 0, 1)" if task.get("reward_mapping") == "log"
                        else "clip((speedup - 1) / (target_speedup - 1), 0, 1)") + " if valid, else 0",
            "baseline_decode_tok_s": round(base_decode, 3) if base_decode else None,
            "target_speedup": task["target_speedup"], "target_source": task.get("target_source"),
            "target_decode_tok_s": round(base_decode * task["target_speedup"], 3) if base_decode else None,
            "speedup": round(speedup, 3) if speedup else None, "valid": valid, "reward": round(reward, 4),
            "noise_margin_diagnostic": task.get("noise_margin", 0.03),
            "within_noise_of_baseline": bool(speedup and abs(speedup - 1.0) <= task.get("noise_margin", 0.03))}


def evaluate(name, patch, which):
    task, base, ref = calibrate(name)
    prompts = task["hidden_prompts"] if which == "hidden" else task["dev_prompts"]
    target = task["target_speedup"]
    mapping = task.get("reward_mapping", "linear")
    base_decode = median([decode_of(base["prompts"][p])[0] for p in prompts])
    blocked = off_limits(task, patch)
    if blocked:
        return {"task": name, "valid": False, "reward": 0.0, "error_kind": "submission",
                "off_limits_touched": blocked, "error": "patch edits off-limits files", "target_speedup": target,
                "scoring": scoring_block(task, which, base_decode, None, 0.0, None, False)}
    g = grade_prompts(task, base, ref, patch, prompts, visible=(which == "dev"), jitter=(which == "hidden"))
    if g.get("infra_error"):
        return {"task": name, "infra_error": g["infra_error"]}  # never a reward
    if g.get("error"):  # the submission didn't build or run: invalid, reward 0
        return {"task": name, "inputs_kind": which, "target_speedup": target, **g,
                "scoring": scoring_block(task, which, base_decode, None, 0.0, None, False)}
    speedup = g["cand_decode"] / g["base_decode"] if g.get("cand_decode") and g.get("base_decode") else 0.0
    reward = reward_of(speedup, target, mapping, g["valid"])
    return {"task": name, "inputs_kind": which, "valid": g["valid"], "reward": round(reward, 4),
            "checks": g["checks"], "flags": g["flags"],
            "metrics": {"decode_tok_s": round(g["cand_decode"], 3) if g.get("cand_decode") else None,
                        "base_decode_tok_s": round(g["base_decode"], 3) if g.get("base_decode") else None,
                        "speedup": round(speedup, 3), "target_speedup": target,
                        "self_reported_decode_tok_s": g.get("self_reported"), "lengths": g["lengths"]},
            "scoring": scoring_block(task, which, g.get("base_decode"), speedup, reward, g["lengths"], g["valid"]),
            "build_seconds": g.get("build_seconds"), "device": g.get("device")}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.info("http %s", fmt % args)

    def do_GET(self):
        self._send(200, {"device": devrun.device_state(), **state} if self.path == "/health" else {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._send(400, {"error": "body must be JSON"})
        need = NEEDS_KEY.get(self.path)
        if need and not hmac.compare_digest(self.headers.get(HEADER[need], ""), secret(need)):
            return self._send(403, {"error": "this endpoint needs the %s header" % HEADER[need]})
        t0 = time.time()
        with lock:
            state["busy"] = self.path
            try:
                if self.path == "/try":
                    r = evaluate(req["task"], req.get("patch", ""), "dev")
                    result = {"ok": "infra_error" not in r and not r.get("error"), "score": r.get("reward"), **r}
                elif self.path == "/grade":
                    result = evaluate(req["task"], req.get("patch", ""), "hidden")
                elif self.path == "/calibrate":
                    task, base, ref = calibrate(req["task"])
                    result = {"task": req["task"], "target_speedup": task["target_speedup"],
                              "hidden_prompts": len(task["hidden_prompts"]), "dev_prompts": task["dev_prompts"],
                              "valid_token_sources": task.get("valid_token_sources"),
                              "kept": len(task["prompt_selection"]["kept"]),
                              "rejected_same_tokens": len(task["prompt_selection"]["rejected_same_tokens"])}
                elif self.path == "/set_target":
                    task = load_task(req["task"])
                    task.update({"target_speedup": float(req["target_speedup"]), "target_source": req.get("source", "manual")})
                    save_task(req["task"], task)
                    result = {"task": req["task"], "target_speedup": task["target_speedup"], "target_source": task["target_source"]}
                elif self.path == "/measure":
                    result = devrun.measure(req["port"], req["commit"], req.get("patch", ""), req.get("prompts", []),
                                            int(req.get("rounds", 1)))
                else:
                    return self._send(404, {"error": "not found"})
            except (KeyError, ValueError, RuntimeError) as exc:
                log.info("%s failed: %s", self.path, exc)
                return self._send(400, {"error": str(exc)[:2000]})
            finally:
                state["busy"] = None
                state["served"] += 1
        result["gateway_seconds"] = round(time.time() - t0, 1)
        log.info("%s done in %.0fs: %s", self.path, time.time() - t0,
                 {k: result.get(k) for k in ("task", "valid", "reward", "flags", "error_kind", "infra_error")})
        self._send(200, signed(result))


if __name__ == "__main__":
    for name in ("gateway-hmac", "gateway-factory-key", "gateway-verifier-key"):
        secret(name)
    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("GATEWAY_PORT", "8788"))), Handler)
    log.info("gateway v2 listening on %s:%d", *server.server_address)
    server.serve_forever()
