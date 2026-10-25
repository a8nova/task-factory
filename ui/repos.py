"""Repos for the monitor: digest a whole repo from a link, count it, and split it into chunks that each fit one
Argon call. Results live in runs/repos/<slug>.json; clones in src/repos/<slug>/ (both gitignored).

A chunk is a folder (or a group of small folders and files) whose code fits BUDGET tokens, so its git history,
the prompt and the answer still fit in Argon's context. File rules match pack.py: binary files, the same
skipped suffixes and files over 400 KB are left out.

The laptop's disk is tight, so clones are partial (`--filter=blob:none`: full commit history, file contents only
for the current tree) and a repo is refused when it wouldn't leave 5 GB free.
"""
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STORE = RUNS / "repos"
CLONES = ROOT / "src" / "repos"
BUILTIN = {"slug": "a8nova__adreno-llms", "url": "https://github.com/a8nova/adreno-llms", "path": ROOT / "src" / "adreno-llms",
           "units": "src/models"}  # the factory's own repo: one model port per chunk, matching runs/mine/<port>

BUDGET = 450_000          # code tokens per chunk; Argon has 1M, the rest is for history, prompt and answer
CONTEXT = 1_000_000
MAX_FILE_BYTES = 400_000  # same as pack.py
SKIP_SUFFIXES = {".bin", ".png", ".jpg", ".jpeg", ".wav", ".mp4", ".gguf", ".safetensors", ".so", ".pdf", ".npy", ".pt",
                 ".gif", ".ico", ".zip", ".gz", ".tar", ".jar", ".a", ".o", ".dylib", ".dll", ".exe", ".onnx", ".tflite",
                 ".webp", ".ttf", ".woff", ".woff2", ".mp3", ".mov", ".pyc", ".class", ".pkl", ".h5", ".ckpt"}
MIN_FREE = 5 * 1024 ** 3
LANG = {".py": "Python", ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".c": "C", ".h": "C/C++ header", ".hpp": "C/C++ header",
        ".cl": "OpenCL", ".cu": "CUDA", ".metal": "Metal", ".glsl": "GLSL", ".comp": "GLSL", ".m": "Objective-C",
        ".mm": "Objective-C", ".swift": "Swift", ".kt": "Kotlin", ".java": "Java", ".go": "Go", ".rs": "Rust",
        ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".rb": "Ruby", ".php": "PHP",
        ".scala": "Scala", ".cs": "C#", ".sh": "Shell", ".bash": "Shell", ".cmake": "CMake", ".md": "Markdown", ".rst": "Docs",
        ".txt": "Text", ".json": "JSON", ".toml": "TOML", ".yaml": "YAML", ".yml": "YAML", ".html": "HTML", ".css": "CSS",
        ".proto": "Protobuf", ".gradle": "Gradle", ".s": "Assembly", ".S": "Assembly", ".asm": "Assembly", ".ipynb": "Notebook"}
NAMES = {"CMakeLists.txt": "CMake", "Makefile": "Make", "Dockerfile": "Docker", "BUILD": "Bazel", "WORKSPACE": "Bazel"}

_lock = threading.Lock()
_jobs = {}


def git(path, *args, timeout=600, text=True):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=text, timeout=timeout, check=True).stdout


def parse_link(text):
    """A GitHub link (repo, or /tree/<ref>/<folder> for one folder), owner/name, or any https git URL."""
    text = (text or "").strip()
    m = re.match(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/tree/([^/\s]+)(?:/(\S+?))?)?/?(?:[?#]\S*)?$", text) \
        or re.match(r"^([\w.-]+)/([\w.-]+)$", text)
    if m:
        owner, name = m.group(1), m.group(2)
        ref = m.group(3) if m.re.groups >= 3 else None
        scope = (m.group(4) or "").strip("/") if m.re.groups >= 4 else ""
        slug = ("%s__%s" % (owner, name)).lower() + ("__" + re.sub(r"[^\w.-]+", "_", scope).lower() if scope else "")
        return {"url": "https://github.com/%s/%s" % (owner, name), "github": (owner, name), "ref": ref,
                "scope": scope, "slug": slug, "name": "%s/%s" % (owner, name) + ("/" + scope if scope else "")}
    if re.match(r"^https://[\w.-]+(:\d+)?/[\w.~/-]+$", text):
        slug = re.sub(r"[^\w.-]+", "_", text.split("://", 1)[1]).strip("_").lower()[:80]
        return {"url": text, "github": None, "ref": None, "scope": "", "slug": slug, "name": text.split("://", 1)[1]}
    return None


def save(rec):
    STORE.mkdir(parents=True, exist_ok=True)
    rec["updated"] = time.time()
    tmp = STORE / (rec["slug"] + ".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(STORE / (rec["slug"] + ".json"))


def load(slug):
    try:
        return json.loads((STORE / (slug + ".json")).read_text())
    except (OSError, ValueError):
        return None


def all_repos():
    out = []
    for p in sorted(STORE.glob("*.json")) if STORE.exists() else []:
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda r: (r["slug"] != BUILTIN["slug"], -(r.get("added") or 0)))


def add(link):
    """Start digesting a link in the background. Returns the record (or an error)."""
    info = parse_link(link)
    if not info:
        return {"error": "Not a repo link. Use https://github.com/owner/name, owner/name, or a /tree/<branch>/<folder> link."}
    with _lock:
        if info["slug"] in _jobs and _jobs[info["slug"]].is_alive():
            return load(info["slug"]) or {"slug": info["slug"], "status": "queued"}
        rec = load(info["slug"]) or {}
        rec.update({k: v for k, v in info.items() if k != "github"}, status="queued", progress="waiting to start",
                   error=None, added=rec.get("added") or time.time())
        save(rec)
        t = threading.Thread(target=_digest, args=(rec, info), daemon=True)
        _jobs[info["slug"]] = t
        t.start()
    return rec


def ensure_builtin():
    """Digest the factory's own repo once, from the copy already on disk."""
    rec = load(BUILTIN["slug"])
    if rec and rec.get("status") in ("done", "counting", "cloning", "queued", "checking"):
        if rec.get("status") == "done" or time.time() - rec.get("updated", 0) < 600:
            return
    if not BUILTIN["path"].exists():
        return
    rec = {"slug": BUILTIN["slug"], "url": BUILTIN["url"], "name": "a8nova/adreno-llms", "scope": "", "ref": None,
           "builtin": True, "status": "queued", "progress": "waiting to start", "added": time.time()}
    save(rec)
    threading.Thread(target=_digest, args=(rec, None), daemon=True).start()


def refresh(slug):
    rec = load(slug)
    if not rec:
        return {"error": "unknown repo"}
    if rec.get("builtin"):
        rec.update(status="queued", progress="waiting to start", error=None)
        save(rec)
        threading.Thread(target=_digest, args=(rec, None), daemon=True).start()
        return rec
    return add(rec["url"] + ("/tree/%s/%s" % (rec.get("ref") or "HEAD", rec["scope"]) if rec.get("scope") else ""))


def _step(rec, status, progress):
    rec.update(status=status, progress=progress)
    save(rec)


def _digest(rec, info):
    try:
        if rec.get("builtin"):
            path = BUILTIN["path"]
            rec.update(path=str(path.relative_to(ROOT)), partial=False)
        else:
            path = _fetch(rec, info)
        _count(rec, path)
        rec.update(status="done", progress="done", error=None, digested=time.time())
        save(rec)
    except Exception as exc:  # any failure is shown on the page, with the step it happened in
        rec.update(status="error", error="%s: %s" % (type(exc).__name__, str(exc)[:400]))
        save(rec)


def github_meta(owner, name):
    """GET the repo from GitHub's API. Uses the system curl, which has macOS's certificates (the python.org build of
    Python may not), and falls back to urllib."""
    url = "https://api.github.com/repos/%s/%s" % (owner, name)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "task-factory-monitor"}
    if shutil.which("curl"):
        r = subprocess.run(["curl", "-sS", "-m", "15", "-w", "\n%{http_code}", *sum((["-H", "%s: %s" % kv] for kv in headers.items()), []),
                            url], capture_output=True, text=True, timeout=20)
        body, _, code = r.stdout.rpartition("\n")
        if r.returncode == 0 and code == "200":
            return json.loads(body)
        if r.returncode == 0:
            raise RuntimeError("GitHub says %s for %s/%s (private or misspelled?)" % (code, owner, name))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError("GitHub says %s for %s/%s (private or misspelled?)" % (exc.code, owner, name))


def _fetch(rec, info):
    dest = CLONES / "__".join(rec["slug"].split("__")[:2])  # folders of one repo share its clone
    if info.get("github"):
        _step(rec, "checking", "asking GitHub for the repo's size")
        owner, name = info["github"]
        meta = github_meta(owner, name)
        rec["github"] = {k: meta.get(k) for k in ("description", "stargazers_count", "forks_count", "default_branch",
                                                  "size", "language", "pushed_at", "license")}
        if isinstance(rec["github"].get("license"), dict):
            rec["github"]["license"] = rec["github"]["license"].get("spdx_id")
        size = (meta.get("size") or 0) * 1024
        free = shutil.disk_usage(ROOT).free
        rec["disk"] = {"repo_bytes": size, "free_bytes": free}
        if not dest.exists() and free - 2 * size < MIN_FREE:
            raise RuntimeError("the repo is about %.1f GB and the laptop has %.1f GB free; cloning it would leave less "
                               "than 5 GB" % (size / 1e9, free / 1e9))
    CLONES.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        _step(rec, "cloning", "fetching new commits")
        git(dest, "fetch", "--quiet", "--filter=blob:none", "origin")
        git(dest, "reset", "--quiet", "--hard", info.get("ref") and "origin/" + info["ref"] or "origin/HEAD")
    else:
        _step(rec, "cloning", "cloning (full history, current files only)")
        args = ["git", "clone", "--filter=blob:none", "--no-tags", "--progress"]
        if info.get("ref"):
            args += ["--branch", info["ref"]]
        p = subprocess.Popen(args + ["--", rec["url"], str(dest)], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        last = 0
        buf = ""
        while True:
            ch = p.stderr.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip() and time.time() - last > 1:
                    rec["progress"] = "cloning: " + buf.strip()[-120:]
                    save(rec)
                    last = time.time()
                buf = ""
            else:
                buf += ch
        if p.wait() != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError("git clone failed (%s)" % buf.strip()[-200:])
    rec.update(path=str(dest.relative_to(ROOT)), partial=True)
    return dest


def _lang(rel):
    name = rel.rsplit("/", 1)[-1]
    if name in NAMES:
        return NAMES[name]
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return LANG.get(suffix) or LANG.get(suffix.lower()) or "Other"


def _count(rec, path):
    scope = rec.get("scope") or ""
    _step(rec, "counting", "listing files")
    out = git(path, "ls-tree", "-r", "-l", "-z", "HEAD", *(["--", scope] if scope else []), text=False)
    files, skipped, binary, other = [], 0, 0, 0
    entries = [e for e in out.split(b"\0") if e]
    for i, e in enumerate(entries):
        meta, _, rel = e.partition(b"\t")
        mode, kind, _sha, size = meta.split()[:4]
        rel = rel.decode("utf-8", "replace")
        if kind != b"blob" or mode == b"120000":
            other += 1
            continue
        size = int(size) if size != b"-" else 0
        if "." + rel.rsplit(".", 1)[-1].lower() in SKIP_SUFFIXES:
            binary += 1
            continue
        if size > MAX_FILE_BYTES:
            skipped += 1
            continue
        try:
            data = (path / rel).read_bytes()
        except OSError:
            other += 1
            continue
        if b"\0" in data[:4096]:
            binary += 1
            continue
        lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        files.append({"path": rel, "lines": lines, "chars": len(data), "lang": _lang(rel)})
        if i % 2000 == 0:
            rec["progress"] = "counting lines: %d of %d files" % (i, len(entries))
            save(rec)
    chars = sum(f["chars"] for f in files)
    langs = {}
    for f in files:
        x = langs.setdefault(f["lang"], {"lang": f["lang"], "files": 0, "lines": 0, "tokens": 0})
        x["files"] += 1
        x["lines"] += f["lines"]
        x["tokens"] += f["chars"] // 4
    _step(rec, "counting", "reading git history")
    head = git(path, "log", "-1", "--format=%H%x00%cI%x00%s").strip().split("\0")
    roots = git(path, "rev-list", "--max-parents=0", "HEAD").split()
    first = git(path, "log", "-1", "--format=%cI", roots[-1]).strip() if roots else None
    commits = int(git(path, "rev-list", "--count", "HEAD", *(["--", scope] if scope else [])).strip() or 0)
    authors = len([l for l in git(path, "shortlog", "-sne", "HEAD", *(["--", scope] if scope else [])).splitlines() if l.strip()])
    rec["stats"] = {"files": len(files), "lines": sum(f["lines"] for f in files), "chars": chars, "tokens": chars // 4,
                    "binary_files": binary, "large_files_skipped": skipped, "other_entries": other, "commits": commits,
                    "contributors": authors, "first_commit": first, "last_commit": head[1] if len(head) > 1 else None,
                    "head": head[0][:12], "head_subject": head[2] if len(head) > 2 else None,
                    "languages": sorted(langs.values(), key=lambda x: -x["lines"]),
                    "fits_one_call": chars // 4 <= BUDGET}
    rec["budget"] = {"chunk_tokens": BUDGET, "context": CONTEXT}
    _step(rec, "counting", "splitting into chunks")
    rec["chunks"] = split(files, scope, units=BUILTIN["units"] if rec.get("builtin") else None)
    _step(rec, "counting", "counting commits per chunk")
    _chunk_history(rec, path, scope)
    rec["top"] = sorted(({"path": f["path"], "lines": f["lines"], "tokens": f["chars"] // 4} for f in files),
                        key=lambda f: -f["tokens"])[:12]


def split(files, scope="", units=None):
    """Folders whose code fits BUDGET become one chunk each; bigger folders are split by their subfolders; small
    leftovers at one level are packed together. `units` (a folder) forces one chunk per subfolder there."""
    tree = {}
    for f in files:
        rel = f["path"][len(scope):].lstrip("/") if scope else f["path"]
        node = tree
        parts = rel.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = f

    def total(node):
        if "path" in node and isinstance(node.get("lines"), int):
            return node["chars"] // 4, 1, node["lines"]
        t = [total(v) for v in node.values()]
        return sum(x[0] for x in t), sum(x[1] for x in t), sum(x[2] for x in t)

    chunks = []

    def emit(paths, node_list, kind):
        tok = files_n = lines = 0
        for n in node_list:
            a, b, c = total(n)
            tok, files_n, lines = tok + a, files_n + b, lines + c
        chunks.append({"paths": paths, "tokens": tok, "files": files_n, "lines": lines, "kind": kind})

    def walk(node, prefix):
        tok = total(node)[0]
        forced = units and prefix.rstrip("/") == units
        if tok <= BUDGET and not forced:
            return emit([prefix.rstrip("/") or "."], [node], "folder")
        small = []
        for name, child in sorted(node.items(), key=lambda kv: -total(kv[1])[0]):
            is_dir = name.endswith("/")
            ctok = total(child)[0]
            path = prefix + name
            on_way = units and not forced and (units == path.rstrip("/") or units.startswith(path))
            if is_dir and (ctok > BUDGET or on_way):
                walk(child, path)
            elif is_dir and (forced or ctok >= BUDGET // 10):
                emit([path.rstrip("/")], [child], "unit" if forced else "folder")
            else:
                small.append((path.rstrip("/"), child, ctok))
        bins = []
        for path, child, ctok in small:  # first fit, biggest first
            for b in bins:
                if b["tok"] + ctok <= BUDGET:
                    b["items"].append((path, child))
                    b["tok"] += ctok
                    break
            else:
                bins.append({"tok": ctok, "items": [(path, child)]})
        for b in bins:
            emit([p for p, _ in b["items"]], [c for _, c in b["items"]], "group")

    walk(tree, "")
    if scope:
        for c in chunks:
            c["paths"] = [scope + "/" + p if p != "." else scope for p in c["paths"]]
    for i, c in enumerate(sorted(chunks, key=lambda c: c["paths"][0])):
        c["id"] = i + 1
    return sorted(chunks, key=lambda c: c["id"])


def _chunk_history(rec, path, scope):
    """Commits touching each chunk (names only, so a partial clone needn't fetch old file contents). For a full
    clone, also the size of each chunk's history as pack.py would include it."""
    try:
        log = git(path, "log", "--format=%x01%H", "--name-only", "HEAD", *(["--", scope] if scope else []), timeout=180)
    except (subprocess.SubprocessError, OSError):
        return
    chunks = rec["chunks"]
    prefixes = [(c["id"], p) for c in chunks for p in c["paths"]]
    count = {c["id"]: 0 for c in chunks}
    for block in log.split("\x01")[1:]:
        names = [n for n in block.splitlines()[1:] if n]
        hit = {cid for cid, p in prefixes for n in names if p == "." or n == p or n.startswith(p + "/")}
        for cid in hit:
            count[cid] += 1
    for c in chunks:
        c["commits"] = count[c["id"]]
    if rec.get("partial"):
        return
    for c in chunks:
        try:
            hist = git(path, "log", "-p", "--stat", "--date=short", "--format=%n=== COMMIT %H %ad%n%s%n%b", "HEAD", "--",
                       *c["paths"], timeout=120)
            c["history_tokens"] = len(hist) // 4
        except (subprocess.SubprocessError, OSError):
            pass
