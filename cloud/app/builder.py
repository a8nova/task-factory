"""Build one adreno-llms version (a commit, optionally plus a patch) inside the gateway container.

The source comes from `git archive` of the root-only clone, so a build directory never contains git history.
A patch may only touch the task's in-scope files, may not add symlinks, and build files are restored from the
commit after applying it. The build itself runs as the unprivileged `builder` user with a clean environment.
The binary and kernels are packed into a tar in Cloud Storage for Device Run.

    python3 -m builder prewarm <port> <commit> [<commit> ...]   # image build: fill the compiler cache
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

CLONE = Path("/opt/adreno-llms")
BUILDS = Path("/srv/b")
BUCKET = os.environ.get("TF_BUCKET", "gs://mockloop-gcs-dadd-devicerun")
PREFIX = os.environ.get("TF_PREFIX", "tf-gateway")
BUILDER_HOME = "/home/builder"
HEX64 = re.compile(r"[0-9a-f]{64}")
BUILD_FILE_HINTS = ("scripts/", "CMakeLists.txt", ".cmake", "/cmake/", "toolchain")

def binary_path(path, port):
    """The binary a port builds, relative to its folder, read from that commit's own run script: fp16 builds
    of nnopt ports (BINARY_NAME="X${NNOPT_BIN_SUFFIX}" -> build/fp16/X_fp16), or Bonsai's default fp32-storage
    build (BINARY_NAME="X${BIN_SUFFIX}" -> build/X)."""
    script = path / "src" / "models" / port / "scripts" / "run_android.sh"
    text = script.read_text() if script.exists() else ""
    m = re.search(r'BINARY_NAME="([A-Za-z0-9_.]+)\$\{NNOPT_BIN_SUFFIX\}"', text)
    if m:
        return Path("build") / "fp16" / (m.group(1) + "_fp16")
    m = re.search(r'BINARY_NAME="([A-Za-z0-9_.]+)\$\{BIN_SUFFIX\}"', text)
    if m:
        return Path("build") / m.group(1)
    raise InfraError("can't find the binary name in %s/scripts/run_android.sh" % port)


class SubmissionError(Exception):
    """The candidate's fault: patch, build or binary."""


class InfraError(Exception):
    """Not the candidate's fault."""


def scrub(text, limit=1500):
    return HEX64.sub("<redacted>", text)[-limit:]


def run(cmd, cwd=None, timeout=1800, env=None, inp=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, input=inp)


def resolve(commit):
    done = run(["git", "-C", str(CLONE), "rev-parse", commit + "^{commit}"])
    if done.returncode != 0:
        run(["git", "-C", str(CLONE), "fetch", "-q", "origin"], timeout=300)
        done = run(["git", "-C", str(CLONE), "rev-parse", commit + "^{commit}"])
    if done.returncode != 0:
        raise InfraError("unknown commit %s" % commit)
    return done.stdout.strip()


def patch_paths(patch_file, cwd):
    """Every path a patch touches, and whether it adds a symlink or changes modes, via git apply --summary."""
    numstat = run(["git", "apply", "--numstat", str(patch_file)], cwd=cwd)
    if numstat.returncode != 0:
        raise SubmissionError("patch does not parse: %s" % scrub(numstat.stderr))
    paths = set()
    for line in numstat.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            name = parts[2]
            if " => " in name:  # renames: both sides count
                left, right = name.replace("{", "").replace("}", "").split(" => ", 1)
                paths.update([left.strip(), right.strip()])
            else:
                paths.add(name.strip())
    summary = run(["git", "apply", "--summary", str(patch_file)], cwd=cwd).stdout
    symlinks = "120000" in summary
    return paths, symlinks


def differs_from_commit(path, sha):
    """Files under path that are new, changed or deleted relative to the commit, and symlinks it doesn't have."""
    base = {}
    for line in run(["git", "-C", str(CLONE), "ls-tree", "-r", sha]).stdout.splitlines():
        meta, rel = line.split("\t", 1)
        mode, _, blob = meta.split()
        base[rel] = (mode, blob)
    regular, links = [], []
    for root, dirs, files in os.walk(path):
        for name in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            if os.path.islink(full):
                if base.get(rel, ("",))[0] != "120000":
                    links.append(rel)
            else:
                regular.append(rel)
    hashes = run(["git", "hash-object", "--stdin-paths"], cwd=path, inp="\n".join(regular) + "\n").stdout.split()
    changed = [rel for rel, h in zip(regular, hashes) if base.get(rel, (None, None))[1] != h]
    present = set(regular) | set(links)
    changed += [rel for rel, (mode, _) in base.items() if rel not in present and mode != "120000"]
    return changed, links


def allowed_by(allow):
    """In-scope entries are files, or folders ending in "/" (any file under them, including new ones)."""
    allow = list(allow or ())
    files = {a for a in allow if not a.endswith("/")}
    folders = tuple(a for a in allow if a.endswith("/"))
    return lambda p: p in files or (bool(folders) and p.startswith(folders) and ".." not in p.split("/"))


def prepare(port, commit, patch="", allow=None):
    """Extract the commit, apply the patch under the task's rules. Returns (sha, tag, directory)."""
    sha = resolve(commit)
    tag = sha[:12] + ("-" + hashlib.sha256(patch.encode()).hexdigest()[:10] if patch else "")
    # One directory per build call, so concurrent grades of the same commit never share a tree; the depth is
    # the same for every build, so the compiler cache still hits across them.
    work = uuid.uuid4().hex[:12]
    path = BUILDS / work / "src-root"
    path.mkdir(parents=True)
    arch = subprocess.run("git -C %s archive %s | tar -x -C %s" % (CLONE, sha, path), shell=True,
                          capture_output=True, text=True)
    if arch.returncode != 0:
        raise InfraError("git archive: %s" % scrub(arch.stderr))
    if patch:
        patch_file = BUILDS / work / "patch.diff"
        patch_file.write_text(patch)
        touched, symlinks = patch_paths(patch_file, path)
        ok = allowed_by(allow)
        bad = sorted(p for p in touched if not ok(p))
        if bad or symlinks:
            raise SubmissionError("patch changes files outside the allowed set%s: %s"
                                  % (" (or adds symlinks)" if symlinks else "", ", ".join(bad[:20]) or "symlink"))
        done = run(["git", "apply", "--whitespace=nowarn", str(patch_file)], cwd=path)
        if done.returncode != 0:
            raise SubmissionError("patch does not apply: %s" % scrub(done.stdout + done.stderr))
        # Checked again after applying, against the commit's own tree: whatever the patch did, the only files
        # that may differ from the commit are in-scope ones, and nothing may be a symlink the commit lacks.
        changed, links = differs_from_commit(path, sha)
        bad = sorted(p for p in changed if not ok(p))
        if bad or links:
            raise SubmissionError("patch changes files outside the allowed set (or adds symlinks): %s"
                                  % ", ".join((bad + links)[:20]))
        restore = [p for p in (allow or ()) if not p.endswith("/") and any(h in p for h in BUILD_FILE_HINTS)]
        if restore:
            subprocess.run("git -C %s archive %s %s | tar -x -C %s" % (CLONE, sha, " ".join(restore), path),
                           shell=True, capture_output=True)
    subprocess.run(["chown", "-R", "builder:builder", str(path)], check=True)
    return sha, tag, path


def build(path, port):
    model = path / "src" / "models" / port
    binary = model / binary_path(path, port)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": BUILDER_HOME, "ANDROID_NDK": "/opt/ndk",
           "NNOPT_DTYPE": "fp16", "CCACHE_DIR": BUILDER_HOME + "/.ccache", "CCACHE_BASEDIR": str(BUILDS),
           "CCACHE_NOHASHDIR": "1", "CMAKE_C_COMPILER_LAUNCHER": "ccache", "CMAKE_CXX_COMPILER_LAUNCHER": "ccache"}
    t0 = time.time()
    done = run(["/usr/sbin/runuser", "-u", "builder", "--", "bash", "scripts/build.sh", "--release"], cwd=model, env=env,
               timeout=3000)
    if done.returncode != 0 or not binary.exists():
        text = done.stdout + done.stderr
        errors = [l for l in text.splitlines() if "error" in l.lower() and "make" not in l.lower()][:25]
        raise SubmissionError("build failed:\n%s\n--- end of build log ---\n%s"
                              % (scrub("\n".join(errors), 2500), scrub(text, 600)))
    return binary, round(time.time() - t0, 1)


def package(path, port, tag, binary):
    """binary + kernels/ into one tar in Cloud Storage; returns its gs:// path."""
    model = path / "src" / "models" / port
    local = path.parent / (tag + ".tar")
    with tarfile.open(local, "w") as tar:
        tar.add(binary, arcname=binary.name)
        tar.add(model / "kernels", arcname="kernels")
        # shared libraries the binary links (e.g. a dynamic CLBlast); never the OpenCL link stub
        for lib in sorted(binary.parent.rglob("lib*.so*")):
            if lib.is_file() and not lib.name.startswith("libOpenCL"):
                tar.add(lib, arcname=lib.name)
    # A unique object per upload, with its own gsutil state: two uploads to one name (the same commit graded
    # twice at once, or two ports at one monorepo commit) share a resumable-upload tracker and fail with 400.
    remote = "%s/%s/builds/%s/%s-%s.tar" % (BUCKET, PREFIX, port, tag, uuid.uuid4().hex[:8])
    done = run(["gsutil", "-q", "-o", "GSUtil:state_dir=%s" % (path.parent / "gsutil-state"), "cp", str(local), remote],
               timeout=600)
    local.unlink(missing_ok=True)
    if done.returncode != 0:
        raise InfraError("upload of the build failed: %s" % scrub(done.stderr))
    return remote


def build_version(port, commit, patch="", allow=None):
    """Prepare, build and package one version. Returns {sha, tag, gcs_tar, build_seconds}."""
    sha, tag, path = prepare(port, commit, patch, allow)
    try:
        binary, seconds = build(path, port)
        remote = package(path, port, tag, binary)
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)
    return {"sha": sha, "tag": tag, "gcs_tar": remote, "build_seconds": seconds, "binary": binary.name}


def prewarm(port, commits):
    for c in commits:
        sha, tag, path = prepare(port, c)
        t0 = time.time()
        try:
            build(path, port)
            print("prewarmed %s in %.0fs" % (tag, time.time() - t0), flush=True)
        except SubmissionError as exc:
            print("prewarm build of %s failed: %s" % (tag, exc), flush=True)
            raise
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)


if __name__ == "__main__":
    if sys.argv[1] == "prewarm":
        prewarm(sys.argv[2], sys.argv[3:])
