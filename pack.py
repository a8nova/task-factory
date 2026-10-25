"""Pack one adreno-llms model port, its full git history and the shared test scripts into one Argon prompt.

    uv run python pack.py smollm2-135m-instruct            # write the pack, estimate tokens
    uv run python pack.py smollm2-135m-instruct --count    # also ask the API for the exact token count
    uv run python pack.py smollm2-135m-instruct --commit f6e489b   # the port as it was at one commit

Output: runs/mine/<port>/pack.txt and pack.json   Progress: tail -f logs/pack.log
"""
import argparse
import json
import subprocess
from pathlib import Path

from common import ADRENO_LLMS, ARGON, RUNS, gemini_client, get_logger

log = get_logger("pack")
SKIP_SUFFIXES = {".bin", ".png", ".jpg", ".jpeg", ".wav", ".mp4", ".gguf", ".safetensors", ".so", ".pdf", ".npy", ".pt"}
SHARED = ["README.md", "docs/TESTING.md", "scripts/ci/run_device_reference.sh", "scripts/ci/_cmp_tokens.py",
          "scripts/setup_deps.sh", "scripts/fetch_weights.sh"]
MAX_FILE_BYTES = 400_000  # big generated tables or dumps add tokens, not understanding


def git(*args, text=True):
    return subprocess.run(["git", "-C", str(ADRENO_LLMS), *args], capture_output=True, text=text, check=True).stdout


def port_files(port, commit):
    files = git("ls-tree", "-r", "--name-only", commit, "--", f"src/models/{port}").splitlines()
    return SHARED + [f"tests/device_golden/{port}.json"] + files


def read_at(commit, rel):
    try:
        return git("show", f"{commit}:{rel}", text=False)
    except subprocess.CalledProcessError:
        return None  # the file didn't exist yet at that commit


def pack(port, commit="HEAD"):
    parts, skipped, included = [], [], []
    for rel in port_files(port, commit):
        if Path(rel).suffix.lower() in SKIP_SUFFIXES:
            continue
        data = read_at(commit, rel)
        if data is None:
            continue
        if len(data) > MAX_FILE_BYTES or b"\0" in data[:4096]:
            skipped.append({"file": rel, "bytes": len(data)})
            continue
        parts.append(f"=== FILE {rel} ===\n{data.decode('utf-8', 'replace')}")
        included.append(rel)
    history = git("log", "-p", "--stat", "--date=short", "--format=%n=== COMMIT %H %ad%n%s%n%b", commit, "--",
                  f"src/models/{port}")
    commits = history.count("=== COMMIT ")
    text = (f"# adreno-llms model port: {port} at commit {commit}\n\n## Code at that commit\n\n" + "\n\n".join(parts)
            + f"\n\n## Full git history of src/models/{port} ({commits} commits, oldest last)\n" + history)
    return text, {"port": port, "commit": commit, "files": len(included), "skipped_large_or_binary": skipped, "commits": commits,
                  "chars": len(text), "est_tokens": len(text) // 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--count", action="store_true", help="exact token count from the API")
    ap.add_argument("--commit", default="HEAD")
    args = ap.parse_args()
    text, stats = pack(args.port, args.commit)
    if args.count:
        client = gemini_client()  # keep a reference: a temporary client closes before the request is sent
        stats["tokens"] = client.models.count_tokens(model=ARGON, contents=text).total_tokens
    out = RUNS / "mine" / args.port
    out.mkdir(parents=True, exist_ok=True)
    name = "pack" if args.commit == "HEAD" else f"pack-{args.commit[:10]}"
    (out / f"{name}.txt").write_text(text)
    (out / f"{name}.json").write_text(json.dumps(stats, indent=1))
    log.info("%s: %d files, %d commits, %d chars, ~%d tokens est%s, %d large/binary files skipped -> %s",
             args.port, stats["files"], stats["commits"], stats["chars"], stats["est_tokens"],
             f", {stats['tokens']} exact" if "tokens" in stats else "", len(stats["skipped_large_or_binary"]),
             out / f"{name}.txt")


if __name__ == "__main__":
    main()
