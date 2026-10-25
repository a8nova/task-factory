"""Mine: Argon reads one whole adreno-llms port and its full history in one call and lists the speedup commits
that can become tasks.

    uv run python pack.py smollm2-135m-instruct
    uv run python mine.py smollm2-135m-instruct

Output: runs/mine/<port>/candidates.json   Progress: tail -f logs/mine.log
"""
import argparse
import json
import time

from common import ARGON, RUNS, gemini_client, get_logger

log = get_logger("mine")

PROMPT = """You are building RL training tasks for coding agents from real engineering history.

Below is one model port from adreno-llms, an OpenCL LLM inference engine for Qualcomm Adreno GPUs on
Android phones: its complete current code, its shared device-test scripts, and the full git history of
the port (every commit with its diff).

Find every commit in that history that made inference measurably FASTER: decode tokens/s, prefill
tokens/s, time to first token, or real-time factor. Ignore commits that only add features, fix bugs,
change docs, or reduce memory without a speed claim.

For each one, decide whether it can become a self-contained task of this form: "Starting from the parent
commit, make <this path> faster on the phone without changing the output tokens." The merged commit is
the reference solution. Grading runs on a real phone: output token IDs must match the device golden file,
and speed is timed on the device (median of 3 warm runs).

Be strict and specific:
- Quote the speed claim exactly as written (commit message, BENCHMARK.md or README) and say where it is.
- Many commits touch several models or mix a speedup with unrelated changes. List the files that carry
  the speedup for THIS port, and say what else the commit changes that the task must leave out.
- Name the concrete ways an agent could fake the speedup for this task (for example caching or
  hardcoding output tokens, skipping layers, lowering precision, timing the wrong region, editing the
  golden file or the benchmark script) and what the grader must check to stop each one.
- Say whether the speedup is likely to hold on other Adreno generations (6xx vs 7xx vs 8xx) and why.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "port": {"type": "string"},
        "commits_reviewed": {"type": "integer"},
        "candidates": {"type": "array", "items": {"type": "object", "properties": {
            "commit": {"type": "string", "description": "full or short sha"},
            "title": {"type": "string"},
            "date": {"type": "string"},
            "metric": {"type": "string", "enum": ["decode_tok_s", "prefill_tok_s", "ttft_s", "rtf", "other"]},
            "claim_quote": {"type": "string"},
            "claim_source": {"type": "string"},
            "claimed_speedup": {"type": "number", "description": "after/before, e.g. 2.3; 0 if not stated"},
            "speedup_files": {"type": "array", "items": {"type": "string"}},
            "unrelated_changes_to_exclude": {"type": "string"},
            "self_contained": {"type": "boolean"},
            "task_title": {"type": "string"},
            "instruction_sketch": {"type": "string"},
            "cheat_risks": {"type": "array", "items": {"type": "object", "properties": {
                "cheat": {"type": "string"}, "grader_check": {"type": "string"}},
                "required": ["cheat", "grader_check"]}},
            "device_dependence": {"type": "string"},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "task_quality": {"type": "integer", "description": "1-5, how good a training task this makes"},
        }, "required": ["commit", "title", "metric", "claim_quote", "claim_source", "claimed_speedup",
                        "speedup_files", "unrelated_changes_to_exclude", "self_contained", "task_title",
                        "instruction_sketch", "cheat_risks", "device_dependence", "difficulty", "task_quality"]}},
        "rejected": {"type": "array", "items": {"type": "object", "properties": {
            "commit": {"type": "string"}, "title": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["commit", "title", "reason"]}},
    },
    "required": ["port", "commits_reviewed", "candidates", "rejected"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    args = ap.parse_args()
    out = RUNS / "mine" / args.port
    pack = (out / "pack.txt").read_text()
    client = gemini_client()
    t0 = time.time()
    interaction = client.interactions.create(
        model=ARGON,
        input=PROMPT + "\n\n" + pack,
        background=True,
        generation_config={"thinking_level": "high", "max_output_tokens": 131072},
        response_format={"type": "text", "mime_type": "application/json", "schema": SCHEMA},
    )
    log.info("started %s on %s (%d chars): interaction %s", ARGON, args.port, len(pack), interaction.id)
    (out / "interaction.txt").write_text(interaction.id)
    while interaction.status in ("in_progress", "queued"):
        time.sleep(15)
        interaction = client.interactions.get(id=interaction.id)
        log.info("  %s after %.0fs", interaction.status, time.time() - t0)
    if interaction.status != "completed":
        raise SystemExit(f"{interaction.status}: {getattr(interaction, 'error', '')}")
    result = json.loads(interaction.output_text)
    (out / "candidates.json").write_text(json.dumps(result, indent=1))
    usage = interaction.usage
    log.info("done in %.0fs: %d candidates, %d rejected, of %s commits; tokens in %s out %s -> %s",
             time.time() - t0, len(result["candidates"]), len(result["rejected"]), result["commits_reviewed"],
             usage.total_input_tokens, usage.total_output_tokens, out / "candidates.json")
    for c in sorted(result["candidates"], key=lambda c: -c["task_quality"]):
        log.info("  q%d %s %s | %s %.2fx | self-contained=%s | %s", c["task_quality"], c["commit"][:8],
                 c["title"][:60], c["metric"], c["claimed_speedup"], c["self_contained"], c["difficulty"])


if __name__ == "__main__":
    main()
