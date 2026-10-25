"""Shared paths, logging and the Gemini client for every Task Factory step."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
ADRENO_LLMS = ROOT / "src" / "adreno-llms"
# Outside the repo on purpose: the repo is public, the key never is.
KEY_FILE = Path.home() / ".config" / "task-factory" / "gemini-api-key"

ARGON = "gemini-4-argon"
ANTIGRAVITY = "antigravity-preview-09-2026"
FLASH = "gemini-3.8-flash"
FLASH_LITE = "gemini-3.5-flash-lite"

RUNS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)


def get_logger(name):
    """Log to logs/<name>.log (tail -f it) and to stdout."""
    log = logging.getLogger(name)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for handler in (logging.FileHandler(LOGS / f"{name}.log"), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt)
            log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log


def gemini_client():
    """One API key for Argon, the Antigravity agent, Flash and Flash-Lite. Never printed or logged."""
    from google import genai

    if not KEY_FILE.exists():
        raise SystemExit(f"Put the Gemini API key in {KEY_FILE} (chmod 600); see AGENTS.md")
    if KEY_FILE.stat().st_mode & 0o077:
        raise SystemExit(f"{KEY_FILE} is readable by others; run: chmod 600 {KEY_FILE}")
    return genai.Client(api_key=KEY_FILE.read_text().strip())
