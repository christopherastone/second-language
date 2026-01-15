import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODEL_CHOICES = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4.1", "gpt-4o"]
DEFAULT_MODEL = "gpt-4o"
DEFAULT_DATABASE_PATH = "./data/app.db"

SENTENCE_SCHEMA_VERSION = 1
LEMMA_SCHEMA_VERSION = 1

MAX_RSS_NEW_SENTENCES = 5
RSS_LLM_CONCURRENCY = 5
MAX_LEMMA_SENTENCES = 20

LANGUAGE_RE = re.compile(r"^[a-z]{2}$")

DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(DOTENV_PATH)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_database_path() -> str:
    return DEFAULT_DATABASE_PATH


def load_version() -> str:
    pyproject_path = Path(__file__).with_name("pyproject.toml")
    if not pyproject_path.exists():
        return "0.0.0"
    try:
        import tomllib

        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        return "0.0.0"


def _validate_feed_entry(entry: dict) -> dict:
    if not isinstance(entry, dict):
        raise RuntimeError("feeds.yaml must contain a list of feed objects")
    for key in ("id", "url", "language"):
        if key not in entry or not entry[key]:
            raise RuntimeError(f"feeds.yaml entry missing required field: {key}")
    language = str(entry["language"]).strip().lower()
    if not LANGUAGE_RE.match(language):
        raise RuntimeError(f"Invalid language code in feeds.yaml: {entry['language']}")
    return {
        "id": str(entry["id"]).strip(),
        "url": str(entry["url"]).strip(),
        "language": language,
        "enabled": bool(entry.get("enabled", True)),
    }


def load_feeds(path: str = "feeds.yaml") -> list[dict]:
    feeds_path = Path(path)
    if not feeds_path.exists():
        raise RuntimeError("feeds.yaml is required but was not found")
    try:
        data = yaml.safe_load(feeds_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"feeds.yaml is malformed: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError("feeds.yaml must contain a list of feed objects")
    return [_validate_feed_entry(entry) for entry in data]


def is_language_valid(language: str, feeds: list[dict]) -> bool:
    return any(feed["language"] == language for feed in feeds)


def enabled_feeds(language: str, feeds: list[dict]) -> list[dict]:
    return [feed for feed in feeds if feed["language"] == language and feed["enabled"]]


def die(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)
