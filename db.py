import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import g

from config import get_database_path
from normalization import normalize_text, token_has_alpha


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def connect_db() -> sqlite3.Connection:
    db_path = get_database_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect_db()
    return g.db


def close_db(exception: Exception | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def refresh_sentence_lemmas(
    db: sqlite3.Connection,
    language: str,
    sentence_id: int,
    tokens: list[dict],
) -> None:
    db.execute("DELETE FROM sentence_lemmas WHERE sentence_id = ?", (sentence_id,))
    seen: set[str] = set()
    for token in tokens:
        surface = token.get("surface", "")
        if not token_has_alpha(surface):
            continue
        lemma = token.get("lemma")
        if not lemma:
            continue
        normalized = normalize_text(lemma)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        db.execute(
            """
            INSERT OR IGNORE INTO sentence_lemmas (language, normalized_lemma, sentence_id)
            VALUES (?, ?, ?)
            """,
            (language, normalized, sentence_id),
        )
