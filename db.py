import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import g

from config import SENTENCE_SCHEMA_VERSION, get_database_path
from normalization import normalize_text, token_has_alpha


def utc_now() -> str:
    """Return the current UTC timestamp in ISO format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_dumps(value) -> str:
    """Serialize a value to JSON with UTF-8 characters preserved."""
    return json.dumps(value, ensure_ascii=False)


def connect_db() -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""
    db_path = get_database_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_db() -> sqlite3.Connection:
    """Return the request-scoped database connection."""
    if "db" not in g:
        g.db = connect_db()
    return g.db


def close_db(exception: Exception | None = None) -> None:
    """Close the request-scoped database connection if present."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def refresh_sentence_lemmas(
    db: sqlite3.Connection,
    language: str,
    sentence_id: int,
    tokens: list[dict],
) -> None:
    """Rebuild lemma rows for a sentence based on token data."""
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


def insert_sentence_from_payload(
    db: sqlite3.Connection,
    *,
    language: str,
    sentence_hash: str,
    text: str,
    article_link: str | None,
    payload: dict,
    created_at: str,
) -> int:
    """Insert a sentence and related lemma metadata, returning its id."""
    cursor = db.execute(
        """
        INSERT INTO sentences (
            language, hash, text, article_link, gloss_json, proper_nouns_json, grammar_notes_json,
            natural_translation, model_used, schema_version, access_count, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            language,
            sentence_hash,
            text,
            article_link,
            json_dumps(payload["tokens"]),
            json_dumps(payload["proper_nouns"]),
            json_dumps(payload["grammar_notes"]),
            payload["natural_english_translation"],
            payload["model_used"],
            SENTENCE_SCHEMA_VERSION,
            created_at,
            created_at,
        ),
    )
    sentence_id = cursor.lastrowid
    refresh_sentence_lemmas(db, language, sentence_id, payload["tokens"])
    return sentence_id
