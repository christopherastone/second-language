import getpass
import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash

from config import get_database_path, is_language_valid, load_feeds

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  password_hash TEXT NOT NULL,
  default_language TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentences (
  id INTEGER PRIMARY KEY,
  language TEXT NOT NULL,
  hash TEXT NOT NULL,
  text TEXT NOT NULL,
  article_link TEXT,
  source_context TEXT,
  gloss_json TEXT,
  proper_nouns_json TEXT,
  grammar_notes_json TEXT,
  chat_json TEXT NOT NULL DEFAULT '[]',
  natural_translation TEXT,
  audio_data BLOB,
  model_used TEXT,
  schema_version INTEGER NOT NULL,
  access_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(language, hash)
);

CREATE INDEX IF NOT EXISTS idx_sentences_language_created_at
  ON sentences(language, created_at DESC);

CREATE TABLE IF NOT EXISTS sentence_lemmas (
  language TEXT NOT NULL,
  normalized_lemma TEXT NOT NULL,
  sentence_id INTEGER NOT NULL,
  PRIMARY KEY(language, normalized_lemma, sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_sentence_lemmas_lookup
  ON sentence_lemmas(language, normalized_lemma);

CREATE INDEX IF NOT EXISTS idx_sentence_lemmas_sentence_id
  ON sentence_lemmas(sentence_id);

CREATE TABLE IF NOT EXISTS lemmas (
  id INTEGER PRIMARY KEY,
  language TEXT NOT NULL,
  normalized_lemma TEXT NOT NULL,
  translation TEXT,
  related_words_json TEXT,
  chat_json TEXT NOT NULL DEFAULT '[]',
  audio_data BLOB,
  model_used TEXT,
  schema_version INTEGER NOT NULL,
  access_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(language, normalized_lemma)
);

CREATE INDEX IF NOT EXISTS idx_lemmas_language_created_at
  ON lemmas(language, created_at DESC);

CREATE TABLE IF NOT EXISTS rss_articles (
  article_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS favorites (
  id INTEGER PRIMARY KEY,
  item_type TEXT NOT NULL CHECK(item_type IN ('sentence', 'lemma')),
  item_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(item_type, item_id)
);

CREATE INDEX IF NOT EXISTS idx_favorites_created_at
  ON favorites(created_at DESC);
"""


def main() -> None:
    feeds = load_feeds()
    languages = sorted({feed["language"] for feed in feeds})
    if not languages:
        raise RuntimeError("No languages configured in feeds.yaml")

    while True:
        password = getpass.getpass("New admin password: ").strip()
        confirm = getpass.getpass("Confirm password: ").strip()
        if not password:
            print("Password cannot be empty.")
            continue
        if password != confirm:
            print("Passwords do not match.")
            continue
        break

    default_language = languages[0]
    language_hint = ", ".join(languages)
    while True:
        language = input(
            f"Default language ({language_hint}) [default: {default_language}]: "
        ).strip().lower()
        if not language:
            language = default_language
        if is_language_valid(language, feeds):
            break
        print("Invalid language. Must match a configured feeds.yaml language.")

    db_path = get_database_path()
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        db_file.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO settings (id, password_hash, default_language) VALUES (1, ?, ?)",
            (generate_password_hash(password), language),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
