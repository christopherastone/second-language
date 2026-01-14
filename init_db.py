import sqlite3
from pathlib import Path

from config import get_database_path

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
  gloss_json TEXT,
  proper_nouns_json TEXT,
  grammar_notes_json TEXT,
  natural_translation TEXT,
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
    db_path = get_database_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
