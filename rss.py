import hashlib
import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import islice

import feedparser
import requests

from config import (
    DEFAULT_MODEL,
    MAX_RSS_NEW_SENTENCES,
    RSS_LLM_CONCURRENCY,
    SENTENCE_SCHEMA_VERSION,
    enabled_feeds,
)
from llm import LLMOutputError, LLMRequestError, generate_sentence_content
from normalization import hash_sentence, normalize_text

EDITORIAL_RE = re.compile(r"\[[^\]]+\]")
TRUNCATION_RE = re.compile(r"(\.\.\.|\u2026)$")
logger = logging.getLogger("second_language.rss")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_headline(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = EDITORIAL_RE.sub("", cleaned)
    cleaned = TRUNCATION_RE.sub("", cleaned)
    cleaned = cleaned.replace("...", "")
    return normalize_text(cleaned)


def _article_id(entry: dict) -> str:
    entry_id = entry.get("id") or entry.get("guid")
    if entry_id:
        return str(entry_id)
    link = entry.get("link", "")
    if link:
        return hashlib.sha256(link.encode("utf-8")).hexdigest()
    return hashlib.sha256(repr(entry).encode("utf-8")).hexdigest()


def _iter_candidates(language: str, feeds: list[dict], db):
    for feed in enabled_feeds(language, feeds):
        logger.info(
            "Fetching feed language=%s id=%s url=%s", language, feed["id"], feed["url"]
        )
        try:
            response = requests.get(feed["url"], timeout=15)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except Exception:
            logger.warning("Feed fetch failed language=%s id=%s", language, feed["id"])
            continue
        for entry in parsed.entries:
            article_id = _article_id(entry)
            link = entry.get("link")
            article_link = str(link).strip() if link else None
            if article_link == "":
                article_link = None
            exists = db.execute(
                "SELECT 1 FROM rss_articles WHERE article_id = ?", (article_id,)
            ).fetchone()
            if exists:
                continue
            headline = _clean_headline(entry.get("title", ""))
            if not headline:
                continue
            sentence_hash = hash_sentence(headline)
            sentence_exists = db.execute(
                "SELECT 1 FROM sentences WHERE language = ? AND hash = ?",
                (language, sentence_hash),
            ).fetchone()
            if sentence_exists:
                db.execute(
                    "INSERT OR IGNORE INTO rss_articles(article_id) VALUES (?)",
                    (article_id,),
                )
                db.commit()
                continue
            yield {
                "article_id": article_id,
                "article_link": article_link,
                "headline": headline,
                "sentence_hash": sentence_hash,
            }


def update_from_feeds(language: str, feeds: list[dict], db) -> int:
    new_count = 0
    candidates = _iter_candidates(language, feeds, db)
    while new_count < MAX_RSS_NEW_SENTENCES:
        batch = list(islice(candidates, RSS_LLM_CONCURRENCY))
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=RSS_LLM_CONCURRENCY) as executor:
            future_map = {
                executor.submit(
                    generate_sentence_content, language, item["headline"], DEFAULT_MODEL
                ): item
                for item in batch
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    payload = future.result()
                except (LLMRequestError, LLMOutputError):
                    logger.warning("LLM failed during RSS import language=%s", language)
                    continue
                if new_count >= MAX_RSS_NEW_SENTENCES:
                    continue
                created_at = utc_now()
                logger.info("Loading into database.")
                db.execute(
                    """
                    INSERT INTO sentences (
                        language, hash, text, article_link, gloss_json, proper_nouns_json, grammar_notes_json,
                        natural_translation, model_used, schema_version, access_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        language,
                        item["sentence_hash"],
                        item["headline"],
                        item["article_link"],
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
                sentence_id = db.execute(
                    "SELECT id FROM sentences WHERE language = ? AND hash = ?",
                    (language, item["sentence_hash"]),
                ).fetchone()["id"]
                _refresh_sentence_lemmas(db, language, sentence_id, payload["tokens"])
                db.execute(
                    "INSERT OR IGNORE INTO rss_articles(article_id) VALUES (?)",
                    (item["article_id"],),
                )
                db.commit()
                new_count += 1
                logger.info(
                    "RSS sentence inserted language=%s hash=%s",
                    language,
                    item["sentence_hash"],
                )
        if new_count >= MAX_RSS_NEW_SENTENCES:
            break
    return new_count


def json_dumps(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _refresh_sentence_lemmas(
    db, language: str, sentence_id: int, tokens: list[dict]
) -> None:
    from normalization import normalize_text, token_has_alpha

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
