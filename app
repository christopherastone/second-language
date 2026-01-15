#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "flask>=3.0.0",
#   "flask-wtf>=1.2.1",
#   "requests>=2.32.0",
#   "feedparser>=6.0.11",
#   "openai>=1.40.0",
#   "pyyaml>=6.0.1",
#   "jsonschema>=4.22.0",
#   "google-cloud-texttospeech>=2.14.0",
#   "python-dotenv>=1.0.1",
# ]
# ///

import json
import logging
import os
import threading
import time
from datetime import timedelta
from urllib.parse import quote, unquote

from flask import (
    Flask,
    Response,
    abort,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf import CSRFProtect
from werkzeug.security import check_password_hash

from config import (
    DEFAULT_MODEL,
    LEMMA_SCHEMA_VERSION,
    MAX_LEMMA_SENTENCES,
    MODEL_CHOICES,
    SENTENCE_SCHEMA_VERSION,
    get_database_path,
    is_language_valid,
    load_feeds,
    load_version,
    require_env,
)
from db import close_db, get_db, json_dumps, refresh_sentence_lemmas, utc_now
from llm import (
    LLMOutputError,
    LLMRequestError,
    generate_audio,
    generate_lemma_chat_reply,
    generate_lemma_content,
    generate_sentence_chat_reply,
    generate_sentence_content,
    validate_openai_key,
)
from normalization import normalize_text, token_has_alpha
from rss import update_from_feeds

APP_NAME = "Second Language"
logger = logging.getLogger("second_language")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

app = Flask(__name__)
app.config["SECRET_KEY"] = require_env("SECRET_KEY")
app.config["DATABASE_PATH"] = get_database_path()
app.config["OPENAI_API_KEY"] = require_env("OPENAI_API_KEY")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=3650)
app.config["APP_VERSION"] = load_version()
app.config["FEEDS"] = load_feeds()

csrf = CSRFProtect(app)
app.teardown_appcontext(close_db)

validate_openai_key()

LOGIN_STATE: dict[str, dict[str, float | int]] = {}
LEMMA_LOCKS: dict[tuple[str, str], dict[str, object]] = {}
LEMMA_LOCK = threading.Lock()


def get_settings() -> dict:
    if "settings" not in g:
        row = get_db().execute("SELECT * FROM settings WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError(
                "Settings not configured. Run cli set-password and cli set-language."
            )
        g.settings = dict(row)
    return g.settings


def get_language_list() -> list[str]:
    languages = {feed["language"] for feed in app.config["FEEDS"]}
    return sorted(languages)


def require_language(language: str) -> str:
    language = language.lower()
    if not is_language_valid(language, app.config["FEEDS"]):
        abort(404)
    return language


def ensure_model(model: str | None) -> str:
    if not model:
        return session.get("model_choice", DEFAULT_MODEL)
    if model not in MODEL_CHOICES:
        return DEFAULT_MODEL
    return model


def ensure_chat_model(model: str | None) -> str:
    if not model:
        return session.get("chat_model_choice", DEFAULT_MODEL)
    if model not in MODEL_CHOICES:
        return DEFAULT_MODEL
    return model


def json_loads(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def load_chat_messages(value: str | None) -> list[dict]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "created_at": item.get("created_at"),
                    "model_used": item.get("model_used"),
                }
            )
    return messages


def get_sentence_content(row: dict) -> dict | None:
    if row["schema_version"] != SENTENCE_SCHEMA_VERSION:
        return None
    if row["model_used"] is None:
        return None
    tokens = json_loads(row["gloss_json"])
    proper = json_loads(row["proper_nouns_json"])
    grammar = json_loads(row["grammar_notes_json"])
    if tokens is None or proper is None or grammar is None:
        return None
    natural = row.get("natural_translation")
    if not natural:
        return None
    return {
        "tokens": tokens,
        "proper_nouns": proper,
        "grammar_notes": grammar,
        "natural_english_translation": natural,
        "model_used": row["model_used"],
    }


def get_lemma_content(row: dict) -> dict | None:
    if row["schema_version"] != LEMMA_SCHEMA_VERSION:
        return None
    if row["model_used"] is None:
        return None
    related = json_loads(row["related_words_json"])
    if row["translation"] is None or related is None:
        return None
    return {
        "lemma": row["normalized_lemma"],
        "translation": row["translation"],
        "related_words": related,
        "model_used": row["model_used"],
    }


def sentence_payload_to_content(payload: dict) -> dict:
    return {
        "tokens": payload["tokens"],
        "proper_nouns": payload["proper_nouns"],
        "grammar_notes": payload["grammar_notes"],
        "natural_english_translation": payload["natural_english_translation"],
        "model_used": payload["model_used"],
    }


def lemma_payload_to_content(payload: dict) -> dict:
    return {
        "lemma": payload["lemma"],
        "translation": payload["translation"],
        "related_words": payload["related_words"],
        "model_used": payload["model_used"],
    }


def generate_sentence_content_cached(
    language: str,
    sentence_id: int,
    sentence_text: str,
    model: str,
    source_context: str | None = None,
) -> tuple[dict | None, str | None]:
    payload, error = ensure_sentence_payload(
        language, sentence_text, model, source_context=source_context
    )
    if payload:
        payload["language"] = language
        update_sentence_cache(sentence_id, payload)
        return sentence_payload_to_content(payload), None
    return None, error


def generate_lemma_content_cached(
    language: str,
    lemma_id: int,
    lemma_text: str,
    normalized_lemma: str,
    model: str,
) -> tuple[dict | None, str | None]:
    payload, error = ensure_lemma_payload(language, lemma_text, normalized_lemma, model)
    if payload:
        update_lemma_cache(lemma_id, payload)
        return lemma_payload_to_content(payload), None
    return None, error


def resolve_sentence_content(
    language: str,
    row,
    model: str,
    force_generate: bool,
) -> tuple[dict | None, str | None]:
    row_dict = dict(row)
    content = get_sentence_content(row_dict)
    error = None
    if force_generate or content is None:
        if force_generate:
            logger.info(
                "Sentence regeneration requested language=%s hash=%s model=%s",
                language,
                row["hash"],
                model,
            )
        else:
            logger.info(
                "Sentence cache miss language=%s hash=%s model=%s",
                language,
                row["hash"],
                model,
            )
        generated_content, error = generate_sentence_content_cached(
            language,
            row["id"],
            row["text"],
            model,
            source_context=row["source_context"],
        )
        if generated_content is None:
            if force_generate:
                logger.warning(
                    "Sentence regeneration failed language=%s hash=%s",
                    language,
                    row["hash"],
                )
            else:
                logger.warning(
                    "Sentence generation failed language=%s hash=%s",
                    language,
                    row["hash"],
                )
        else:
            content = generated_content
    return content, error


def resolve_lemma_content(
    db,
    language: str,
    normalized_lemma: str,
    row,
    model: str,
    force_generate: bool,
) -> tuple[dict | None, str | None]:
    row_dict = dict(row)
    content = get_lemma_content(row_dict)
    error = None
    if force_generate:
        logger.info(
            "Lemma regeneration requested language=%s lemma=%s model=%s",
            language,
            normalized_lemma,
            model,
        )
        generated_content, error = generate_lemma_content_cached(
            language,
            row["id"],
            normalized_lemma,
            normalized_lemma,
            model,
        )
        if generated_content is None:
            logger.warning(
                "Lemma regeneration failed language=%s lemma=%s",
                language,
                normalized_lemma,
            )
        else:
            content = generated_content
        return content, error

    if content is None:

        def checker():
            refreshed = db.execute(
                "SELECT * FROM lemmas WHERE language = ? AND normalized_lemma = ?",
                (language, normalized_lemma),
            ).fetchone()
            if refreshed is None:
                return None
            return get_lemma_content(dict(refreshed))

        def generator():
            logger.info(
                "Lemma cache miss language=%s lemma=%s model=%s",
                language,
                normalized_lemma,
                model,
            )
            return generate_lemma_content_cached(
                language,
                row["id"],
                normalized_lemma,
                normalized_lemma,
                model,
            )

        content, error = with_lemma_singleflight(
            language, normalized_lemma, checker, generator
        )
        if error:
            logger.warning(
                "Lemma generation failed language=%s lemma=%s",
                language,
                normalized_lemma,
            )
    return content, error


def increment_access(table: str, row_id: int) -> None:
    db = get_db()
    db.execute(
        f"UPDATE {table} SET access_count = access_count + 1 WHERE id = ?", (row_id,)
    )
    db.commit()


def fetch_sentence_list(language: str):
    db = get_db()
    return db.execute(
        """
        SELECT s.*, EXISTS(
            SELECT 1 FROM favorites f
            WHERE f.item_type = 'sentence' AND f.item_id = s.id
        ) AS is_favorite
        FROM sentences s
        WHERE s.language = ?
        ORDER BY s.created_at DESC
        """,
        (language,),
    ).fetchall()


def fetch_sentences_for_lemma(language: str, normalized_lemma: str):
    db = get_db()
    return db.execute(
        """
        SELECT s.*
        FROM sentences s
        JOIN sentence_lemmas sl ON sl.sentence_id = s.id
        WHERE sl.language = ? AND sl.normalized_lemma = ?
        ORDER BY s.created_at DESC
        LIMIT ?
        """,
        (language, normalized_lemma, MAX_LEMMA_SENTENCES),
    ).fetchall()


def toggle_favorite(item_type: str, item_id: int) -> bool:
    db = get_db()
    favorite = db.execute(
        "SELECT 1 FROM favorites WHERE item_type = ? AND item_id = ?",
        (item_type, item_id),
    ).fetchone()
    if favorite:
        db.execute(
            "DELETE FROM favorites WHERE item_type = ? AND item_id = ?",
            (item_type, item_id),
        )
        db.commit()
        return False
    db.execute(
        "INSERT OR IGNORE INTO favorites (item_type, item_id, created_at) VALUES (?, ?, ?)",
        (item_type, item_id, utc_now()),
    )
    db.commit()
    return True


def update_sentence_cache(sentence_id: int, payload: dict) -> None:
    db = get_db()
    db.execute(
        """
        UPDATE sentences
        SET gloss_json = ?,
            proper_nouns_json = ?,
            grammar_notes_json = ?,
            natural_translation = ?,
            model_used = ?,
            schema_version = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            json_dumps(payload["tokens"]),
            json_dumps(payload["proper_nouns"]),
            json_dumps(payload["grammar_notes"]),
            payload["natural_english_translation"],
            payload["model_used"],
            SENTENCE_SCHEMA_VERSION,
            utc_now(),
            sentence_id,
        ),
    )
    refresh_sentence_lemmas(
        db,
        payload["language"],
        sentence_id,
        payload["tokens"],
    )
    db.commit()


def update_lemma_cache(lemma_id: int, payload: dict) -> None:
    db = get_db()
    db.execute(
        """
        UPDATE lemmas
        SET translation = ?,
            related_words_json = ?,
            model_used = ?,
            schema_version = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            payload["translation"],
            json_dumps(payload["related_words"]),
            payload["model_used"],
            LEMMA_SCHEMA_VERSION,
            utc_now(),
            lemma_id,
        ),
    )
    db.commit()


def clear_sentence_chat(sentence_id: int) -> None:
    db = get_db()
    db.execute(
        "UPDATE sentences SET chat_json = ?, updated_at = ? WHERE id = ?",
        (json_dumps([]), utc_now(), sentence_id),
    )
    db.commit()


def clear_lemma_chat(lemma_id: int) -> None:
    db = get_db()
    db.execute(
        "UPDATE lemmas SET chat_json = ?, updated_at = ? WHERE id = ?",
        (json_dumps([]), utc_now(), lemma_id),
    )
    db.commit()


def ensure_sentence_payload(
    language: str,
    sentence_text: str,
    model: str,
    source_context: str | None = None,
) -> tuple[dict | None, str | None]:
    try:
        payload = generate_sentence_content(
            language, sentence_text, model, source_context=source_context
        )
        payload["language"] = language
        return payload, None
    except (LLMRequestError, LLMOutputError):
        return None, "LLM generation failed. Please try again."


def ensure_lemma_payload(
    language: str, lemma: str, normalized_lemma: str, model: str
) -> tuple[dict | None, str | None]:
    try:
        payload = generate_lemma_content(language, lemma, normalized_lemma, model)
        return payload, None
    except (LLMRequestError, LLMOutputError):
        return None, "LLM generation failed. Please try again."


def get_lemma_lock(key: tuple[str, str]) -> dict[str, object]:
    with LEMMA_LOCK:
        if key not in LEMMA_LOCKS:
            LEMMA_LOCKS[key] = {"cond": threading.Condition(), "in_progress": False}
        return LEMMA_LOCKS[key]


def with_lemma_singleflight(language: str, normalized_lemma: str, checker, generator):
    key = (language, normalized_lemma)
    entry = get_lemma_lock(key)
    cond: threading.Condition = entry["cond"]  # type: ignore[assignment]
    with cond:
        waited = False
        while entry["in_progress"]:
            if not waited:
                logger.info(
                    "Lemma generation in progress, waiting language=%s lemma=%s",
                    language,
                    normalized_lemma,
                )
                waited = True
            cond.wait(timeout=70)
        existing = checker()
        if existing is not None:
            if waited:
                logger.info(
                    "Lemma content available after wait language=%s lemma=%s",
                    language,
                    normalized_lemma,
                )
            return existing, None
        entry["in_progress"] = True
    try:
        return generator()
    finally:
        with cond:
            entry["in_progress"] = False
            cond.notify_all()


def enforce_login_backoff(key: str) -> None:
    record = LOGIN_STATE.get(key)
    now = time.time()
    if record is None:
        return
    last_attempt = record.get("last_attempt", 0)
    failures = record.get("failures", 0)
    if now - last_attempt > 3600:
        record["failures"] = 0
        return
    if failures <= 0:
        return
    delay = min(5 * (2 ** (failures - 1)), 60)
    elapsed = now - last_attempt
    if elapsed < delay:
        time.sleep(delay - elapsed)


def register_login_failure(key: str) -> None:
    now = time.time()
    record = LOGIN_STATE.setdefault(key, {"failures": 0, "last_attempt": 0})
    record["failures"] = int(record.get("failures", 0)) + 1
    record["last_attempt"] = now


def reset_login_failures(key: str) -> None:
    LOGIN_STATE[key] = {"failures": 0, "last_attempt": 0}


@app.before_request
def require_login():
    app.config["SESSION_COOKIE_SECURE"] = request.is_secure
    if request.path.startswith("/static") or request.path == "/favicon.ico":
        return None
    if request.path == "/login":
        return None
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    session.setdefault("show_glosses", True)
    session.setdefault("show_translations", True)
    if session.get("model_choice_source") != "user":
        session["model_choice"] = DEFAULT_MODEL
        session["model_choice_source"] = "default"
    if session.get("chat_model_choice_source") != "user":
        session["chat_model_choice"] = DEFAULT_MODEL
        session["chat_model_choice_source"] = "default"
    session.permanent = True
    return None


@app.context_processor
def inject_globals():
    settings = get_settings()
    return {
        "app_name": APP_NAME,
        "app_version": app.config["APP_VERSION"],
        "languages": get_language_list(),
        "default_language": settings["default_language"],
        "show_glosses": session.get("show_glosses", True),
        "show_translations": session.get("show_translations", True),
        "model_choices": MODEL_CHOICES,
        "selected_model": session.get("model_choice", DEFAULT_MODEL),
        "selected_chat_model": session.get("chat_model_choice", DEFAULT_MODEL),
        "show_gloss_toggle": True,
        "show_translations_toggle": False,
    }


@app.template_filter("lemma_url")
def lemma_url_filter(lemma: str, language: str) -> str:
    normalized = normalize_text(lemma)
    return url_for("lemma_detail", lang=language, lemma=quote(normalized))


@app.template_filter("has_alpha")
def has_alpha_filter(text: str) -> bool:
    return token_has_alpha(text)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        enforce_login_backoff("default")
        password = request.form.get("password", "")
        settings = get_settings()
        if check_password_hash(settings["password_hash"], password):
            reset_login_failures("default")
            session["logged_in"] = True
            session.permanent = True
            return redirect(url_for("sentence_list", lang=settings["default_language"]))
        register_login_failure("default")
        error = "Invalid password"
    return render_template("login.html", error=error)


@app.route("/")
def root():
    settings = get_settings()
    return redirect(url_for("sentence_list", lang=settings["default_language"]))


@app.route("/<lang>/")
def sentence_list(lang: str):
    language = require_language(lang)
    sentences = fetch_sentence_list(language)
    return render_template(
        "sentence_list.html",
        language=language,
        sentences=sentences,
        show_translations_toggle=True,
        show_gloss_toggle=False,
    )


@app.post("/<lang>/update")
def update_sentences(lang: str):
    language = require_language(lang)
    db = get_db()
    logger.info("RSS update started language=%s", language)
    new_count = update_from_feeds(language, app.config["FEEDS"], db)
    logger.info("RSS update finished language=%s new_count=%s", language, new_count)
    sentences = fetch_sentence_list(language)
    if request.headers.get("HX-Request"):
        return render_template(
            "partials/sentence_list_inner.html", language=language, sentences=sentences
        )
    return redirect(url_for("sentence_list", lang=language))


@app.route("/<lang>/sentence/<hash_slug>")
def sentence_detail(lang: str, hash_slug: str):
    language = require_language(lang)
    db = get_db()
    row = db.execute(
        "SELECT * FROM sentences WHERE language = ? AND hash = ?",
        (language, hash_slug),
    ).fetchone()
    if row is None:
        return redirect(url_for("sentence_list", lang=language))

    model = ensure_model(None)
    content, error = resolve_sentence_content(
        language,
        row,
        model,
        force_generate=False,
    )

    is_favorite = db.execute(
        "SELECT 1 FROM favorites WHERE item_type = 'sentence' AND item_id = ?",
        (row["id"],),
    ).fetchone()

    if not request.headers.get("HX-Request") and content is not None:
        increment_access("sentences", row["id"])

    chat_messages = load_chat_messages(row["chat_json"])

    return render_template(
        "sentence_detail.html",
        language=language,
        sentence=row,
        content=content,
        error=error,
        is_favorite=bool(is_favorite),
        chat_messages=chat_messages,
    )


@app.post("/<lang>/sentence/<hash_slug>/regenerate")
def sentence_regenerate(lang: str, hash_slug: str):
    language = require_language(lang)
    model = ensure_model(request.form.get("model"))
    session["model_choice"] = model
    session["model_choice_source"] = "user"
    db = get_db()
    row = db.execute(
        "SELECT * FROM sentences WHERE language = ? AND hash = ?",
        (language, hash_slug),
    ).fetchone()
    if row is None:
        abort(404)
    content, error = resolve_sentence_content(
        language,
        row,
        model,
        force_generate=True,
    )
    if content is not None:
        clear_sentence_chat(row["id"])
        chat_messages: list[dict] = []
    else:
        chat_messages = load_chat_messages(row["chat_json"])
    is_favorite = db.execute(
        "SELECT 1 FROM favorites WHERE item_type = 'sentence' AND item_id = ?",
        (row["id"],),
    ).fetchone()
    return render_template(
        "partials/sentence_content.html",
        language=language,
        sentence=row,
        content=content,
        error=error,
        is_favorite=bool(is_favorite),
        chat_messages=chat_messages,
    )


@app.post("/<lang>/sentence/<hash_slug>/chat")
def sentence_chat(lang: str, hash_slug: str):
    language = require_language(lang)
    message = request.form.get("message", "").strip()
    model = ensure_chat_model(request.form.get("model"))
    content_model = ensure_model(None)
    session["chat_model_choice"] = model
    session["chat_model_choice_source"] = "user"
    db = get_db()
    row = db.execute(
        "SELECT * FROM sentences WHERE language = ? AND hash = ?",
        (language, hash_slug),
    ).fetchone()
    if row is None:
        abort(404)
    chat_messages = load_chat_messages(row["chat_json"])
    chat_error = None
    if message:
        now = utc_now()
        chat_messages.append({"role": "user", "content": message, "created_at": now})
        db.execute(
            "UPDATE sentences SET chat_json = ?, updated_at = ? WHERE id = ?",
            (json_dumps(chat_messages), now, row["id"]),
        )
        db.commit()
        content, error = resolve_sentence_content(
            language,
            row,
            content_model,
            force_generate=False,
        )
        if content is None:
            chat_error = error or "Unable to load sentence context."
        else:
            try:
                reply = generate_sentence_chat_reply(
                    language,
                    row["text"],
                    content,
                    chat_messages,
                    model=model,
                )
                if reply:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "created_at": utc_now(),
                            "model_used": model,
                        }
                    )
                    db.execute(
                        "UPDATE sentences SET chat_json = ?, updated_at = ? WHERE id = ?",
                        (json_dumps(chat_messages), utc_now(), row["id"]),
                    )
                    db.commit()
                else:
                    chat_error = "No response returned. Please try again."
            except (LLMRequestError, LLMOutputError):
                chat_error = "LLM generation failed. Please try again."

    return render_template(
        "partials/chat_section.html",
        chat_id="sentence-chat",
        chat_title="Chat",
        chat_action=url_for("sentence_chat", lang=language, hash_slug=hash_slug),
        chat_placeholder="Ask about this sentence...",
        chat_item_label="sentence",
        chat_messages=chat_messages,
        chat_error=chat_error,
    )


@app.post("/<lang>/sentence/<hash_slug>/favorite")
def sentence_favorite(lang: str, hash_slug: str):
    language = require_language(lang)
    db = get_db()
    row = db.execute(
        "SELECT id FROM sentences WHERE language = ? AND hash = ?",
        (language, hash_slug),
    ).fetchone()
    if row is None:
        abort(404)
    is_favorite = toggle_favorite("sentence", row["id"])
    return render_template(
        "partials/favorite_star.html",
        item_type="sentence",
        is_favorite=is_favorite,
        target_id=f"sentence-favorite-{row['id']}",
        action=url_for("sentence_favorite", lang=language, hash_slug=hash_slug),
    )


@app.post("/<lang>/sentence/<hash_slug>/delete")
def sentence_delete(lang: str, hash_slug: str):
    language = require_language(lang)
    db = get_db()
    row = db.execute(
        "SELECT id FROM sentences WHERE language = ? AND hash = ?",
        (language, hash_slug),
    ).fetchone()
    if row is None:
        abort(404)
    db.execute(
        "DELETE FROM favorites WHERE item_type = 'sentence' AND item_id = ?",
        (row["id"],),
    )
    db.execute("DELETE FROM sentence_lemmas WHERE sentence_id = ?", (row["id"],))
    db.execute("DELETE FROM sentences WHERE id = ?", (row["id"],))
    db.commit()
    return redirect(url_for("sentence_list", lang=language))


@app.get("/<lang>/sentence/<hash_slug>/audio")
def sentence_audio(lang: str, hash_slug: str):
    language = require_language(lang)
    db = get_db()
    row = db.execute(
        "SELECT id, text, audio_data FROM sentences WHERE language = ? AND hash = ?",
        (language, hash_slug),
    ).fetchone()
    if row is None:
        abort(404)
    audio_data = row["audio_data"]
    if audio_data is None:
        audio_data = generate_audio(row["text"], language)
        db.execute(
            "UPDATE sentences SET audio_data = ? WHERE id = ?",
            (audio_data, row["id"]),
        )
        db.commit()
    return Response(audio_data, mimetype="audio/mpeg")


@app.get("/<lang>/lemma/<lemma>/audio")
def lemma_audio(lang: str, lemma: str):
    language = require_language(lang)
    lemma_display = unquote(lemma)
    normalized = normalize_text(lemma_display)
    db = get_db()
    row = db.execute(
        "SELECT id, normalized_lemma, audio_data FROM lemmas WHERE language = ? AND normalized_lemma = ?",
        (language, normalized),
    ).fetchone()
    if row is None:
        abort(404)
    audio_data = row["audio_data"]
    if audio_data is None:
        audio_data = generate_audio(row["normalized_lemma"], language)
        db.execute(
            "UPDATE lemmas SET audio_data = ? WHERE id = ?",
            (audio_data, row["id"]),
        )
        db.commit()
    return Response(audio_data, mimetype="audio/mpeg")


@app.route("/<lang>/lemma/<lemma>")
def lemma_detail(lang: str, lemma: str):
    language = require_language(lang)
    lemma_display = unquote(lemma)
    normalized = normalize_text(lemma_display)
    db = get_db()
    row = db.execute(
        "SELECT * FROM lemmas WHERE language = ? AND normalized_lemma = ?",
        (language, normalized),
    ).fetchone()
    if row is None:
        created_at = utc_now()
        db.execute(
            """
            INSERT INTO lemmas (
                language, normalized_lemma, translation, related_words_json, chat_json,
                model_used, schema_version, access_count, created_at, updated_at
            ) VALUES (?, ?, NULL, NULL, ?, NULL, ?, 0, ?, ?)
            """,
            (
                language,
                normalized,
                json_dumps([]),
                LEMMA_SCHEMA_VERSION,
                created_at,
                created_at,
            ),
        )
        db.commit()
        row = db.execute(
            "SELECT * FROM lemmas WHERE language = ? AND normalized_lemma = ?",
            (language, normalized),
        ).fetchone()

    model = ensure_model(None)
    content, error = resolve_lemma_content(
        db,
        language,
        normalized,
        row,
        model,
        force_generate=False,
    )

    is_favorite = db.execute(
        "SELECT 1 FROM favorites WHERE item_type = 'lemma' AND item_id = ?",
        (row["id"],),
    ).fetchone()

    if not request.headers.get("HX-Request") and content is not None:
        increment_access("lemmas", row["id"])

    sentences = fetch_sentences_for_lemma(language, normalized)
    chat_messages = load_chat_messages(row["chat_json"])

    return render_template(
        "lemma_detail.html",
        language=language,
        lemma_text=normalized,
        lemma_display=lemma_display,
        lemma_id=row["id"],
        content=content,
        error=error,
        is_favorite=bool(is_favorite),
        sentences=sentences,
        chat_messages=chat_messages,
    )


@app.post("/<lang>/lemma/<lemma>/regenerate")
def lemma_regenerate(lang: str, lemma: str):
    language = require_language(lang)
    lemma_display = unquote(lemma)
    normalized = normalize_text(lemma_display)
    model = ensure_model(request.form.get("model"))
    session["model_choice"] = model
    session["model_choice_source"] = "user"
    db = get_db()
    row = db.execute(
        "SELECT * FROM lemmas WHERE language = ? AND normalized_lemma = ?",
        (language, normalized),
    ).fetchone()
    if row is None:
        abort(404)

    content, error = resolve_lemma_content(
        db,
        language,
        normalized,
        row,
        model,
        force_generate=True,
    )
    if content is not None:
        clear_lemma_chat(row["id"])
        chat_messages: list[dict] = []
    else:
        chat_messages = load_chat_messages(row["chat_json"])

    is_favorite = db.execute(
        "SELECT 1 FROM favorites WHERE item_type = 'lemma' AND item_id = ?",
        (row["id"],),
    ).fetchone()

    sentences = fetch_sentences_for_lemma(language, normalized)

    return render_template(
        "partials/lemma_content.html",
        language=language,
        lemma_text=normalized,
        lemma_display=lemma_display,
        lemma_id=row["id"],
        content=content,
        error=error,
        is_favorite=bool(is_favorite),
        sentences=sentences,
        chat_messages=chat_messages,
    )


@app.post("/<lang>/lemma/<lemma>/chat")
def lemma_chat(lang: str, lemma: str):
    language = require_language(lang)
    lemma_display = unquote(lemma)
    normalized = normalize_text(lemma_display)
    message = request.form.get("message", "").strip()
    model = ensure_chat_model(request.form.get("model"))
    content_model = ensure_model(None)
    session["chat_model_choice"] = model
    session["chat_model_choice_source"] = "user"
    db = get_db()
    row = db.execute(
        "SELECT * FROM lemmas WHERE language = ? AND normalized_lemma = ?",
        (language, normalized),
    ).fetchone()
    if row is None:
        abort(404)
    chat_messages = load_chat_messages(row["chat_json"])
    chat_error = None
    if message:
        now = utc_now()
        chat_messages.append({"role": "user", "content": message, "created_at": now})
        db.execute(
            "UPDATE lemmas SET chat_json = ?, updated_at = ? WHERE id = ?",
            (json_dumps(chat_messages), now, row["id"]),
        )
        db.commit()
        content, error = resolve_lemma_content(
            db,
            language,
            normalized,
            row,
            content_model,
            force_generate=False,
        )
        if content is None:
            chat_error = error or "Unable to load lemma context."
        else:
            sentence_rows = fetch_sentences_for_lemma(language, normalized)
            sentence_dicts = [dict(item) for item in sentence_rows]
            try:
                reply = generate_lemma_chat_reply(
                    language,
                    normalized,
                    content,
                    sentence_dicts,
                    chat_messages,
                    model=model,
                )
                if reply:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "created_at": utc_now(),
                            "model_used": model,
                        }
                    )
                    db.execute(
                        "UPDATE lemmas SET chat_json = ?, updated_at = ? WHERE id = ?",
                        (json_dumps(chat_messages), utc_now(), row["id"]),
                    )
                    db.commit()
                else:
                    chat_error = "No response returned. Please try again."
            except (LLMRequestError, LLMOutputError):
                chat_error = "LLM generation failed. Please try again."

    return render_template(
        "partials/chat_section.html",
        chat_id="lemma-chat",
        chat_title="Chat",
        chat_action=url_for("lemma_chat", lang=language, lemma=quote(normalized)),
        chat_placeholder="Ask about this lemma...",
        chat_item_label="lemma",
        chat_messages=chat_messages,
        chat_error=chat_error,
    )


@app.post("/<lang>/lemma/<lemma>/favorite")
def lemma_favorite(lang: str, lemma: str):
    language = require_language(lang)
    normalized = normalize_text(lemma)
    db = get_db()
    row = db.execute(
        "SELECT id FROM lemmas WHERE language = ? AND normalized_lemma = ?",
        (language, normalized),
    ).fetchone()
    if row is None:
        abort(404)
    is_favorite = toggle_favorite("lemma", row["id"])
    return render_template(
        "partials/favorite_star.html",
        item_type="lemma",
        is_favorite=is_favorite,
        target_id=f"lemma-favorite-{row['id']}",
        action=url_for("lemma_favorite", lang=language, lemma=quote(normalized)),
    )


@app.post("/toggle-glosses")
def toggle_glosses():
    show = not session.get("show_glosses", True)
    session["show_glosses"] = show
    return render_template("partials/gloss_toggle.html", show_glosses=show)


@app.post("/toggle-translations")
def toggle_translations():
    show = not session.get("show_translations", True)
    session["show_translations"] = show
    return render_template(
        "partials/translations_toggle.html", show_translations=show
    )


@app.route("/favorites")
def favorites():
    db = get_db()
    sentence_favorites = db.execute(
        """
        SELECT s.*, f.created_at AS favorited_at
        FROM favorites f
        JOIN sentences s ON s.id = f.item_id
        WHERE f.item_type = 'sentence'
        ORDER BY f.created_at DESC
        """
    ).fetchall()
    lemma_favorites = db.execute(
        """
        SELECT l.*, f.created_at AS favorited_at
        FROM favorites f
        JOIN lemmas l ON l.id = f.item_id
        WHERE f.item_type = 'lemma'
        ORDER BY f.created_at DESC
        """
    ).fetchall()
    return render_template(
        "favorites.html",
        sentence_favorites=sentence_favorites,
        lemma_favorites=lemma_favorites,
    )


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.route("/favicon.ico")
def favicon():
    return redirect(
        "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        "<rect width='100' height='100' fill='%230a0a0a'/><text x='50' y='65' "
        "font-size='48' text-anchor='middle' fill='%23ffffff'>SL</text></svg>"
    )


def main() -> None:
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "6543")),
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
