import json
import logging
import time
from typing import Any

from jsonschema import ValidationError, validate
from openai import OpenAI, OpenAIError

from config import DEFAULT_MODEL, MODEL_CHOICES, require_env
from normalization import normalize_text, token_has_alpha

SENTENCE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SentenceDetailGenerationV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "sentence_text",
        "natural_english_translation",
        "tokens",
        "proper_nouns",
        "grammar_notes",
    ],
    "properties": {
        "sentence_text": {"type": "string", "minLength": 1},
        "natural_english_translation": {"type": "string", "minLength": 1},
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["surface"],
                "properties": {
                    "surface": {"type": "string", "minLength": 1},
                    "lemma": {"type": "string", "minLength": 1},
                    "translation": {"type": "string", "minLength": 1},
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
            },
        },
        "proper_nouns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["nominative", "definition"],
                "properties": {
                    "nominative": {"type": "string", "minLength": 1},
                    "definition": {"type": "string", "minLength": 1},
                },
            },
        },
        "grammar_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "note"],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "note": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}
SENTENCE_SCHEMA_OUTPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "sentence_text",
        "natural_english_translation",
        "tokens",
        "proper_nouns",
        "grammar_notes",
    ],
    "properties": {
        "sentence_text": {"type": "string"},
        "natural_english_translation": {"type": "string"},
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["surface", "lemma", "translation", "tags"],
                "properties": {
                    "surface": {"type": "string"},
                    "lemma": {"type": "string"},
                    "translation": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "proper_nouns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["nominative", "definition"],
                "properties": {
                    "nominative": {"type": "string"},
                    "definition": {"type": "string"},
                },
            },
        },
        "grammar_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "note"],
                "properties": {
                    "title": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
}
SENTENCE_SCHEMA_NAME = "SentenceDetailGenerationV1"

LEMMA_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LemmaPageGenerationV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lemma",
        "normalized_lemma",
        "translation",
        "related_words",
    ],
    "properties": {
        "lemma": {"type": "string", "minLength": 1},
        "normalized_lemma": {"type": "string", "minLength": 1},
        "translation": {"type": "string", "minLength": 1},
        "related_words": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["word", "normalized_lemma", "translation", "note"],
                "properties": {
                    "word": {"type": "string", "minLength": 1},
                    "normalized_lemma": {"type": "string", "minLength": 1},
                    "translation": {"type": "string", "minLength": 1},
                    "note": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}
LEMMA_SCHEMA_OUTPUT = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lemma",
        "normalized_lemma",
        "translation",
        "related_words",
    ],
    "properties": {
        "lemma": {"type": "string"},
        "normalized_lemma": {"type": "string"},
        "translation": {"type": "string"},
        "related_words": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["word", "normalized_lemma", "translation", "note"],
                "properties": {
                    "word": {"type": "string"},
                    "normalized_lemma": {"type": "string"},
                    "translation": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
}
LEMMA_SCHEMA_NAME = "LemmaPageGenerationV1"

TRANSIENT_STATUS = {429, 500, 502, 503, 504}

_client: OpenAI | None = None
logger = logging.getLogger("second_language.llm")


class LLMRequestError(RuntimeError):
    pass


class LLMOutputError(RuntimeError):
    pass


def _log_openai_error(model: str, exc: Exception) -> None:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if body:
        logger.warning(
            "LLM request failed model=%s status=%s body=%s", model, status, body
        )
        return
    logger.warning("LLM request failed model=%s status=%s error=%s", model, status, exc)


def _extract_output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", None)
    if output:
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for piece in content:
                piece_text = getattr(piece, "text", None)
                if piece_text:
                    parts.append(piece_text)
        if parts:
            return "".join(parts)
    raise LLMOutputError("No output text from LLM response")


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = require_env("OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key, timeout=180, max_retries=0)
    return _client


def validate_openai_key() -> None:
    client = get_client()
    try:
        client.models.list()
    except OpenAIError as exc:
        raise RuntimeError("OpenAI API key validation failed") from exc


def _is_transient_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in TRANSIENT_STATUS


def _is_unsupported_text_format(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if not body:
        return False
    param = body.get("param")
    message = str(body.get("message", ""))
    if param and "text" in str(param):
        return True
    return "text" in message or "format" in message


def request_raw_text(system_prompt: str, user_prompt: str, model: str) -> str:
    client = get_client()
    start = time.monotonic()
    logger.info("LLM request started model=%s", model)

    def make_request(use_text_format: bool):
        params: dict[str, Any] = {
            "model": model,
            "input": user_prompt,
            "instructions": system_prompt,
        }
        if use_text_format:
            params["text"] = {"format": {"type": "json_object"}}
        return client.responses.create(**params)

    try:
        response = make_request(use_text_format=True)
    except OpenAIError as exc:
        if _is_unsupported_text_format(exc):
            logger.warning(
                "LLM text format unsupported; retrying without it model=%s", model
            )
            response = make_request(use_text_format=False)
        else:
            raise
    logger.info("LLM response received model=%s", model)
    text = _extract_output_text(response)
    elapsed = time.monotonic() - start
    logger.info("LLM request completed model=%s elapsed=%.2fs", model, elapsed)
    return text


def request_raw_schema_text(
    system_prompt: str,
    user_prompt: str,
    model: str,
    schema: dict[str, Any],
    schema_name: str,
) -> str:
    client = get_client()
    start = time.monotonic()
    logger.info("LLM request started model=%s", model)
    logger.info(system_prompt)
    logger.info(user_prompt)

    def make_request(use_text_format: bool):
        params: dict[str, Any] = {
            "model": model,
            "input": user_prompt,
            "instructions": system_prompt,
        }
        if use_text_format:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            }
        return client.responses.create(**params)

    try:
        response = make_request(use_text_format=True)
    except OpenAIError as exc:
        if _is_unsupported_text_format(exc):
            body = getattr(exc, "body", None)
            if body:
                logger.warning(
                    "LLM structured output rejected model=%s body=%s", model, body
                )
            logger.warning(
                "LLM structured output unsupported; retrying without it model=%s",
                model,
            )
            response = make_request(use_text_format=False)
        else:
            raise
    logger.info("LLM response received model=%s", model)
    text = _extract_output_text(response)
    elapsed = time.monotonic() - start
    logger.info("LLM request completed model=%s elapsed=%.2fs", model, elapsed)
    return text


def _request_json_schema(
    system_prompt: str,
    user_prompt: str,
    model: str,
    schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    text = request_raw_schema_text(
        system_prompt,
        user_prompt,
        model,
        schema,
        schema_name,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM JSON decode failed model=%s text_prefix=%r", model, text[:200]
        )
        raise exc


def _validate_sentence_payload(payload: dict[str, Any], sentence_text: str) -> None:
    _normalize_sentence_payload(payload)
    validate(instance=payload, schema=SENTENCE_SCHEMA)
    if payload["sentence_text"] != sentence_text:
        raise LLMOutputError("sentence_text mismatch")
    if not isinstance(payload.get("natural_english_translation"), str):
        raise LLMOutputError("natural_english_translation missing")
    for token in payload["tokens"]:
        if not isinstance(token, dict):
            continue
        surface = token.get("surface", "")
        if token_has_alpha(surface):
            for field in ("lemma", "translation", "tags"):
                if field not in token:
                    raise LLMOutputError(f"token missing {field}")


def _normalize_sentence_payload(payload: dict[str, Any]) -> None:
    payload.pop("model_used", None)
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        tokens = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        if "surface" not in token:
            if "text" in token:
                token["surface"] = token.pop("text")
            elif "token" in token:
                token["surface"] = token.pop("token")
        token.pop("leading", None)
        token.pop("is_punct", None)
        token.pop("pos", None)
        tags = token.get("tags")
        if isinstance(tags, list):
            normalized = []
            for tag in tags:
                if isinstance(tag, str):
                    normalized.append(tag.strip().lower())
                else:
                    normalized.append(tag)
            token["tags"] = normalized
    payload["tokens"] = tokens

    grammar_notes = payload.get("grammar_notes")
    if isinstance(grammar_notes, list):
        normalized_notes = []
        for note in grammar_notes:
            if isinstance(note, str):
                note_text = note.strip()
                if note_text:
                    normalized_notes.append({"title": "Note", "note": note_text})
                continue
            if isinstance(note, dict):
                title = note.get("title")
                note_text = note.get("note")
                if isinstance(title, str) and isinstance(note_text, str):
                    title = title.strip()
                    note_text = note_text.strip()
                    if title and note_text:
                        normalized_notes.append({"title": title, "note": note_text})
                        continue
                if isinstance(note_text, str):
                    note_text = note_text.strip()
                    if note_text:
                        normalized_notes.append({"title": "Note", "note": note_text})
                        continue
                if isinstance(title, str):
                    title = title.strip()
                    if title:
                        normalized_notes.append({"title": title, "note": title})
                        continue
        payload["grammar_notes"] = normalized_notes

    proper_nouns = payload.get("proper_nouns")
    if isinstance(proper_nouns, list):
        normalized_proper = []
        for item in proper_nouns:
            if isinstance(item, dict):
                nominative = item.get("nominative")
                definition = item.get("definition")
                if isinstance(nominative, str) and isinstance(definition, str):
                    nominative = nominative.strip()
                    definition = definition.strip()
                    if nominative and definition:
                        normalized_proper.append(
                            {"nominative": nominative, "definition": definition}
                        )
                        continue
            if isinstance(item, str):
                text = item.strip()
                if ":" in text:
                    head, tail = text.split(":", 1)
                    head = head.strip()
                    tail = tail.strip()
                    if head and tail:
                        normalized_proper.append(
                            {"nominative": head, "definition": tail}
                        )
        payload["proper_nouns"] = normalized_proper


def _normalize_lemma_payload(payload: dict[str, Any]) -> None:
    payload.pop("model_used", None)
    related = payload.get("related_words")
    if not isinstance(related, list):
        return
    normalized_related = []
    for item in related:
        if not isinstance(item, dict):
            continue
        if "word" not in item and "lemma" in item:
            item["word"] = item.get("lemma")
        if "normalized_lemma" not in item and "word" in item:
            item["normalized_lemma"] = normalize_text(item.get("word", ""))
        word = item.get("word")
        normalized_lemma = item.get("normalized_lemma")
        translation = item.get("translation")
        note = item.get("note")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (word, normalized_lemma, translation, note)
        ):
            continue
        normalized_related.append(
            {
                "word": word.strip(),
                "normalized_lemma": normalized_lemma.strip(),
                "translation": translation.strip(),
                "note": note.strip(),
            }
        )
    payload["related_words"] = normalized_related


def _validate_lemma_payload(payload: dict[str, Any], normalized_lemma: str) -> None:
    _normalize_lemma_payload(payload)
    validate(instance=payload, schema=LEMMA_SCHEMA)
    if normalize_text(payload["normalized_lemma"]) != normalized_lemma:
        raise LLMOutputError("normalized_lemma mismatch")


SENTENCE_SYSTEM_PROMPT = (
    "You are a careful linguist turning sentences into JSON for a language learning app. "
    "Return only JSON that matches the required schema. Do not include any extra keys. "
    "Tokens must be in reading order. Each token must include a `surface` field with the exact text. "
    "Use a single token for multi-word proper nouns "
    "(including internal spaces). Punctuation must be separate tokens. "
    "Include a natural_english_translation for the whole sentence. "
    "For non-punctuation tokens, include their lemma, their English translation, and relevant grammatical tags. "
    "The lemma is the dictionary form of the word, e.g., infinitive for verbs (including 'se' if reflexive), nominative for nouns, masculine singular for adjectives. "
    "Translate helper verbs as AUX (instead of 'be' or 'to be') and participles as PTCP. "
    "Tags must be lowercase and selected from this list only: "
    "1, 2, 3, f, nsg, du, pl, nom, gen, dat, acc, ins, loc, refl, ptcp. "
    "Include the f tag only when it is semantically relevant that the word is feminine. "
    "Do not tag prepositions, conjunctions, or auxiliary verbs. "
    "proper_nouns must be an array of objects with nominative and definition. "
    "Include unfamiliar proper nouns for most Americans only: exclude globally famous names "
    "(e.g., Paris, Mozart) but include local figures, places, and organizations. "
    "Include famous places if the English name differs significantly (e.g., Dunaj for Vienna). Explain who people are (e.g., 'Slovenian president'). "
    "Detect proper nouns in any grammatical case and provide the nominative form in the definition. "
    "grammar_notes must be an array of objects with title and note. "
    "Only include surprising grammar features for an English speaker; exclude basics like adjective gender "
    "agreement or verb number agreement. Focus on tense/mood/aspect differences, unusual word order, "
    "or constructions without direct English equivalents. "
    "If a surface form is ambiguous, choose the single most contextually appropriate lemma. "
    "Include arrays even if empty."
)

LEMMA_SYSTEM_PROMPT = (
    "You are a careful linguist producing JSON for a language learning app. "
    "Return only JSON that matches the required schema. Do not include any extra keys. "
    "Provide translation as a single English string. If the word is polysemous, "
    'combine 2-3 meanings into one string separated by semicolons (e.g., "firm; steady; certain"). '
    "Related words must be linguistically relevant, "
    "and include a short note explaining the relationship. "
    "Include up to 8 related entries; each must include word, normalized_lemma, translation, and note, "
    "and can be a phrase. "
    'Prefer etymologically related but distinct forms (e.g., "lep", "lepoten", "lepta", and "lepšati se" are related; "gora" and "gorica" are related), easy-to-confuse (e.g., "kokos" and "kokoš", or "brat" and "brati", or "učiti" and "naučiti se"), exact antonyms (e.g., "varen" and "nevaren"); useful contrasts (e.g., "vedeti" and "znati"). '
    "only claim etymological relationships when you are confident—otherwise describe it as a related concept "
    "or contrast instead of asserting shared origin. "
    "All related words must be in the target language provided in the prompt; omit any entry you are not sure belongs. "
    "Exclude identical lemma variants or mere inflections."
)


def generate_sentence_content(
    language: str,
    sentence_text: str,
    model: str | None = None,
) -> dict[str, Any]:
    model = model or DEFAULT_MODEL
    if model not in MODEL_CHOICES:
        raise LLMRequestError(f"Unsupported model: {model}")

    user_prompt = (
        "Generate sentence analysis for the following.\n"
        f"Language: {language}\n"
        f"Sentence (normalized, must match exactly): {sentence_text}\n\n"
        "Output JSON with keys: sentence_text, natural_english_translation, tokens, proper_nouns, grammar_notes."
    )

    transient_retries = 2
    invalid_retries = 1
    attempt = 0
    while True:
        try:
            attempt += 1
            payload = _request_json_schema(
                SENTENCE_SYSTEM_PROMPT,
                user_prompt,
                model,
                SENTENCE_SCHEMA_OUTPUT,
                SENTENCE_SCHEMA_NAME,
            )
            logger.debug("LLM returned payload: %s", payload)
            _validate_sentence_payload(payload, sentence_text)
            payload["model_used"] = model
            logger.debug("Successfully validated LLM payload")
            return payload
        except (json.JSONDecodeError, ValidationError, LLMOutputError) as exc:
            if invalid_retries > 0:
                logger.warning(
                    "LLM output invalid, retrying model=%s attempt=%s remaining=%s error=%s",
                    model,
                    attempt,
                    invalid_retries,
                    exc,
                )
                invalid_retries -= 1
                continue
            logger.warning("LLM output invalid model=%s error=%s", model, exc)
            raise LLMOutputError("Invalid LLM output") from exc
        except OpenAIError as exc:
            if _is_transient_error(exc) and transient_retries > 0:
                logger.warning(
                    "LLM transient error, retrying model=%s attempt=%s remaining=%s",
                    model,
                    attempt,
                    transient_retries,
                )
                transient_retries -= 1
                continue
            _log_openai_error(model, exc)
            raise LLMRequestError("LLM request failed") from exc


def generate_lemma_content(
    language: str,
    lemma: str,
    normalized_lemma: str,
    model: str | None = None,
) -> dict[str, Any]:
    model = model or DEFAULT_MODEL
    if model not in MODEL_CHOICES:
        raise LLMRequestError(f"Unsupported model: {model}")

    user_prompt = (
        "Generate lemma details for the following.\n"
        f"Language: {language}\n"
        f"Lemma: {lemma}\n"
        f"Normalized lemma (must match exactly): {normalized_lemma}\n\n"
        "All related words must be in the target language above.\n"
        "Output JSON with keys: lemma, normalized_lemma, translation, related_words."
    )

    transient_retries = 2
    invalid_retries = 1
    attempt = 0
    while True:
        try:
            attempt += 1
            payload = _request_json_schema(
                LEMMA_SYSTEM_PROMPT,
                user_prompt,
                model,
                LEMMA_SCHEMA_OUTPUT,
                LEMMA_SCHEMA_NAME,
            )
            _validate_lemma_payload(payload, normalized_lemma)
            logger.debug("Successfully validated LLM lemma payload")
            payload["model_used"] = model
            return payload
        except (json.JSONDecodeError, ValidationError, LLMOutputError) as exc:
            if invalid_retries > 0:
                logger.warning(
                    "LLM output invalid, retrying model=%s attempt=%s remaining=%s error=%s",
                    model,
                    attempt,
                    invalid_retries,
                    exc,
                )
                invalid_retries -= 1
                continue
            logger.warning("LLM output invalid model=%s error=%s", model, exc)
            raise LLMOutputError("Invalid LLM output") from exc
        except OpenAIError as exc:
            if _is_transient_error(exc) and transient_retries > 0:
                logger.warning(
                    "LLM transient error, retrying model=%s attempt=%s remaining=%s",
                    model,
                    attempt,
                    transient_retries,
                )
                transient_retries -= 1
                continue
            _log_openai_error(model, exc)
            raise LLMRequestError("LLM request failed") from exc
