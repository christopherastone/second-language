import hashlib
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def hash_sentence(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:16].lower()


def token_has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text or "")

