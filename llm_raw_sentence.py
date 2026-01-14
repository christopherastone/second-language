#!/usr/bin/env python3
import argparse
import logging

from config import DEFAULT_MODEL, MODEL_CHOICES
from llm import (
    SENTENCE_SCHEMA_NAME,
    SENTENCE_SCHEMA_OUTPUT,
    SENTENCE_SYSTEM_PROMPT,
    request_raw_schema_text,
)

SENTENCE_TEXT = (
    "Ali bo premier Robert Golob odstopil? Sam pravi: "
    "»Komunikacija z ministrico ni kršitev integritete«"
)
LANGUAGE = "sl"


def build_prompt(sentence_text: str, language: str) -> str:
    return (
        "Generate sentence analysis for the following.\n"
        f"Language: {language}\n"
        f"Sentence (normalized, must match exactly): {sentence_text}\n\n"
        "Output JSON with keys: sentence_text, tokens, proper_nouns, grammar_notes."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print raw LLM JSON output for a specific sentence."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help="OpenAI model to use",
    )
    args = parser.parse_args()

    user_prompt = build_prompt(SENTENCE_TEXT, LANGUAGE)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("llm_raw_sentence")
    logger.info("System prompt:\n%s", SENTENCE_SYSTEM_PROMPT)
    logger.info("User prompt:\n%s", user_prompt)
    output_text = request_raw_schema_text(
        SENTENCE_SYSTEM_PROMPT,
        user_prompt,
        args.model,
        SENTENCE_SCHEMA_OUTPUT,
        SENTENCE_SCHEMA_NAME,
    )
    print(output_text)


if __name__ == "__main__":
    main()
