#!/usr/bin/env python3
import argparse
import logging

from config import DEFAULT_MODEL, MODEL_CHOICES
from llm import LEMMA_SYSTEM_PROMPT, request_raw_text

LEMMA_TEXT = "negotov"
LANGUAGE = "sl"


def build_prompt(lemma_text: str, language: str) -> str:
    return (
        "Generate lemma details for the following.\n"
        f"Language: {language}\n"
        f"Lemma: {lemma_text}\n"
        f"Normalized lemma (must match exactly): {lemma_text}\n\n"
        "Output JSON with keys: lemma, normalized_lemma, translation, related_words."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print raw LLM JSON output for a specific lemma."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help="OpenAI model to use",
    )
    parser.add_argument(
        "lemma",
        nargs="?",
        default=LEMMA_TEXT,
        help="Lemma to look up (default: negotov)",
    )
    args = parser.parse_args()

    user_prompt = build_prompt(args.lemma, LANGUAGE)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("llm_raw_lemma")
    logger.info("System prompt:\n%s", LEMMA_SYSTEM_PROMPT)
    logger.info("User prompt:\n%s", user_prompt)
    output_text = request_raw_text(LEMMA_SYSTEM_PROMPT, user_prompt, args.model)
    print(output_text)


if __name__ == "__main__":
    main()
