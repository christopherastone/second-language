# Specifications

## Overview

This project is a local web app for learning a foreign language (initially Slovenian) aimed at an American English speaker. The app teaches vocabulary and grammar *in context* by presenting sentences and word (lemma) pages generated with an LLM and cached in a local SQLite database.

## Goals / Non-goals

### Goals

* Teach foreign-language vocabulary and grammar in context to an American English speaker.
* Focus on reading and listening.
* Generate content on demand (to minimize LLM usage/costs) and cache results.

### Non-goals

* Writing or speaking practice.

## Technology Choices

* Flask (Python) back-end.
* HTMX for client-side interactivity.
* HTML and CSS for markup and styling.
* SQLite for the database.
* LLMs for translation/generation.

## Key Terms

* **Language**: identified by a short code in the URL (e.g., `sl`, `de`, `en`).
* **Sentence**: a single sentence of foreign-language text (either sourced from RSS headlines or generated on demand to exhibit uses of a particular word).
    * Each sentence is uniquely identified in the database and URLs by a hash of its full text string.
* **Token/word**: a clickable unit in a sentence (used to navigate to lemma pages).
    * Tokenization is mostly orthographic: split on whitespace and punctuation; keep word-internal apostrophes/hyphens when surrounded by letters/digits; emit punctuation as separate tokens.
    * Proper nouns are kept as a single token even if they contain internal whitespace.
    * Tokenization preserves the surface string; normalization (lowercasing/Unicode normalization) is used only for lookup.
Proper-noun vs lemma mapping is determined by the LLM analysis, not capitalization heuristics.
* **Lemma**: canonical dictionary form of a token.
    * Common noun lemma: nominative singular, lowercase.
    * Adjective lemma: masculine singular, lowercase.
    * Verb lemma: infinitive, lowercase.
    * Adverbs, Prepositions, Conjunctions, etc: lowercase.
    * Proper noun: the complete single or multi-word string, with proper capitalization.
    * Punctuation: has no corresponding lemma.
    * A lemma is uniquely identified in the database by its unicode-normalized string form.

## User Model

* The app requires login (username/password) to control LLM costs.
* The app does **not** track individual users for learning progress or per-user caches.
    * There may be more than one username/password.
    * Cached content and access counts are shared globally across logins.
* User accounts are configured locally and are not self-service.
* Each user is associated with a default foreign language (e.g., `sl`)

### Credential Provisioning and Management
* Usernames and password hashes are stored in the SQLite database in a `users` table.
* Initial users are created by a one-time command-line script/CLI, not through the web UI.
    * The script takes user input to insert, update, or remove user records.
* There is no in-app registration or password reset flow; adding/removing users is done via the admin script.

### Password Storage
* Passwords are never stored in plaintext.
* Passwords are hashed using a standard, salted password hashing algorithm.
* The userdatabase stores only:
    * `username`
    * `password_hash`
    * `default_language`
* If a password needs to be changed, an admin runs the provisioning script again to update the stored hash.

### Session / Cookie Policy
* Flask session cookies are used to maintain login state.
* Sessions are configured as “remember me”:
    * The session cookie is persistent and does not expire until the browser clears cookies, or until the server secret key changes.
* Session cookies are:
    * `HttpOnly`
    * `Secure` when running over HTTPS
    * Restricted to the app’s path/domain.
* A per-request check ensures a valid session before serving any application page (including sentence and lemma pages).
* CSRF protection is enabled for the login form and any state-changing endpoints (e.g., regeneration actions).

## Pages and Behaviors

### Sentence List Page (per language)

* There is a per-language page listing sentences in the database.
* The list shows how many times each sentence has been accessed.
    * A full load of a sentence page counts as one access.
* Default sort order is newest to oldest (based on insert time).
* The URL of the page is `/<lang>/`, e.g., `/sl/`.
* There is no pagination, but the page is limited to 100 sentences.

### Sentence Detail Page

* Selecting a specific sentence shows a page for that sentence.
* The page starts with the sentence in bold. The words are in freeform (not a table), but below each word is the corresponding gloss (a literal translation plus tags taken from Leipzig Glossing Notation).
    * So effectively each word (or proper noun) is a two-row table.
    * Each word links to the corresponding lemma page (which will be generated on demand).
    * Allowable gloss tags are: `1`, `2`, `3`, `sg`, `du`, `pl`, `nom`, `gen`, `dat`, `acc`, `ins`, `loc`, `refl`, `m`, `f`, `n`.
* Next is a list defining all proper nouns in the sentence (e.g., cities, sports teams, bands) unless they are already well-known to an American .
    * Proper nouns should be listed in nominative case in these definitions, even if they are declined in the sentence.
    * If there are no such proper nouns in the sentence, this section is omitted.
* Next is a list of grammar features that might be surprising to an American English speaker.
    * Exclude very basic rules (e.g., adjectives declining to match gender; verbs conjugating to match number).
    * Prefer differences from English (tense/mood/number differences; word order changes required for natural English).
    * If there are no particularly surprising grammar features, this section is omitted.
* At the bottom of the page, there is a button to regenerate the page next to a dropdown that allows selecting from the supported LLM models, and a label saying what model was used to generated the page.
    * The page is regenerated and reloaded when the button is clicked.
    * Regenerating a page replaces it in the database; no history is kept.

### Lemma (Word) Detail Page

* Selecting a specific word/token from a sentence navigates to a page for the lemma of that word.
* The lemma page includes:
    * The lemma.
    * A translation of the lemma.
    * At least three example sentences containing a conjugated/declined form of the lemma.
        * One sentence will be the sentence where the word was first clicked.
        * The other two sentences must be generated by an LLM. 
            * These should use common words and grammar features. 
            * They should be different from each other. 
            * They should use the lemma in a gramatically correct way (declined/inflected/conjugated as appropriate.
        * Words in these sentences are linked to their lemma pages.
    * A list of common, related words and their translations, including:
        * common words that are etymologically related
            * E.g., in Slovenian, `ključ` (lock) is related to `ključavnica`
        * common words that sound similar but have different meanings.
            * E.g., in Slovenian, `kokoš` (hen) is similar to `kokos`
        * words that have similar meanings but are not interchangeable (i.e., disguinshable meanings)
            * E.g., in Slovenian, `enako` (same) and `isto` (identical).
        * Do not include declined or conjugated forms of the same lemma on this list
        * The list should be at most 6 words long, and can be shorter or even omitted.
* If the LLM generates example sentences for a lemma, those sentences do not go into the database until/unless the user clicks on them.

## Data Storage (SQLite)

* The database caches all information needed to render sentence and lemma pages without re-calling the LLM.
* The database tracks how many times each sentence page and each lemma page has been accessed.

**MISSING/OPEN**

* Database schema (tables/columns/indexes/constraints).
* How cached content is stored (structured JSON blob? normalized tables?).
* Caching/versioning policy (prompt/model version recorded? last updated timestamps?).
* How to handle invalid/outdated cached content after changes to prompts or rendering.

## RSS Ingestion

* There is a way to load sentences from an RSS feed.
    * Periodically (e.g., every six hours), sentences are taken from all headlines of articles in a hard-coded foreign-language RSS feed.
* The database tracks which RSS article IDs have already been seen/processed so only new sentences are added.
* RSS headlines must be split into individual sentences.
    * Most headlines are a single sentence, but occasionally there is more than one.
    * Database entries should be created for each sentence.

**MISSING/OPEN**

* Which RSS feed(s) are used initially, and how feeds are configured per language.
* What constitutes an “RSS article id” (GUID? link? hash?), and deduplication strategy.
* How sentence splitting is implemented (rule-based vs NLP library; language-specific heuristics).
* How periodic scheduling is implemented for local deployment (cron? APScheduler? manual endpoint?).

## LLM Support

* API keys are supplied externally via environment variables.
* The back-end initially supports only `OpenAI gpt-5-mini`.
* The system should be extensible to other OpenAI and Anthropic LLMs.

**MISSING/OPEN**

* Exact list of supported models and how model selection is represented/stored.
* Request/response handling requirements (timeouts, retries, rate limiting, malformed outputs).
* Cost controls (limits on regeneration frequency, daily caps, etc.).

## Browser History / Navigation

* Browser history should match pages visited.
    * Example: if the user visits a lemma, then a sentence containing that lemma, then another lemma in that sentence, the back button should return to the sentence page and then to the original lemma.

**MISSING/OPEN**

* HTMX navigation approach to ensure proper history behavior (`hx-push-url` strategy, full vs partial page loads).

## Deployment

* The app should support local deployment.

**MISSING/OPEN**

* How the app is started locally (Flask dev server vs production server).
* Required environment variables beyond API keys (e.g., Flask secret key, DB path).
* Initial setup steps (creating DB schema/migrations; initial credentials).

## Open Questions (Consolidated)

* Tokenization:
    * What are the tokenization rules per language, and how do we handle punctuation/clitics/multiword named entities?
* LLM contracts:
    * What is the exact structured output for sentence generation (gloss lines, lemmas, morph tags, proper nouns, grammar notes)?
    * What is the exact structured output for lemma generation (translation, example sentences, token/lemma links)?
* Caching/versioning:
    * Do we keep history of regenerations or only the latest cached version?
    * How do we handle prompt/schema changes over time?
* Access counting:
    * What counts as an “access” (per request, per session, excluding HTMX fragments, etc.)?
* RSS:
    * Which feeds are used, and what is the deduplication key?
    * How is the 6-hour schedule implemented locally?
* Auth:
    * How are usernames/passwords provisioned, stored, and rotated locally?
    * What is the precise session lifetime/cookie configuration?