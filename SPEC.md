# Specifications

## Project Goals

* Teach foreign-language vocabulary and grammar in context to an American English speaker.
* Focused on reading and listening, not writing or speaking.

## Technologies

* Flask (Python) as the back-end
* HTMX for client-side interactivity
* HTML and CSS for markup and styling
* SQLite for database 
* LLMs for language translation and generation

## Project Components

### Individual Sentence pages

* Upon selecting a specific sentence, there is a page for that sentence.
* The page includes:
    * The fully glossed sentence, including
        * The original sentence
        * The glossed sentence (consisting of translated lemmas + morphological features using Leipzing Glossing Notation)
        * A translation of the sentence into natural English.
    * Definition of any proper nouns that might be unfamiliar to an American (e.g., cities, sports teams, bands, etc.)
        * In the definitions, the proper nouns are given in nominative case
    * Explanation of any grammar features in the sentence that might be surprising to someone speaking American English.
        * This does not include very basic rules, such as adjectives declining to match the gender of the noun they modify
          or verbs conjugating to match the number of the subject.
        * Places where tense/mood/number etc. are different from English are most interesting, or if the order of the words 
          is necessarily different from English.

    * A button to regenerate the page, selecting from the supported LLM models.

### Individual Word pages

* Upon selecting a specific word from a sentence, there is a page for the *lemma* of that word
    * If the word is a noun, its lemma is the nominative singular form
    * If the word is an adjective, its lemma is the masculine singular form
    * If the word is a verb, its lemma is the infinitive

* The page includes
    * The lemma
    * A translation of the lemma
    * At least three sentences including a conjugated/declined form of the lemma

### Sentence Page

* There should be a main (per-language) pages listing all sentences currently in the database and how many times they have been accessed.
     * Default sorting is newest to oldest

### Database

* There is a database to cache all information necessary for generating Sentence and Word pages.
* There is a way to load sentences from an RSS feed
    * Periodically (e.g., every six hours), sentences are taken from all headlines of articles in a hard-coded foreign-language RSS feed.
* The database also tracks how many times each sentence page and word page has been accessed.
* The database also tracks which RSS article ids have already been seen and processed, so that we only add new sentences.

## Functional and Nonfunctional Requirements

* Pages are generated only on demand.
    * On first access, an LLM is called to gather relevant information.
    * On later accesses, cached information from the database is used.
        * Exception: if the user clicks on Regenerate, that sentence page has its explanatory information regenerated and the database information is replaced before redisplaying the page.
    * If the LLM makes up sample sentences for a word, that sentence does not go into the database
      until/unless the user clicks on it.
* On the lemma pages, words in the sample sentences are linked to their lemma pages.
* The language should initially support Slovenian, but should be extensible to other languages.
    * E.g., the URL for a word page might be `/sl/lemma` (Slovenian) or `/de/lemma` (German).
    * `/en/gift` and `/de/gift` need to be different in the database.
    * Hard-coded RSS feeds should identify which language they are contributing sentences to.
* API keys will be supplied externally via environment variables.
* The back-end should initially support only OpenAI gpt-5-mini, but should be extensible to other OpenAI and Anthropic LLMs.
* Because using the app incurs LLM costs, the app must support login via username and password.
    * A login page should appear on first access.
    * Logins should not expire until the browser clears cookies.
* Browser history should match the pages visited
    * E.g., if the user visits a word, then a sentence containing that word, then another word in that sentence, the back button should take us back to the sentence page and then to the original word.
* RSS headlines must be split into individual sentences. 
    * Most headlines are a single sentence, but occasionally there is more than one.
    * Database entries should be created for each sentence.
 
 * The app should support local deployment. 
 * The app does not need to track individual users.
    * There might be more than one username and password, but a single database of cached data, single count of accesses for each word page, etc.