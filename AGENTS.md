# Repository Guidelines

## Project Structure & Module Organization
This repository includes the product spec plus the Flask implementation. Key top-level paths:
- `SPEC.md`: authoritative product and implementation specification.
- `CLAUDE.md`: working notes and commands for contributors/agents.
- `README.md`: project name placeholder.
- `app.py`, `config.py`, `db.py`, `llm.py`, `rss.py`, `normalization.py`: application modules.
- `templates/`, `static/`, `input.css`: UI templates and Tailwind assets.
- `init_db.py`, `cli`: database initializer and admin CLI.
- `llm_raw_sentence.py`: debug script to print raw LLM output for a fixed sentence.
- `llm_raw_lemma.py`: debug script to print raw LLM output for a fixed lemma.
- `pyproject.toml`: dependency and project metadata.

As implementation evolves, keep source and templates grouped logically. If you add new top-level paths, update this section.

## Build, Test, and Development Commands
Use the tooling defined in `CLAUDE.md`:
- `uv sync`: install dependencies.
- `tailwindcss -i input.css -o static/output.css`: build Tailwind CSS.
- `uv run python init_db.py`: initialize SQLite.
- `uv run python cli set-password <password>` / `uv run python cli set-language <lang>`: set admin settings.
- `uv run flask run`: start dev server.
- `uv run gunicorn app:app`: production server.

## Coding Style & Naming Conventions
- Python/Flask project; prefer small, clear modules over large files.
- Use consistent 4-space indentation in Python; keep HTML/Tailwind tidy and minimal.
- Normalize language codes to lowercase two-letter strings (e.g., `sl`, `de`).
- Keep database and JSON schema names aligned with `SPEC.md`.

## Testing Guidelines
No test framework is defined yet. If adding tests, document the framework and conventions here (e.g., `tests/` folder, `test_*.py` names, coverage expectations) and add commands under Build/Test.

## Commit & Pull Request Guidelines
Recent commits use short, imperative, sentence-case subjects without prefixes (e.g., “Simplify to single-user architecture”). Follow that pattern.
For PRs, include: a concise summary, the SPEC/behavior changes, and any migration/setup steps (e.g., new env vars, DB changes). Add screenshots only if UI changes are involved.

## Security & Configuration Tips
All required env vars are `SECRET_KEY` and `OPENAI_API_KEY`. The database path is hard-coded to `./data/app.db`. Treat `feeds.yaml` as required configuration; malformed or missing files should fail fast.
