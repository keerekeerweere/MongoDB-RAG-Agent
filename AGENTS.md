# Repository Guidelines

## Project Structure & Modules
- `src/` holds the MongoDB RAG agent: `agent.py` (Pydantic AI agent), `tools.py` (vector/text/hybrid search), `dependencies.py` (Mongo client wiring), `providers.py` (LLM/embedding clients), `cli.py` (Rich CLI), and `prompts.py`. Ingestion lives in `src/ingestion/` (`ingest.py`, `chunker.py`, `embedder.py`). Configuration is in `settings.py` with `.env` support.
- `documents/` contains ingestible content; `examples/` is a read-only Postgres reference.
- `test_scripts/` and `comprehensive_e2e_test.py` provide validation and smoke tests; keep new tests beside related code.

## Setup, Build, and Run
- Install deps: `uv sync` (creates/updates the venv). Activate `.venv` per your shell.
- Config check: `uv run python -m src.test_config`.
- Ingest documents: `uv run python -m src.ingestion.ingest -d ./documents`.
- Run the agent CLI: `uv run python -m src.cli`.

## Coding Style & Naming
- Python 3.10+, 4-space indentation, type hints for public functions. Use docstrings only where behavior or params are non-obvious.
- Format with `uv run black src test_scripts` and lint with `uv run ruff check`.
- Naming: modules and functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`; CLI commands and tools should stay descriptive (e.g., `search_knowledge_base`).

## Testing Guidelines
- Primary framework: `pytest`. Common runs: `uv run pytest test_scripts/test_search.py`, `uv run pytest test_scripts/test_rag_pipeline.py`, or full suite `uv run pytest test_scripts`.
- End-to-end: `uv run python comprehensive_e2e_test.py` before major changes.
- Add tests mirroring new behaviors; prefer async tests where code is async. Name tests `test_<behavior>` and place fixtures near usage.

## Commit & Pull Request Guidelines
- Commit messages: concise imperative or past-tense summaries, mirroring existing history (e.g., “add local embedding models”, “Complete Phase 2: ...”). One topic per commit.
- PRs should include: a clear description, linked issue/plan (if applicable), notable configs or migrations, and screenshots or logs for CLI/test output. Call out any breaking changes or required Atlas index updates.

## Security & Configuration
- Do not commit `.env` or credentials; use `.env.example` as the template. Verify `MONGODB_URI`, `LLM_API_KEY`, and `EMBEDDING_API_KEY` locally via `src.test_config`.
- When modifying ingestion or search, note any Atlas Search/Vector index changes so reviewers can update their clusters.***
