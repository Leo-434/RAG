dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

sync:
	uv sync

lint:
	uv run ruff check app/

format:
	uv run ruff format app/

.PHONY: dev sync lint format
