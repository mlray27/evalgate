.PHONY: install db seed test lint chat serve eval eval-oracle

install:
	uv sync

db:
	docker compose up -d postgres

seed:
	uv run python -m evalgate.seed

test:
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

chat:
	uv run evalgate chat

serve:
	uv run uvicorn evalgate.api:app --reload

eval:
	uv run evalgate eval --split train

eval-oracle:
	uv run evalgate eval --oracle --split all --concurrency 16
