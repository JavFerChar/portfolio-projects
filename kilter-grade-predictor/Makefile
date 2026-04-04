.PHONY: install test lint format train serve dashboard docker-build docker-up

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

train:
	uv run python -m src.models.train

serve:
	uv run uvicorn src.serving.app:app --reload --port 8000

dashboard:
	uv run streamlit run src/serving/dashboard.py

docker-build:
	docker compose build

docker-up:
	docker compose up
