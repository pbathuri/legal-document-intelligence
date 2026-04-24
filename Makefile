.PHONY: install dev test lint scrape train ui clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

dev: install
	cp -n .env.example .env || true

test:
	LEGAL_INTEL_MOCK_LLM=1 QDRANT_URL=:memory: .venv/bin/pytest -q

lint:
	.venv/bin/ruff check src/ tests/

# Run all scrapers and generate training data
scrape:
	.venv/bin/legal-scrape igrs --max-results 20 --training --output data/raw/igrs_batch.json
	.venv/bin/legal-scrape ecourts --party-name "Test Party" --max-results 10 --training --output data/raw/ecourts_batch.json
	.venv/bin/legal-scrape kaveri --district "Bangalore Urban" --max-results 10 --training --output data/raw/kaveri_batch.json

train: scrape
	.venv/bin/legal-train-prep --data-dir data/raw --output-dir data/training

ui:
	LEGAL_INTEL_MOCK_LLM=1 QDRANT_URL=:memory: .venv/bin/streamlit run streamlit_app.py

# Production UI (real LLM)
ui-prod:
	LEGAL_INTEL_MOCK_LLM=0 .venv/bin/streamlit run streamlit_app.py

clean:
	rm -rf .venv data/raw data/processed data/training __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
