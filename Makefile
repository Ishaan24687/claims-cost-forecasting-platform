.PHONY: generate-data train serve test dashboard lint clean all

generate-data:
	python -m src.data.generate_claims

train:
	python -m src.models.train

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --tb=short

dashboard:
	streamlit run dashboard/streamlit_app.py --server.port 8501

lint:
	ruff check src/ tests/

clean:
	rm -rf data/*.csv models/*.pkl artifacts/ mlruns/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down -v

all: generate-data train test
