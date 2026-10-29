.PHONY: install data train train-fast mlflow-ui serve test lint clean

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python -m src.models.train

train-fast:
	python -m src.models.train --fast

mlflow-ui:
	mlflow ui --port 5000

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
