.PHONY: install test lint train serve clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100

train:
	python src/models/train.py

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache htmlcov .coverage
