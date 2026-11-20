.PHONY: install lint test train serve docker-build

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .
	ruff format --check .

test:
	pytest

train:
	nascar-performance demo-data --output-dir data/demo
	nascar-performance train --data-dir data/demo

serve:
	uvicorn nascar_performance.api:app --reload

docker-build:
	docker build -t nascar-performance-lab .

