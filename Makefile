.PHONY: lint test run clean

lint:
	flake8 .

test:
	python -m pytest test_models_performance.py

run:
	docker-compose up -d

clean:
	docker-compose down
	find . -type d -name "__pycache__" -exec rm -r {} + 