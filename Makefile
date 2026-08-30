.PHONY: compile test lint

compile:
	python -m compileall src api app scripts

test:
	python -m pytest

lint:
	python -m ruff check src api app scripts tests
