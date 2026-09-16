.PHONY: compile test lint stack

compile:
	python -m compileall src api app scripts

test:
	python -m pytest

lint:
	python -m ruff check src api app scripts tests

stack:
	python scripts/run_stack.py --bundle "$(BUNDLE)"
