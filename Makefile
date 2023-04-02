.PHONY: all install lint format test clean FORCE

all: lint

install: FORCE
	pip install -e '.[test]'

lint: FORCE
	flake8
	black --check .
	isort --check .
	mypy --install-types --non-interactive .

format: FORCE
	black .
	isort .

test: lint FORCE
	pytest -v test
	ls notebooks/*.ipynb | grep -v migrate | xargs pytest -v --nbval-lax

clean: FORCE
	git clean -dfx

FORCE:
