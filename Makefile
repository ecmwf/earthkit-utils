NBSPHINX_EXECUTE := never

setup:
	pre-commit install

default: qa tests

qa:
	pre-commit run --all-files

tests:
	python -m pytest -vv --cov=. --cov-report=html

docs-build:
	cd docs && make clean && make html SPHINXOPTS="-D nbsphinx_execute=$(NBSPHINX_EXECUTE)"
