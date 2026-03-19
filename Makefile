PROJECT := earthkit-utils
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

setup:
	pre-commit install

default: qa unit-tests type-check

qa:
	pre-commit run --all-files


unit-tests:
	python -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --ignore=tests/legacy-api

type-check:
	python -m mypy .

conda-env-update:
	$(CONDA) install -y -c conda-forge conda-merge
	$(CONDA) run conda-merge environment.yml ci/environment-ci.yml > ci/combined-environment-ci.yml
	$(CONDA) env update $(CONDAFLAGS) -f ci/combined-environment-ci.yml

docker-build:
	docker build -t $(PROJECT) .

docker-run:
	docker run --rm -ti -v $(PWD):/srv $(PROJECT)

docs-build:
	cd docs && rm -fr _api && make clean && make html
