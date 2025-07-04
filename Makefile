#!/usr/bin/make -f

PYTHON=python3
SPHINX_APIDOC=sphinx-apidoc
TARGET=s1etad

.PHONY: default help dist check fullcheck coverage clean cleaner distclean \
        lint docs api data

default: help

help:
	@echo "Usage: make <TARGET>"
	@echo "Available targets:"
	@echo "  help      - print this help message"
	@echo "  dist      - generate the distribution packages (source and wheel)"
	@echo "  check     - run a full test (using pytest)"
	@echo "  fullcheck - run a full test (using tox)"
	@echo "  coverage  - run tests and generate the coverage report"
	@echo "  clean     - clean build artifacts"
	@echo "  cleaner   - clean cache files and working directories of al tools"
	@echo "  distclean - clean all the generated files"
	@echo "  lint      - perform check with code linter (flake8, black)"
	@echo "  docs      - generate the sphinx documentation"
	@echo "  api       - update the API source files in the documentation"

dist:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*.tar.gz dist/*.whl

check:
	$(PYTHON) -m pytest --doctest-modules $(TARGET) tests

fullcheck:
	$(PYTHON) -m tox run

coverage:
	$(PYTHON) -m pytest --doctest-modules --cov=$(TARGET) --cov-report=html --cov-report=term $(TARGET) tests

clean:
	$(RM) -r *.*-info build
	find . -name __pycache__ -type d -exec $(RM) -r {} +
	# $(RM) -r __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	$(RM) $(TARGET)/*.c $(TARGET)/*.cpp $(TARGET)/*.so $(TARGET)/*.o
	if [ -f docs/Makefile ] ; then $(MAKE) -C docs clean; fi
	$(RM) -r docs/_build
	$(RM) -r docs/_static

cleaner: clean
	$(RM) -r .coverage htmlcov
	$(RM) -r .pytest_cache
	$(RM) -r .tox
	$(RM) -r .mypy_cache
	$(RM) -r .ruff_cache
	$(RM) -r .ipynb_checkpoints
	$(RM) -r docs/notebooks/data

distclean: cleaner
	$(RM) -r dist
	env PYTHONPATH=. \
	    $(PYTHON) -c "from tests.dataset import clean_cache; clean_cache()"
	find . -name __pycache__ -type d -exec $(RM) -r {} +

lint:
	$(PYTHON) -m flake8 --count --statistics $(TARGET) tests
	$(PYTHON) -m pydocstyle --count $(TARGET)
	$(PYTHON) -m isort --check $(TARGET) tests
	$(PYTHON) -m black --check $(TARGET) tests
	# $(PYTHON) -m fawltydeps
	# $(PYTHON) -m mypy --check-untyped-defs --ignore-missing-imports $(TARGET) tests
	ruff check $(TARGET) tests
	codespell $(TARGET) tests/*.py docs/*.rst

docs:  # data
	# mkdir -p docs/notebooks/data
	# ln -srf tests/data/*/*.SAFE docs/notebooks/data/
	mkdir -p docs/_static
	$(MAKE) -C docs html
	$(MAKE) -C docs linkcheck
	$(MAKE) -C docs spelling

api:
	$(RM) -r docs/api
	$(SPHINX_APIDOC) --module-first --separate --no-toc -o docs/api \
	  --doc-project "$(TARGET) API" --templatedir docs/_templates/apidoc \
	  $(TARGET) $(TARGET)/tests

data:
	env PYTHONPATH=. \
	    $(PYTHON) -c "from tests.dataset import download_all; download_all()"
