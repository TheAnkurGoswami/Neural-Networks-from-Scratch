help:
	@echo "The following make targets are available:"
	@echo "install install all dependencies"
	@echo "lint-all	run all lint steps"
	@echo "lint-comment	run linter check over regular comments"
	@echo "lint-emptyinit	main inits must be empty"
	@echo "lint-flake8	run flake8 checker to deteck missing trailing comma"
	@echo "lint-forgottenformat	ensures format strings are used"
	@echo "lint-indent	run indent format check"
	@echo "lint-pycodestyle	run linter check using pycodestyle standard"
	@echo "lint-pycodestyle-debug	run linter in debug mode"
	@echo "lint-pyi	Ensure no regular python files exist in stubs"
	@echo "lint-pylint	run linter check using pylint standard"
	@echo "lint-requirements	run requirements check"
	@echo "lint-stringformat	run string format check"
	@echo "lint-type-check	run type check"
	@echo "pre-commit 	sort python package imports using isort"

ENV ?= nn_from_scratch
PY_VERSION = 3.12

create-env:
	python3 -m venv $(ENV)
	. $(ENV)/bin/activate
	make install

create-conda-env:
	conda create -y -n $(ENV) python=$(PY_VERSION)

create-uv-env:
	uv venv $(ENV)
	. $(ENV)/bin/activate
	uv sync

delete-merged-branches:
	git branch --merged | grep -vE '^\*|main|master' | xargs -r git branch -d

install:
	python -m pip install --upgrade pip
	python -m pip install --progress-bar off --upgrade -r requirements.txt
	python -m pip install --progress-bar off --upgrade -r requirements.lint.txt

lint-requirements:
	locale
	cat requirements.lint.txt
	sort -ufc requirements.lint.txt
	cat requirements.txt
	sort -ufc requirements.txt

lint-type-check:
	mypy . --config-file mypy.ini --show-error-codes

lint-all:
	$(MAKE) lint-requirements
	ruff check
	$(MAKE) lint-type-check

fix-lints:
	black .
	ruff check --fix

pytest:
	./run_pytest.sh $(FILE)

