.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .
	pip install -r requirements.txt

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]
	pip install -r requirements.txt

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 ta2
	isort -c --recursive ta2

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find ta2 -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive ta2
	isort --apply --atomic --recursive ta2

.PHONY: bumpversion-release
bumpversion-release: ## Merge master to stable and bumpversion release
	git checkout stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release
	git push --tags origin stable

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to master and bumpversion patch
	git checkout master
	git merge stable
	bumpversion --no-tag patch
	git push

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
CHANGELOG_LINES := $(shell git diff HEAD..stable HISTORY.md | wc -l)

.PHONY: check-release
check-release: ## Check if the release can be made
ifneq ($(CURRENT_BRANCH),master)
	$(error Please make the release from master branch\n)
endif
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: release
release: check-release bumpversion-release bumpversion-patch

.PHONY: release-minor
release-minor: check-release bumpversion-minor release

.PHONY: release-major
release-major: check-release bumpversion-major release

.PHONY: clean
clean: clean-build clean-pyc ## remove all build and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

login:
	docker login registry.datadrivendiscovery.org

build:
	docker build -t mit-d3m-ta2 .

submit: login build ## push to TA2 submission registry keeping a timestamped version in EC2
	docker tag mit-d3m-ta2:latest registry.datadrivendiscovery.org/ta2-submissions/ta2-mit/winter-2019
	docker push registry.datadrivendiscovery.org/ta2-submissions/ta2-mit/winter-2019
