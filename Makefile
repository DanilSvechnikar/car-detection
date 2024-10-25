#* Variables
SHELL := /usr/bin/env bash
PYTHON ?= python3

ifeq ($(OS),Windows_NT)
	detected_OS := Windows
	MKDIR_CMD := mkdir
else
	detected_OS := $(shell uname -s)
	MKDIR_CMD := mkdir -p
endif


#* Installation
.PHONY: project-init
project-init: poetry-install

.PHONY: poetry-install
poetry-install:
	poetry install --no-interaction --without dev

.PHONY: pip-install
pip-install: poetry-export
	pip3 install --no-cache-dir --upgrade pip && \
	pip3 install --no-cache-dir -r requirements.txt

.PHONY: poetry-export
poetry-export:
	poetry lock -n && poetry export --without-hashes > requirements.txt

.PHONY: poetry-export-dev
poetry-export-dev:
	poetry lock -n && poetry export --with dev --without-hashes > requirements.dev.txt

.PHONY: tools-install
tools-install:
	poetry run pre-commit install --hook-type prepare-commit-msg --hook-type pre-commit
	poetry run nbdime config-git --enable
	#poetry run mypy --install-types --non-interactive ./


#* Cleaning
.PHONY: clean-all
clean-all: pycache-remove build-remove poetry-cache-clear pip-cache-clear

.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E "(.ipynb_checkpoints$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: poetry-cache-clear
poetry-cache-clear:
	poetry cache clear --all . -n

.PHONY: pip-cache-clear
pip-cache-clear:
	pip cache purge


#* Docker
include .env

.PHONY: docker-make-image
docker-make-image:
	docker build -t car-detection .

.PHONY: docker-cont-gpu
docker-cont-gpu:
	docker run --rm -it --gpus=all -p ${HOST}:${PORT}:${PORT} --name car-cont-gpu car-detection

.PHONY: docker-cont-cpu
docker-cont-cpu:
	docker run --rm -it -p ${HOST}:${PORT}:${PORT} --name car-cont-cpu car-detection
