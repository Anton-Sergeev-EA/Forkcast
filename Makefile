# Forkcast — типовые команды. Использование: make <цель>
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install run quick test backtest fetch lint clean

help:            ## список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

venv:            ## создать виртуальное окружение
	$(PY) -m venv $(VENV)

install: venv    ## установить зависимости и пакет
	$(BIN)/pip install -U pip setuptools wheel
	$(BIN)/pip install -r requirements-lock.txt
	$(BIN)/pip install --no-build-isolation --no-deps -e .

run:             ## полный расчёт и все отчёты
	$(BIN)/forkcast run

quick:           ## быстрый прогон (меньше симуляций)
	$(BIN)/forkcast run --quick

test:            ## автотесты
	$(BIN)/pytest -q

backtest:        ## ретроспективная проверка модели
	$(BIN)/forkcast backtest

fetch:           ## обновить ключевую ставку и курс с cbr.ru
	$(BIN)/forkcast fetch-macro

lint:            ## статический анализ
	$(BIN)/ruff check src tests

clean:           ## удалить кэши
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
