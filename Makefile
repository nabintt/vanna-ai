.RECIPEPREFIX := >

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup check-ollama train run dev test

setup:
>python3 -m venv $(VENV)
>$(PIP) install --upgrade pip
>$(PIP) install -r requirements.txt

check-ollama:
>$(PYTHON) scripts/check_ollama.py

train:
>$(PYTHON) scripts/bootstrap_training.py

run:
>$(PYTHON) scripts/run_server.py

dev:
>$(PYTHON) scripts/run_server.py --reload

test:
>$(PYTHON) -m pytest -q
