.RECIPEPREFIX := >

VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: setup check-glm train run dev test

setup:
>python3 -m venv $(VENV)
>$(PYTHON) -m pip install --upgrade pip
>$(PYTHON) -m pip install -r requirements.txt

check-glm:
>$(PYTHON) scripts/check_glm.py

train:
>$(PYTHON) scripts/bootstrap_training.py

run:
>$(PYTHON) scripts/run_server.py

dev:
>$(PYTHON) scripts/run_server.py --reload

test:
>$(PYTHON) -m pytest -q
