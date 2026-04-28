PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

.PHONY: venv install run check

venv:
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	@test -x "$(STREAMLIT)" || (echo "Virtualenv not found. Run 'make install' first." && exit 1)
	$(STREAMLIT) run app.py

check:
	$(PYTHON) -m compileall app.py cheatsheet_ai
