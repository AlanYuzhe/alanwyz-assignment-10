
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

.PHONY: all
all: setup run

.PHONY: setup
setup:
	@echo "Creating virtual environment..."
	python3.12 -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: run
run:
	@echo "Starting Flask application..."
	$(PYTHON) app.py

