.PHONY: dev clean

VENV = .venv
PYTHON = python3
ACTIVATE = $(VENV)/bin/activate
PIP = $(VENV)/bin/pip

dev: $(ACTIVATE)
	@echo "Use 'source $(ACTIVATE)' to activate."

$(ACTIVATE): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt
	@echo "# Virtual environment created and requirements installed."

clean:
	rm -rf $(VENV)
	@echo "Virtual environment removed."
