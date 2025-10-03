.PHONY: help install-dev lint clean test clean venv

# Default target
help:
	@echo Available make commands:
	@echo   make help        	- Show this help message
	@echo   make install-dev 	- Install development dependencies
	@echo   make lint        	- Run Ruff on all Python files
	@echo   make test 		- Run pytest
	@echo   make clean 		- Remove Python cache files
	@echo   make venv 		- Create virtual Python environment (requires pyenv to be installed - see README.md in docs)

# Install Dev dependencies
install-dev:
	@echo Installing dev dependencies ...
	pip install -e .

# Run (likely not needed since modules will be ran individually)
run:
	@echo Running project ...
	pyenv exec python __init__.py

# Run Linter
lint:
	@echo Running linter (TODO) ...

# Run Test
test:
	@echo Running tests (TODO) ...

# Clean
clean:
	@echo Removing cached .pyc files ...
	rm -rf __pycache__

# Create virtual environment (requires pyenv to be installed)
venv:
	@echo Creating virtual Python environment ...
	pyenv exec python -m venv .venv
	@echo Virtual environment created. Activate it with the appropriate command for your OS:
	@echo "./.venv/Scripts/activate"	If on Windows
	@echo "source .venv/bin/activate" If on macOS/Linux
