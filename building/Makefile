.PHONY: help install-dev lint clean test clean venv

# Default target
help:
	@echo Available targets:
	@echo   help        - Show this help message
	@echo   install-dev - Install development dependencies
# 	@echo "  lint        - Run Ruff on all Python files"
# 	@echo "  test        - Run pytest" - ai team todo after
	@echo   clean       - Remove Python cache files
	@echo   venv        - Create virtual environment

# Install Dev dependencies
install-dev:
	@echo Installing dev dependencies ...

# Run
run:
	@echo Running project ...

# Run Linter
lint:
	@echo Running linter ...

# Run Test
test:
	@echo Running tests ...

# Clean
clean:
	@echo Removing cached .pyc files ...
	rm -rf __pycache__

# Create virtual environment
venv:
	@echo Creating virtual environment ...
	python -m venv venv
	@echo Virtual environment created. Activate it with the command:
# 	@echo "source venv/bin/activate  # On macOS/Linux"
# 	@echo ".\venv\bin\activate   # On Windows"
