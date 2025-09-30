# Python project for building the machine learning / deep learning models
- [Makefile](#makefile)
- [Commands](#commands)

## Makefile

The `Makefile` is used for executing common commands.

If you are on Windows, you will need to install `make`. There are three options to do this:

### Chocolatey:

    1. Install Chocolatey, a package manager for Windows 
    2. Open PowerShell and run: choco install make
Chocolatey install Instructions: https://chocolatey.org/install#individual

### Scoop:

    1. Install Scoop, another package manager for Windows
    2. Open PowerShell and run: scoop install make
Scoop install Instructions: https://scoop.sh/ (see Quickstart section)

### MinGW:

    Install MinGW, a collection of development tools for Windows, which includes mingw32-make
    You might need to rename mingw32-make.exe to make.exe in order to execute it with "make" command instead of "mingw32-make"

## Commands

### Install Dev Dependencies
  ```
  make install-dev
  ```

### Run project
  ```
  make run
  ```

### Run Linter
  ```
  make lint
  ```
  
### Run Tests
  ```
  make test
  ```
  
### Clean
  ```
  make clean
  ```

### Create virtual environment
  ```
  make venv
  ```
