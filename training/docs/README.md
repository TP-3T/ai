# Python project for building the machine learning / deep learning models

- [Makefile](#makefile)
- [Commands](#commands)
- [Installation Steps](#installation-steps)

## Makefile

The `Makefile` is used for executing common commands.

If you are on Windows, you will need to install a ported version of the `make` command. There are two options to do this:

### Chocolatey:

    1. Install Chocolatey, a package manager for Windows 
    2. Open PowerShell and run: choco install make
Chocolatey install Instructions: https://chocolatey.org/install#individual

### Scoop:

    1. Install Scoop, another package manager for Windows
    2. Open PowerShell and run: scoop install make
Scoop install Instructions: https://scoop.sh/ (see Quickstart section)

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

## Installation Steps:

  ### 1. Install pyenv (Python version manager):
  
  __If using Windows__ - install pyenv-win:
  
  - Run Windows PowerShell as administrator and and the command:
  ```
  Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
  ```

  (Source https://pypi.org/project/pyenv-win/)

  __If using macOS/Linux__ - install pyenv:
  
  - Linux or Unix: https://github.com/pyenv/pyenv?tab=readme-ov-file#linuxunix
  
  - macOS: https://github.com/pyenv/pyenv?tab=readme-ov-file#macos

  ### 3. Install Python 3.13.7 using pyenv:

  ```
  pyenv install 3.13.7
  ```

  ### 2. (Skip this step if using Linux or Unix) - If on Windows, [Install make](#makefile) for the `make` command (if a version of make not already installed)

  ### 3. In terminal, create a virtual Python environment (venv)
  ```
  make venv
  ```

  ### 4. Activate the venv (you will have to do this every time you reopen the project in your IDE)
      
  If on macOS/Linux:

  ```
  source .venv/bin/activate
  ```

  If on Windows:

  ```
  ./.venv/Scripts/activate
  ```
      
  ### 5. Install dev dependencies
  ```
  make install-dev
  ```
