#!/usr/bin/env fish

# Setup script for Python virtual environment.

set -l VIRTUALENV_PATH "$VIRTUALENV_PATH"
if test -z "$VIRTUALENV_PATH"
    set VIRTUALENV_PATH ".virtualenvs/pyroboplan"
end

# Create a virtual environment, if one does not exist
if not test -d "$VIRTUALENV_PATH"
    echo "Creating virtual environment"
    python3 -m venv $VIRTUALENV_PATH
end

echo "Sourcing virtual environment"
source $VIRTUALENV_PATH/bin/activate.fish
