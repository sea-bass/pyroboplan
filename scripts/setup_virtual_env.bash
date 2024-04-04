#!/bin/bash

# Setup script for Python virtual environment.

VIRTUALENV_PATH=".virtualenvs/pyroboplan"

# Create a virtual environment, if one does not exist
if [ ! -d "${VIRTUALENV_PATH}" ]; then
    echo "Creating virtual environment"
    python3 -m venv ${VIRTUALENV_PATH}
fi

echo "Sourcing virtual environment"
source "${VIRTUALENV_PATH}/bin/activate"
