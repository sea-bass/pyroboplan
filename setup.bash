# Setup script for examples

VIRTUALENV_PATH=".virtualenvs/intro_manip"

# Create a virtual environment, if one does not exist
if [ ! -d "${VIRTUALENV_PATH}" ]; then
    echo "Creating virtual environment"
    python3 -m venv ${VIRTUALENV_PATH}
    source "${VIRTUALENV_PATH}/bin/activate"
    pip3 install -r requirements.txt
else
    echo "Sourcing virtual environment"
    source "${VIRTUALENV_PATH}/bin/activate"
fi
