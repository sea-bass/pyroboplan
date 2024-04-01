#!/bin/bash

# Generate Sphinx documentation
#
# Note that you may need some additional Python packages:
# pip3 install sphinx sphinx-rtd-theme

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd "${SCRIPT_DIR}" || exit

# Clean up
rm -rf build/
rm -rf source/modules.rst
rm -rf source/pyroboplan*

# Regenerate the API docs
sphinx-apidoc -f -o source ../src/pyroboplan

# Build the docs
make html

popd || exit
