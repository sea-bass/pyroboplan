#!/bin/bash

# Runs all unit tests

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TEST_RESULTS_DIR="${SCRIPT_DIR}/results"

echo "Running unit tests..."
python3 -m pytest "$SCRIPT_DIR" \
 --cov="$SCRIPT_DIR" --cov-branch --cov-report term \
 --cov-report html:"$TEST_RESULTS_DIR/test_results_coverage_html" \
 --cov-report xml:"$TEST_RESULTS_DIR/test_results_coverage.xml" \
 --junitxml="$TEST_RESULTS_DIR/test_results.xml" \
 --html="$TEST_RESULTS_DIR/test_results.html" \
 --self-contained-html
