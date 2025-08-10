# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyroboplan"
copyright = "2024-2025, Sebastian Castro"
author = "Sebastian Castro"
version = release = "1.3.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../src/"))

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# External links
extlinks = {
    "examples": ("https://github.com/sea-bass/pyroboplan/tree/main/examples/%s", ""),
    "source": (
        "https://github.com/sea-bass/pyroboplan/tree/main/src/pyroboplan/%s",
        "",
    ),
}

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
