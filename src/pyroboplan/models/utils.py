""" Utilities for model loading. """

import importlib.resources
import pyroboplan


def get_example_models_folder():
    """
    Returns the full path the example models folder.

    Returns
    -------
        str
            The path to the `pyroboplan` example models folder.
    """
    return importlib.resources.path(pyroboplan, "models").as_posix()
