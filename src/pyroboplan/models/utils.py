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
    resource_path = importlib.resources.files(pyroboplan) / "models"
    return resource_path.as_posix()
