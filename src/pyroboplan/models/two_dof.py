""" Utilities to load example 2-DOF manipulator. """

import os
import pinocchio

from .utils import get_example_models_folder


def load_models():
    """
    Gets the example 2-DOF models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "two_dof_description")
    urdf_filename = os.path.join(package_dir, "two_dof.urdf")

    return pinocchio.buildModelsFromUrdf(urdf_filename, package_dirs=models_folder)
