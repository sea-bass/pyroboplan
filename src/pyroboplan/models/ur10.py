"""Utilities to load example Universal Robots UR10 robot models."""

import os
import pinocchio

from .utils import get_example_models_folder


def load_ur10_on_base_models():
    """
    Gets the example UR10 on base models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "ur_description")
    urdf_filename = os.path.join(package_dir, "urdf", "ur10_on_base.urdf")

    return pinocchio.buildModelsFromUrdf(urdf_filename, package_dirs=models_folder)


def add_ur10_on_base_self_collisions(model, collision_model, srdf_filename=None):
    """
    Adds link self-collisions to the UR10 collision model.

    This uses an SRDF file to remove any excluded collision pairs.

    Parameters
    ----------
        model : `pinocchio.Model`
            The UR10 model.
        collision_model : `pinocchio.Model`
            The UR10 collision geometry model.
        srdf_filename : str, optional
            Path to the SRDF file describing the excluded collision pairs.
            If not specified, uses a default file included with the UR10 model.
    """
    if srdf_filename is None:
        models_folder = get_example_models_folder()
        package_dir = os.path.join(models_folder, "ur_description")
        srdf_filename = os.path.join(package_dir, "srdf", "ur10_on_base.srdf")

    collision_model.addAllCollisionPairs()
    pinocchio.removeCollisionPairs(model, collision_model, srdf_filename)
