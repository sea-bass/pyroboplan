""" Utilities for visualization using MeshCat. """

import meshcat.geometry as mg
import numpy as np


FRAME_AXIS_POSITIONS = (
    np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
    .astype(np.float32)
    .T
)
FRAME_AXIS_COLORS = (
    np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]])
    .astype(np.float32)
    .T
)


def visualize_frame(
    visualizer, name, tform, line_length=0.2, line_width=3, line_color=None
):
    """
    Visualizes a coordinate frame as an axis triad at a specified pose.

    Parameters
    ----------
        visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`
            The visualizer instance.
        name : str
            The name of the MeshCat component to add.
        tform : `pinocchio.SE3`
            The transform at which to display the frame.
        line_length : float, optional
            The length of the axes in the triad.
        line_width : float, optional
            The width of the axes in the triad.
        line_color : array-like, optional
            The line colors to use. If None, chooses default axes colors.
    """
    if line_color:
        color = np.array([line_color] * 6).T
    else:
        color = FRAME_AXIS_COLORS

    visualizer.viewer[name].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=line_length * FRAME_AXIS_POSITIONS,
                color=color,
            ),
            mg.LineBasicMaterial(
                linewidth=line_width,
                vertexColors=True,
            ),
        )
    )
    visualizer.viewer[name].set_transform(tform.homogeneous)


def visualize_frames(
    visualizer, prefix_name, tforms, line_length=0.2, line_width=3, line_color=None
):
    """
    Visualizes a set of coordinate frames as axis triads at specified poses.

    Parameters
    ----------
        visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`
            The visualizer instance.
        prefix_name : str
            The name of the MeshCat component to add.
        tforms : list[`pinocchio.SE3`]
            A list of transforms at which to display frames.
        line_length : float, optional
            The length of the axes in the triad.
        line_width : float, optional
            The width of the axes in the triad.
        line_color : array-like, optional
            The line colors to use. If None, chooses default axes colors.
    """
    for idx, tform in enumerate(tforms):
        visualize_frame(
            visualizer,
            f"{prefix_name}/frame{idx}",
            tform,
            line_length=line_length,
            line_width=line_width,
            line_color=line_color,
        )


def visualize_path(visualizer, name, tforms, line_width=3, line_color=[0.0, 0.0, 0.0]):
    """
    Visualizes a path of poses as lines containing only the translation component.

    Parameters
    ----------
        visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`
            The visualizer instance.
        name : str
            The name of the MeshCat component to add.
        tforms : list[`pinocchio.SE3`]
            A list of transforms representing the vertices of the path.
        line_width : float, optional
            The width of the axes in the triad.
        line_color : array-like, optional
            The line color to use.
    """
    positions = []
    for idx in range(1, len(tforms)):
        positions.extend([tforms[idx - 1].translation, tforms[idx].translation])

    visualizer.viewer[name].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=np.array(positions).T,
                color=np.array([line_color for _ in positions]).T,
            ),
            mg.LineBasicMaterial(
                linewidth=line_width,
                vertexColors=True,
            ),
        )
    )
