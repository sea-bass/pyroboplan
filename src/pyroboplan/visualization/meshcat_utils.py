import meshcat.geometry as mg
import numpy as np


def visualize_frame(visualizer, name, tform):

    FRAME_AXIS_POSITIONS = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
        .astype(np.float32)
        .T
    )
    FRAME_AXIS_COLORS = (
        np.array(
            [[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]
        )
        .astype(np.float32)
        .T
    )
    visualizer.viewer[name].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=0.2 * FRAME_AXIS_POSITIONS,
                color=FRAME_AXIS_COLORS,
            ),
            mg.LineBasicMaterial(
                linewidth=4,
                vertexColors=True,
            ),
        )
    )
    visualizer.viewer[name].set_transform(tform.homogeneous)
