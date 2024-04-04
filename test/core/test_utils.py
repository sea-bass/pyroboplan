import numpy as np
import pytest

from pyroboplan.core.utils import configuration_distance


def test_configuration_distance():
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    q_end = np.array([0.3, 0.0, -0.4, 0.0, 0.0])
    assert configuration_distance(q_start, q_end) == pytest.approx(0.5)
