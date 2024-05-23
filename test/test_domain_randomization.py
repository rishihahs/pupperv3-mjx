"""
pytest test/test_domain_randomization.py
"""

import pytest
from pupperv3_mjx import domain_randomization
from jax import numpy as jp
import jax
from pathlib import Path

from jax import random
from unittest.mock import MagicMock
import os
from brax.io import mjcf


def test_randomize_qpos():
    start_position_config = domain_randomization.StartPositionRandomization(
        x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=-0.5, z_max=0.5
    )

    rng = jax.random.PRNGKey(0)

    qpos = domain_randomization.randomize_qpos(
        jp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jp.float32), start_position_config, rng
    )

    expected_qpos = jp.array([-0.184, 0.165, 0.278, 0.709, 0.0, 0.0, 0.705, 7.0, 8.0, 9.0, 10.0])

    assert jp.isclose(
        qpos, expected_qpos, atol=1e-3
    ).all(), f"Expected: {expected_qpos}, but got: {qpos}"


def test_domain_randomize():
    # Create a mock sys object with necessary attributes
    ORIGINAL_MODEL_PATH = Path(
        os.path.expanduser(
            "~/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml"
        )
    )

    sys = mjcf.load(ORIGINAL_MODEL_PATH)

    # Generate a random key
    rng = random.PRNGKey(0)

    original_kp = sys.actuator_gainprm[:, 0]
    original_kd = -sys.actuator_biasprm[:, 2]

    # Call the domain_randomize function
    sys, in_axes = domain_randomization.domain_randomize(
        sys,
        jp.array([rng]),
        friction_range=(0.6, 1.4),
        kp_multiplier_range=(0.75, 1.25),
        kd_multiplier_range=(0.5, 2.0),
    )

    # Check if the output sys has the attributes updated correctly
    assert sys.geom_friction.shape == (1, 24, 3)
    assert sys.actuator_gainprm.shape == (1, 12, 10)
    assert sys.actuator_biasprm.shape == (1, 12, 10)

    # Further assertions can be added to check the exact values within the expected ranges
    # For example:
    assert (sys.geom_friction[0, :, 0] >= 0.6).all() and (sys.geom_friction[0, :, 0] <= 1.4).all()

    assert (sys.actuator_gainprm[0, :, 0] >= 0.75 * original_kp).all() and (
        sys.actuator_gainprm[0, :, 0] <= 1.25 * original_kp
    ).all()
    assert (-sys.actuator_biasprm[0, :, 2] >= 0.5 * original_kd).all() and (
        -sys.actuator_biasprm[0, :, 2] <= 2.0 * original_kd
    ).all()


if __name__ == "__main__":
    pytest.main()


if __name__ == "__main__":
    pytest.main()
