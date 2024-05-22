from pupperv3_mjx import environment, domain_randomization, utils, config, obstacles
from datetime import datetime
import functools
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from jax import numpy as jp
import numpy as np
import jax
from pathlib import Path
from pupperv3_mjx import obstacles
import xml.etree.ElementTree as ET

ORIGINAL_MODEL_PATH = Path(
    "/Users/nathankau/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml"
)

start_position_config = domain_randomization.StartPositionRandomization(
    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z_min=-0.5, z_max=0.5
)

rng = jax.random.PRNGKey(0)

qpos = domain_randomization.randomize_qpos(
    jp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jp.float32), start_position_config, rng
)

assert jp.isclose(
    qpos, jp.array([-0.184, 0.165, 0.278, 0.709, 0.0, 0.0, 0.705, 7.0, 8.0, 9.0, 10.0]), atol=1e-3
).all()

print("Randomized qpos: ", qpos)
