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

xml_str = ORIGINAL_MODEL_PATH.read_text()
tree = ET.ElementTree(ET.fromstring(xml_str))

utils.set_starting_position(tree, starting_pos=[0.1, 0.2, 0.5], starting_quat=[0.1, 0.2, 0.3, 0.4])

body = tree.find(".//worldbody/body[@name='base_link']")
assert body.get("pos").split(" ")[0] == "0.1"
assert body.get("pos").split(" ")[1] == "0.2"
assert body.get("pos").split(" ")[2] == "0.5"

assert body.get("quat").split(" ")[0] == "0.1"
assert body.get("quat").split(" ")[1] == "0.2"
assert body.get("quat").split(" ")[2] == "0.3"
assert body.get("quat").split(" ")[3] == "0.4"

home_position = tree.find(".//keyframe/key[@name='home']")
assert list(map(float, home_position.get("qpos").split(" ")))[:7] == [
    0.1,
    0.2,
    0.5,
    0.1,
    0.2,
    0.3,
    0.4,
]
