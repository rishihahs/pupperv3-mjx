import xml.etree.ElementTree as ET
import random
from typing import Tuple
import math


def random_z_rotation_quaternion():
    """Generates a random quaternion with a random yaw angle."""
    # Generate a random yaw angle
    yaw = random.uniform(-math.pi, math.pi)

    # Construct the quaternion from the yaw angle
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def add_boxes_to_model(
    tree,
    n_boxes: int,
    x_range: Tuple,
    y_range: Tuple,
    height=0.02,
    depth=0.02,
    length=3.0,
    group="0",
):
    root = tree.getroot()

    # Find the worldbody element
    worldbody = root.find("worldbody")

    # Add N boxes to the worldbody
    for i in range(n_boxes):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        quat = random_z_rotation_quaternion()
        quat_str = " ".join(map(str, quat))
        box_body = ET.SubElement(
            worldbody, "body", name=f"box_{i}", pos=f"{x} {y} 0", quat=quat_str
        )
        ET.SubElement(
            box_body,
            "geom",
            name=f"box_geom_{i}",
            type="box",
            size=f"{depth/2.0} {length/2.0} {height}",
            rgba="0.1 0.5 0.8 1",
            conaffinity="1",
            contype="1",
            condim="3",
            group=group,
        )

    # Convert the modified XML tree back to a string
    return tree
