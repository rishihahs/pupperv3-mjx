import xml.etree.ElementTree as ET
import random


def add_boxes_to_model(tree, n_boxes, x_range, height=0.02, depth=0.02, length=3.0, group="0"):
    root = tree.getroot()

    # Find the worldbody element
    worldbody = root.find("worldbody")

    # Add N boxes to the worldbody
    for i in range(n_boxes):
        box_body = ET.SubElement(
            worldbody, "body", name=f"box_{i}", pos=f"{(random.random()-0.5)*2*x_range} 0 0"
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
