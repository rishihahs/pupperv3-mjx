from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import difflib
import numpy as np
import re
import xml.etree.ElementTree as ET
from typing import List, Callable
import mediapy as media
import os
import wandb
import jax


def progress(
    num_steps: int,
    metrics: dict,
    times: list,
    x_data: list,
    y_data: list,
    ydataerr: list,
    num_timesteps: int,
    min_y: float,
    max_y: float,
):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, num_timesteps * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")

    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.show()


def plot_multi_series(data, dt=1.0, display_axes=None, title=None):
    # Create a Plotly figure to plot the three series
    fig = go.Figure()
    time_index = np.arange(len(data)) * dt

    if display_axes is None:
        display_axes = list(range(data.shape[1]))

    for i in range(data.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=time_index,
                y=data[:, i],
                mode="lines",
                name=f"Series {i}",
                visible=True if i in display_axes else "legendonly",
            )
        )

    # Customize the layout with titles and axis labels
    fig.update_layout(
        title=title or "Time Series Visualization with Plotly",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Value"),
        legend=dict(title="Series"),
    )
    fig.show()


def fuzzy_search(obj, search_str: str, cutoff: float = 0.6):
    """
    Search through an object's properties and find those that match the search string in a fuzzy way.

    Args:
    obj: The object to search through.
    search_str: The string to match properties against.
    cutoff: The cutoff for matching ratio (0.0 to 1.0), higher means more strict matching.

    Returns:
    A list of tuples containing (property_name, match_ratio) that match the search string.
    """
    results = []

    # Get all properties of the object
    properties = dir(obj)

    # Search for fuzzy matches
    for prop in properties:
        ratio = difflib.SequenceMatcher(None, search_str, prop).ratio()
        if ratio >= cutoff:
            results.append((prop, ratio))

    # Sort results by match ratio in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def set_mjx_custom_options(tree: ET.ElementTree, max_contact_points: int, max_geom_pairs: int):
    root = tree.getroot()
    custom = root.find("custom")
    if custom is not None:
        for numeric in custom.findall("numeric"):
            name = numeric.get("name")
            if name == "max_contact_points":
                numeric.set("data", str(max_contact_points))
            elif name == "max_geom_pairs":
                numeric.set("data", str(max_geom_pairs))

        return tree
    return None


def set_robot_starting_position(
    tree: ET.ElementTree, starting_pos: List, starting_quat: List = None
):
    """Change the starting position of the robot in the xml mujoco model file"""

    body = tree.find(".//worldbody/body[@name='base_link']")
    body.set("pos", f"{starting_pos[0]} {starting_pos[1]} {starting_pos[2]}")
    if starting_quat is not None:
        body.set(
            "quat", f"{starting_quat[0]} {starting_quat[1]} {starting_quat[2]} {starting_quat[3]}"
        )

    home_position = tree.find(".//keyframe/key[@name='home']")
    qpos_scalar = list(map(float, re.split(r"\s+", home_position.get("qpos").strip())))
    qpos_scalar[:3] = starting_pos
    if starting_quat is not None:
        qpos_scalar[3:7] = starting_quat
    updated_qpos = " ".join(map(str, qpos_scalar))
    home_position.set("qpos", updated_qpos)
    return tree


def visualize_policy(
    current_step,
    make_policy_fn,
    params,
    eval_env,
    jit_step: Callable,
    jit_reset: Callable,
    output_folder: str,
    vx: float = 0.5,
    vy: float = 0.4,
    wz: float = 1.5,
):
    inference_fn = make_policy_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    # Make robot go forward, back, left, right
    command_seq = jp.array(
        [
            [vx, 0.0, 0.0],
            [-vx, 0.0, 0.0],
            [0.0, vy, 0.0],
            [0.0, -vy, 0.0],
            [0.0, 0.0, wz],
            [0.0, 0.0, -wz],
        ]
    )

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = command_seq[0]
    rollout = [state.pipeline_state]

    # grab a trajectory
    n_steps = 480
    render_every = 2
    ctrls = []

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)

        # Change command every 80 steps
        state.info["command"] = command_seq[int(i / 80)]

        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        ctrls.append(ctrl)

    filename = os.path.join(output_folder, f"step_{current_step}_policy.mp4")
    fps = int(1.0 / eval_env.dt / render_every)
    media.write_video(
        filename,
        eval_env.render(rollout[::render_every], camera="tracking_cam"),
        fps=fps,
    )
    wandb.log(
        {
            "eval/video/command/vx": the_command[0],
            "eval/video/command/vy": the_command[1],
            "eval/video/command/wz": the_command[2],
            "eval/video": wandb.Video(filename, format="mp4", fps=fps),
        },
        step=current_step,
    )
