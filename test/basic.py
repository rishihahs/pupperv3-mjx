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

# Pupper model configuration
PATH = Path(
    "/Users/nathankau/pupper_v3_description/description/mujoco_xml/model_with_obstacles.xml"
)

LOWER_LEG_BODY_NAMES = [
    "leg_front_r_3",
    "leg_front_l_3",
    "leg_back_r_3",
    "leg_back_l_3",
]
FOOT_SITE_NAMES = [
    "leg_front_r_3_foot_site",
    "leg_front_l_3_foot_site",
    "leg_back_r_3_foot_site",
    "leg_back_l_3_foot_site",
]
FOOT_RADIUS = 0.02
TORSO_NAME = "base_link"

# Domain randomization
POSITION_CONTROL_KP_DELTA_RANGE = (-0.5, 0.5)

# Collision detection
MAX_CONTACT_POINTS = 8
MAX_GEOM_PAIRS = 8

# Obstacles
N_OBSTACLES = 10
OBSTACLE_X_RANGE = 1.0
OBSTACLE_HEIGHT = 0.02

sys = mjcf.load(ORIGINAL_MODEL_PATH.as_posix())
JOINT_UPPER_LIMITS = sys.jnt_range[1:, 1]
JOINT_LOWER_LIMITS = sys.jnt_range[1:, 0]

# Environment timestep
ENVIRONMENT_DT = 0.02

# Command sampling
LIN_VEL_X_RANGE = [-0.75, 0.75]  # min max [m/s]
LIN_VEL_Y_RANGE = [-0.5, 0.5]  # min max [m/s]
ANG_VEL_YAW_RANGE = [-2.0, 2.0]  # min max [rad/s]

# Termination
TERMINAL_BODY_Z = 0.10

# Joint PD overrides
DOF_DAMPING = 0.25
POSITION_CONTROL_KP = 5.0

# Default joint angles
DEFAULT_POSE = jp.array([0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52])

# PPO params
NUM_TIMESTEPS = 100_000_000  # originally 100M
EPISODE_LENGTH = 500

OBSERVATION_HISTORY = 1  # number of stacked observations to give the policy
ACTION_SCALE = 0.3  # originally 0.3
HIDDEN_LAYER_SIZES = (128, 128, 128, 128)

env_kwargs = dict(
    path=PATH.as_posix(),
    action_scale=ACTION_SCALE,
    observation_history=2,
    joint_lower_limits=JOINT_LOWER_LIMITS,
    joint_upper_limits=JOINT_UPPER_LIMITS,
    dof_damping=DOF_DAMPING,
    position_control_kp=POSITION_CONTROL_KP,
    foot_site_names=FOOT_SITE_NAMES,
    torso_name=TORSO_NAME,
    lower_leg_body_names=LOWER_LEG_BODY_NAMES,
    resample_velocity_step=500,
    linear_velocity_x_range=LIN_VEL_X_RANGE,
    linear_velocity_y_range=LIN_VEL_Y_RANGE,
    angular_velocity_range=ANG_VEL_YAW_RANGE,
    default_pose=DEFAULT_POSE,
    reward_config=config.get_config(),
)

N_OBSTACLES = 50
OBSTACLE_X_RANGE = (-5, 5)
OBSTACLE_Y_RANGE = (-5, 5)
OBSTACLE_HEIGHT = 0.02

# Add obstacles
xml_str = ORIGINAL_MODEL_PATH.read_text()
tree = ET.ElementTree(ET.fromstring(xml_str))
tree = obstacles.add_boxes_to_model(
    tree,
    n_boxes=50,
    x_range=OBSTACLE_X_RANGE,
    y_range=OBSTACLE_Y_RANGE,
    height=OBSTACLE_HEIGHT,
)
with open(PATH, "w+") as file:
    tree.write(file, encoding="unicode")

# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = environment.PupperV3Env(**env_kwargs)
eval_env = environment.PupperV3Env(**env_kwargs)

# initialize the state
rng = jax.random.PRNGKey(0)

print("Jitting step and reset")
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

state = jit_reset(rng)
state.info["command"] = jp.array([0, 0, 0])

rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 200
render_every = 2

for i in range(n_steps):
    print("Step: ", i)
    act_rng, rng = jax.random.split(rng)
    ctrl = jp.array(np.random.uniform(size=sys.nu))
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)

import mediapy as media

print("Writing video")
media.write_video(
    "video.mp4",
    eval_env.render(rollout[::render_every], camera="tracking_cam"),
    fps=1.0 / eval_env.dt / render_every,
)

# x_data = []
# y_data = []
# ydataerr = []
# times = [datetime.now()]
# max_y, min_y = 40, 0

# make_networks_factory = functools.partial(
#     ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 128, 128)
# )
# train_fn = functools.partial(
#     ppo.train,
#     num_timesteps=100_000_000,
#     num_evals=1,  # 10,
#     reward_scaling=1,
#     episode_length=10,  # 1000,
#     normalize_observations=True,
#     action_repeat=1,
#     unroll_length=20,
#     num_minibatches=2,  # 32
#     num_updates_per_batch=1,  # 4,
#     discounting=0.97,
#     learning_rate=3.0e-4,
#     entropy_cost=1e-2,
#     num_envs=1,
#     batch_size=8,  # 256,
#     network_factory=make_networks_factory,
#     randomization_fn=domain_randomization.domain_randomize,
#     seed=0,
# )

# make_inference_fn, params, _ = train_fn(
#     environment=env,
#     progress_fn=lambda num_steps, metrics: progress(
#         num_steps=num_steps,
#         metrics=metrics,
#         times=times,
#         x_data=x_data,
#         y_data=y_data,
#         ydataerr=ydataerr,
#         num_timesteps=train_fn.keywords["num_timesteps"],
#         max_y=max_y,
#         min_y=min_y,
#     ),
#     eval_env=eval_env,
# )

# train_fn = functools.partial(
#     ppo.train,
#     num_timesteps=100_000_000,
#     num_evals=10,
#     reward_scaling=1,
#     episode_length=1000,
#     normalize_observations=True,
#     action_repeat=1,
#     unroll_length=20,
#     num_minibatches=32,
#     num_updates_per_batch=4,
#     discounting=0.97,
#     learning_rate=3.0e-4,
#     entropy_cost=1e-2,
#     num_envs=8192,
#     batch_size=256,
#     network_factory=make_networks_factory,
#     randomization_fn=domain_randomize,
#     seed=0,
# )
