import jax
import math
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math


# ------------ reward functions----------------
def reward_lin_vel_z(xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.square(xd.vel[0, 2])


def reward_ang_vel_xy(xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.sum(jp.square(xd.ang[0, :2]))


def reward_orientation(x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))


def reward_torques(torques: jax.Array) -> jax.Array:
    # Penalize torques
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))


def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize changes in actions
    return jp.sum(jp.square(act - last_act))


def reward_tracking_lin_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(-lin_vel_error / tracking_sigma)
    return lin_vel_reward


def reward_tracking_ang_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error / tracking_sigma)


def reward_feet_air_time(
    air_time: jax.Array, first_contact: jax.Array, commands: jax.Array, minimum_airtime: float = 0.1
) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - minimum_airtime) * first_contact)
    rew_air_time *= math.normalize(commands[:3])[1] > 0.05  # no reward for zero command
    return rew_air_time


def reward_stand_still(
    commands: jax.Array,
    joint_angles: jax.Array,
    default_pose: jax.Array,
) -> jax.Array:
    # Penalize motion at zero commands
    return jp.sum(jp.abs(joint_angles - default_pose)) * (math.normalize(commands[:3])[1] < 0.1)


def reward_foot_slip(
    pipeline_state: base.State,
    contact_filt: jax.Array,
    feet_site_id,
    lower_leg_body_id,
) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.data.site_xpos[feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.data.xpos[lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))


def reward_termination(done: jax.Array, step: jax.Array, step_threshold: int) -> jax.Array:
    return done & (step < step_threshold)
