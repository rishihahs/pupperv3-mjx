import jax
from dataclasses import dataclass
from jax import numpy as jp


def domain_randomize(sys, rng, friction_range=(0.6, 1.4), gain_range=(-0.5, 0.5)):
    """Randomizes the mjx.Model.

    TODO: What is the dimension of rng? The number of environments?
    """

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        param = (
            jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1])
            + sys.actuator_gainprm[:, 0]
        )
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


@dataclass
class StartPositionRandomization:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def random_z_rotation_quaternion(rng):
    """Generates a random quaternion with a random yaw angle."""
    yaw = jax.random.uniform(rng, (1,), minval=-jp.pi, maxval=jp.pi)
    cos_yaw = jp.cos(yaw / 2)
    sin_yaw = jp.sin(yaw / 2)
    return jp.concatenate((cos_yaw, jp.zeros(2), sin_yaw))


def randomize_qpos(qpos: jp.array, start_position_config: StartPositionRandomization, rng):
    """Return qpos with randomized position of first body. Do not use rng again!"""

    key_x, key_y, key_z, key_yaw = jax.random.split(rng, 4)
    qpos = qpos.at[:3].set(
        jax.random.uniform(
            key_z,
            shape=(3,),
            minval=jp.array(
                (
                    start_position_config.x_min,
                    start_position_config.y_min,
                    start_position_config.z_min,
                )
            ),
            maxval=jp.array(
                (
                    start_position_config.x_max,
                    start_position_config.y_max,
                    start_position_config.z_max,
                )
            ),
        )
    )
    random_yaw_quat = random_z_rotation_quaternion(key_yaw)
    qpos = qpos.at[3:7].set(random_yaw_quat)
    return qpos
