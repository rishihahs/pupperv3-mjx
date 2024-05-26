import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from pathlib import Path
import mujoco

from pupperv3_mjx import rewards, config
from etils import epath

from pupperv3_mjx import domain_randomization


class PupperV3Env(PipelineEnv):
    """Environment for training the Pupper V3 quadruped joystick policy in MJX."""

    def __init__(
        self,
        path: str,
        action_scale: float,
        observation_history: int,
        joint_lower_limits: List,
        joint_upper_limits: List,
        dof_damping: float,
        position_control_kp: float,
        foot_site_names: List[str],
        torso_name: str,
        lower_leg_body_names: List[str],
        resample_velocity_step: int,
        linear_velocity_x_range: Tuple,
        linear_velocity_y_range: Tuple,
        angular_velocity_range: Tuple,
        start_position_config: domain_randomization.StartPositionRandomization,
        default_pose: jp.array,
        reward_config,
        obs_noise: float = 0.05,
        kick_vel: float = 0.05,  # [m/s]
        push_interval: int = 10,
        terminal_body_z: float = 0.10,  # [m]
        early_termination_step_threshold: int = 500,
        terminal_body_angle: float = 0.52,  # [rad]
        foot_radius: float = 0.02,
        environment_timestep: float = 0.02,
        physics_timestep: float = 0.004,
        **kwargs,
    ):
        sys = mjcf.load(path)
        self._dt = environment_timestep  # this environment is 50 fps
        sys = sys.tree_replace({"opt.timestep": physics_timestep})

        # override menagerie params for smoother policy
        sys = sys.replace(
            # dof_damping=sys.dof_damping.at[6:].set(DOF_DAMPING),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(position_control_kp),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1]
            .set(-position_control_kp)
            .at[:, 2]
            .set(-dof_damping),
        )

        # override the default joint angles with DEFAULT_POSE
        # sys.mj_model.keyframe('home').qpos = sys.mj_model.keyframe('home').qpos.at[7:].set(DEFAULT_POSE)
        sys.mj_model.keyframe("home").qpos[7:] = default_pose

        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._reward_config = reward_config
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith("_scale"):
                self._reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, torso_name
        )
        assert self._torso_idx != -1, "torso not found"
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = default_pose
        self.lowers = joint_lower_limits
        self.uppers = joint_upper_limits
        feet_site = foot_site_names
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body_names
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = foot_radius
        self._nv = sys.nv

        # start pos randomization params
        self._start_position_config = start_position_config

        # training params
        self._linear_velocity_x_range = linear_velocity_x_range
        self._linear_velocity_y_range = linear_velocity_y_range
        self._angular_velocity_range = angular_velocity_range

        self._push_interval = push_interval
        self._resample_velocity_step = resample_velocity_step

        # observation configuration
        self.observation_dim = 33
        self._observation_history = observation_history

        # reward configuration
        self._early_termination_step_threshold = early_termination_step_threshold

        # terminal condition
        self._terminal_body_z = terminal_body_z
        self._terminal_body_angle = terminal_body_angle

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = self._linear_velocity_x_range  # min max [m/s]
        lin_vel_y = self._linear_velocity_y_range  # min max [m/s]
        ang_vel_yaw = self._angular_velocity_range  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, sample_command_key, randomize_pos_key = jax.random.split(rng, 3)

        init_q = domain_randomization.randomize_qpos(
            self._init_q, self._start_position_config, rng=randomize_pos_key
        )

        pipeline_state = self.pipeline_init(init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12, dtype=float),
            "last_vel": jp.zeros(12, dtype=float),
            "command": self.sample_command(sample_command_key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4, dtype=float),
            "rewards": {k: 0.0 for k in self._reward_config.rewards.scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
        }

        obs_history = jp.zeros(
            self._observation_history * self.observation_dim, dtype=float
        )  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2, dtype=float)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info["rng"], 3)

        # kick
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info["step"], self._push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < np.cos(
            self._terminal_body_angle
        )
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self._terminal_body_z

        # reward
        rewards_dict = {
            "tracking_lin_vel": (
                rewards.reward_tracking_lin_vel(
                    state.info["command"],
                    x,
                    xd,
                    tracking_sigma=self._reward_config.rewards.tracking_sigma,
                )
            ),
            "tracking_ang_vel": (
                rewards.reward_tracking_ang_vel(
                    state.info["command"],
                    x,
                    xd,
                    tracking_sigma=self._reward_config.rewards.tracking_sigma,
                )
            ),
            "lin_vel_z": rewards.reward_lin_vel_z(xd),
            "ang_vel_xy": rewards.reward_ang_vel_xy(xd),
            "orientation": rewards.reward_orientation(x),
            "torques": rewards.reward_torques(
                pipeline_state.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "joint_acceleration": rewards.reward_joint_acceleration(
                joint_vel, state.info["last_vel"], dt=self._dt
            ),
            "mechanical_work": rewards.reward_mechanical_work(
                pipeline_state.qfrc_actuator[6:], pipeline_state.qvel[6:]
            ),
            "action_rate": rewards.reward_action_rate(action, state.info["last_act"]),
            "stand_still": rewards.reward_stand_still(
                state.info["command"], joint_angles, self._default_pose
            ),
            "feet_air_time": rewards.reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": rewards.reward_foot_slip(
                pipeline_state,
                contact_filt_cm,
                feet_site_id=self._feet_site_id,
                lower_leg_body_id=self._lower_leg_body_id,
            ),
            "termination": rewards.reward_termination(
                done,
                state.info["step"],
                step_threshold=self._early_termination_step_threshold,
            ),
        }
        rewards_dict = {
            k: v * self._reward_config.rewards.scales[k] for k, v in rewards_dict.items()
        }
        reward = jp.clip(sum(rewards_dict.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards_dict
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > self._resample_velocity_step,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._resample_velocity_step),
            0,
            state.info["step"],
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_body_angular_velocity = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate(
            [
                local_body_angular_velocity,  # angular velocity
                math.rotate(jp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
                state_info["command"],  # command
                pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info["last_act"],  # last action
            ]
        )

        assert self.observation_dim == obs.shape[0]

        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )

        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    def render(
        self, trajectory: List[base.State], camera: str | None = None
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera)
