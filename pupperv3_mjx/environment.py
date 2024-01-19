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
from brax.envs.base import MjxEnv, State
from brax.io import mjcf

from pathlib import Path
import mujoco

from pupperv3_mjx import reward, config


class BarkourEnv(MjxEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        **kwargs,
    ):
        """
        Initialize the BarkourEnv environment.

        Args:
            obs_noise (float): The observation noise.
            action_scale (float): The action scale.
            kick_vel (float): The kick velocity.
            **kwargs: Additional keyword arguments.
        """
        path = Path(__file__).parent / "google_barkour_vb/scene_mjx.xml"
        self._dt = 0.02  # this environment is 50 fps
        self.brax_sys = mjcf.load(path).replace(dt=self._dt)
        model = self.brax_sys.get_model()
        model.opt.timestep = 0.004
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

        # override menagerie params for smoother policy
        model.dof_damping[6:] = 0.5239
        model.actuator_gainprm[:, 0] = 35.0
        model.actuator_biasprm[:, 1] = -35.0

        n_frames = kwargs.pop("n_frames", int(self._dt / model.opt.timestep))
        super().__init__(model=model, n_frames=n_frames)

        self.reward_config = config.get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith("_scale"):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY.value, "torso"
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(model.keyframe("home").qpos)
        self._default_pose = model.keyframe("home").qpos[7:]
        self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jp.array([0.52, 2.1, 2.1] * 4)
        feet_site = [
            "foot_front_left",
            "foot_hind_left",
            "foot_front_right",
            "foot_hind_right",
        ]
        feet_site_id = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body = [
            "lower_leg_front_left",
            "lower_leg_hind_left",
            "lower_leg_front_right",
            "lower_leg_hind_right",
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = model.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_vel": jp.zeros(12),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
        }

        obs_history = jp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info["rng"], 3)

        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info["step"], push_interval) == 0
        qvel = state.pipeline_state.data.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({"pipeline_state.data.qvel": qvel})

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
        foot_pos = pipeline_state.data.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = {
            "tracking_lin_vel": (
                reward.reward_tracking_lin_vel(
                    commands=state.info["command"],
                    x=x,
                    xd=xd,
                    tracking_sigma=self.reward_config.rewards.tracking_sigma,
                )
            ),
            "tracking_ang_vel": (
                reward.reward_tracking_ang_vel(
                    commands=state.info["command"],
                    x=x,
                    xd=xd,
                    tracking_sigma=self.reward_config.rewards.tracking_sigma,
                )
            ),
            "lin_vel_z": reward.reward_lin_vel_z(xd=xd),
            "ang_vel_xy": reward.reward_ang_vel_xy(xd=xd),
            "orientation": reward.reward_orientation(x=x),
            "torques": reward.reward_torques(
                torques=pipeline_state.data.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "action_rate": reward.reward_action_rate(
                act=action, last_act=state.info["last_act"]
            ),
            "stand_still": reward.reward_stand_still(
                commands=state.info["command"],
                joint_angles=joint_angles,
                default_pose=self._default_pose,
            ),
            "feet_air_time": reward.reward_feet_air_time(
                air_time=state.info["feet_air_time"],
                first_contact=first_contact,
                commands=state.info["command"],
            ),
            "foot_slip": reward.reward_foot_slip(
                pipeline_state=pipeline_state,
                contact_filt=contact_filt_cm,
                feet_site_id=self._feet_site_id,
                lower_leg_body_id=self._lower_leg_body_id,
            ),
            "termination": reward.reward_termination(
                done=done, step=state.info["step"]
            ),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward_sum = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward_sum, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate(
            [
                jp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                math.rotate(jp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
                state_info["command"] * jp.array([2.0, 2.0, 0.25]),  # command
                pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info["last_act"],  # last action
            ]
        )

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
        """
        Renders the trajectory of the environment.

        Args:
            trajectory (List[base.State]): The trajectory of the environment to render.
            camera (str | None): The camera view for rendering. If None, defaults to "track".

        Returns:
            Sequence[np.ndarray]: The rendered frames of the environment trajectory.
        """
        camera = camera or "track"
        return super().render(trajectory, camera)
