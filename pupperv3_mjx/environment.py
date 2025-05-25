from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import mujoco
from mujoco import mjx
import numpy as np
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp

from pupperv3_mjx import domain_randomization, rewards, utils

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def body_names_to_body_ids(mj_model, body_names: List[str]) -> np.array:
    body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, body_name) for body_name in body_names]
    assert not any(id_ == -1 for id_ in body_ids), "Body not found."
    return np.array(body_ids)


def body_name_to_geom_ids(mj_model, body_name: str) -> np.array:
    body = mj_model.body(body_name)
    return body.geomadr + np.arange(np.squeeze(body.geomnum))


def body_names_to_geom_ids(mj_model, body_names: List[str]) -> np.array:
    return np.concatenate(list(body_name_to_geom_ids(mj_model, name) for name in body_names))


# Given an mjx state `d`, calculates forces for all contacts.
@jax.jit
def get_contact_forces(d: mjx.Data):
    # mju_decodePyramid
    # 1: force: result
    # 2: pyramid: d.efc_force + contact.efc_address
    # 3: mu: contact.friction
    # 4: dim: contact.dim

    contact = d.contact
    cnt = d.ncon
    
    # Generate 2d array of efc_force indexed by efc_address containing the maximum
    # number of potential elements (10).
    # This enables us to operate on each contact force pyramid rowwise.
    efc_argmap = jp.linspace(
      contact.efc_address,
      contact.efc_address + 9,
      10, dtype=jp.int32
    ).T
    # OOB access clamps in jax, this is safe
    pyramid = d.efc_force[efc_argmap.reshape((efc_argmap.size))].reshape(efc_argmap.shape)

    # Calculate normal forces
    # force[0] = 0
    # for (int i=0; i < 2*(dim-1); i++) {
    #   force[0] += pyramid[i];
    # }
    index_matrix = jp.repeat(jp.arange(10)[None, :], cnt, axis=0)
    force_normal_mask = index_matrix < (2 * (contact.dim - 1)).reshape((cnt, 1))
    force_normal = jp.sum(jp.where(force_normal_mask, pyramid, 0), axis=1)

    # Calculate tangent forces
    # for (int i=0; i < dim-1; i++) {
    #   force[i+1] = (pyramid[2*i] - pyramid[2*i+1]) * mu[i];
    # }
    pyramid_indexes = jp.arange(5) * 2
    force_tan_all = (pyramid[:, pyramid_indexes] - pyramid[:, pyramid_indexes + 1]) * contact.friction
    force_tan = jp.where(pyramid_indexes < contact.dim.reshape((cnt, 1)), force_tan_all, 0)

    # Full force array
    forces = jp.concatenate((force_normal.reshape((cnt, 1)), force_tan), axis=1)
    
    # Special case frictionless contacts
    # if (dim == 1) {
    #   force[0] = pyramid[0];
    #   return;
    # }
    frictionless_mask = contact.dim == 1
    frictionless_forces = jp.concatenate((pyramid[:,0:1], jp.zeros((pyramid.shape[0], 5))), axis=1)
    return jp.where(
        frictionless_mask.reshape((cnt, 1)),
        frictionless_forces,
        forces
    )


class PupperV3Env(PipelineEnv):
    """Environment for training the Pupper V3 quadruped joystick policy in MJX."""

    def __init__(
        self,
        path: str,
        reward_config: Dict,
        action_scale: float,
        observation_history: int,
        joint_lower_limits: List = [
            -1.220,
            -0.420,
            -2.790,
            -2.510,
            -3.140,
            -0.710,
            -1.220,
            -0.420,
            -2.790,
            -2.510,
            -3.140,
            -0.710,
        ],
        joint_upper_limits: List = [
            2.510,
            3.140,
            0.710,
            1.220,
            0.420,
            2.790,
            2.510,
            3.140,
            0.710,
            1.220,
            0.420,
            2.790,
        ],
        dof_damping: float = 0.25,
        position_control_kp: float = 5.0,
        start_position_config: domain_randomization.StartPositionRandomization = (
            domain_randomization.StartPositionRandomization(
                x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.15, z_max=0.20
            )
        ),
        foot_site_names: List[str] = [
            "leg_front_r_3_foot_site",
            "leg_front_l_3_foot_site",
            "leg_back_r_3_foot_site",
            "leg_back_l_3_foot_site",
        ],
        foot_geom_names: List[str] = [
            "leg_front_r_3_foot_geom",
            "leg_front_l_3_foot_geom",
            "leg_back_r_3_foot_geom",
            "leg_back_l_3_foot_geom",
        ],
        torso_name: str = "base_link",
        upper_leg_body_names: List[str] = [
            "leg_front_r_2",
            "leg_front_l_2",
            "leg_back_r_2",
            "leg_back_l_2",
        ],
        lower_leg_body_names: List[str] = [
            "leg_front_r_3",
            "leg_front_l_3",
            "leg_back_r_3",
            "leg_back_l_3",
        ],
        resample_velocity_step: int = 500,
        linear_velocity_x_range: Tuple[float, float] = (-0.75, 0.75),
        linear_velocity_y_range: Tuple[float, float] = (-0.5, 0.5),
        angular_velocity_range: Tuple[float, float] = (-2.0, 2.0),
        zero_command_probability: float = 0.01,
        stand_still_command_threshold: float = 0.1,
        maximum_pitch_command: float = 0.0,  # degrees
        maximum_roll_command: float = 0.0,  # degrees
        default_pose: jax.Array = jp.array([0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52]),
        desired_abduction_angles: jax.Array = jp.array([0.0, 0.0, 0.0, 0.0]),
        angular_velocity_noise: float = 0.3,
        gravity_noise: float = 0.1,
        motor_angle_noise: float = 0.1,
        last_action_noise: float = 0.01,
        kick_vel: float = 0.2,
        kick_probability: float = 0.02,
        # terminal_body_z: float = 0.1,
        terminal_body_z: float = 0.05,
        early_termination_step_threshold: int = 500,
        terminal_body_angle: float = 0.52,
        foot_radius: float = 0.02,
        environment_timestep: float = 0.02,
        physics_timestep: float = 0.004,
        latency_distribution: jax.Array = jp.array([0.2, 0.8]),
        imu_latency_distribution: jax.Array = jp.array([0.5, 0.5]),  # TODO: Measure on pupper
        desired_world_z_in_body_frame: jax.Array = jp.array([0.0, 0.0, 1.0]),
        use_imu: bool = True,
    ):
        """
        Args:
            path (str): The path to the MJCF file.
            reward_config (Dict): The reward configuration.
            action_scale (float): The scale to apply to actions.
            observation_history (int): The number of previous observations to include in the state.
            joint_lower_limits (List): The lower limits for the joint angles.
            joint_upper_limits (List): The upper limits for the joint angles.
            dof_damping (float): The damping to apply to the DOFs.
            position_control_kp (float): The position control kp.
            start_position_config (domain_randomization.StartPositionRandomization):
            The start position randomization config.
            foot_site_names (List[str]): The names of the foot sites.
            torso_name (str): The name of the torso.
            upper_leg_body_names (List[str]): The names of the upper leg bodies.
            lower_leg_body_names (List[str]): The names of the lower leg bodies.
            resample_velocity_step (int): The number of steps to resample the velocity.
            linear_velocity_x_range (Tuple): The range of linear velocity in the x-direction.
            linear_velocity_y_range (Tuple): The range of linear velocity in the y-direction.
            angular_velocity_range (Tuple): The range of angular velocity.
            zero_command_probability (float): The probability of a near-zero command. Ensures enough
                training data with near-zero velocity command to ensure robot learns to stand still
            stand_still_command_threshold (float): The threshold for the stand still command.
            maximum_pitch_command (float): Maximum abs value of pitch command in degrees
            maximum_roll_command (float):  Maximum abs value of roll command in degrees
            default_pose (jp.array): The default pose.
            angular_velocity_noise (float): The angular velocity noise.
            gravity_noise (float): The gravity noise.
            motor_angle_noise (float): The motor angle noise.
            last_action_noise (float): The last action noise.
            kick_vel (float): The kick velocity.
            kick_probability (float): The kick probability.
            terminal_body_z (float): The terminal body z.
            early_termination_step_threshold (int): The early termination step threshold.
            terminal_body_angle (float): The terminal body angle.
            foot_radius (float): The foot radius.
            environment_timestep (float): The environment timestep.
            physics_timestep (float): The physics timestep.
            latency_distribution (jax.Array): Probability distribution for action latency.
            First element corresponds to 0 latency. Shape: (N, 1)
            desired_world_z_in_body_frame (jax.Array): The desired world z in body frame.
            use_imu (bool): Whether to use IMU.
        """
        sys = mjcf.load(path)
        self._dt = environment_timestep  # this environment is 50 fps
        sys = sys.tree_replace({"opt.timestep": physics_timestep})

        # override menagerie params for smoother policy
        sys = sys.replace(
            # dof_damping=sys.dof_damping.at[6:].set(DOF_DAMPING),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(position_control_kp),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-position_control_kp).at[:, 2].set(-dof_damping),
        )

        # override the default joint angles with default_pose
        sys.mj_model.keyframe("home").qpos[7:] = default_pose

        n_frames = self._dt // sys.opt.timestep
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self._reward_config = reward_config
        self._torso_geom_ids = body_name_to_geom_ids(sys.mj_model, torso_name)
        self._torso_idx = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, torso_name)
        assert self._torso_idx != -1, "torso not found"
        self._action_scale = jp.array(action_scale)
        self._angular_velocity_noise = angular_velocity_noise
        self._gravity_noise = gravity_noise
        self._motor_angle_noise = motor_angle_noise
        self._last_action_noise = last_action_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = default_pose
        self._desired_abduction_angles = desired_abduction_angles
        self.lowers = joint_lower_limits
        self.uppers = joint_upper_limits
        feet_site = foot_site_names
        feet_site_id = [mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)
        floor_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_geom_ids = np.array([mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_geom_names])
        assert not any(id_ == -1 for id_ in foot_geom_ids), "Site not found."
        self._floor_geom_id_pairs = np.concatenate((np.repeat(floor_id, foot_geom_ids.shape[0]).reshape(-1, 1), foot_geom_ids.reshape(-1, 1)), axis=1)
        assert(sys.mj_model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL) # Assert cone is PYRAMIDAL

        self._lower_leg_body_id = body_names_to_body_ids(sys.mj_model, lower_leg_body_names)
        self._upper_leg_geom_ids = body_names_to_geom_ids(sys.mj_model, upper_leg_body_names)

        self._foot_radius = foot_radius
        self._nv = sys.nv

        # start pos randomization params
        self._start_position_config = start_position_config

        # training params
        self._linear_velocity_x_range = linear_velocity_x_range
        self._linear_velocity_y_range = linear_velocity_y_range
        self._angular_velocity_range = angular_velocity_range
        self._zero_command_probability = zero_command_probability
        self._stand_still_command_threshold = stand_still_command_threshold

        # command for body orientation
        self._maximum_pitch_command = maximum_pitch_command
        self._maximum_roll_command = maximum_roll_command

        self._kick_probability = kick_probability
        self._resample_velocity_step = resample_velocity_step

        # observation configuration
        # self.observation_dim = 36  # 33 without orientation, 36 with orientation
        # self.observation_dim = 35  # 33 without orientation, 36 with orientation
        self.observation_dim = 36  # 33 without orientation, 36 with orientation
        self._observation_history = observation_history

        # reward configuration
        self._early_termination_step_threshold = early_termination_step_threshold

        # terminal condition
        self._terminal_body_z = terminal_body_z
        self._terminal_body_angle = terminal_body_angle

        # desired orientation
        self._desired_world_z_in_body_frame = jp.array(desired_world_z_in_body_frame)

        # latency
        self._latency_distribution = latency_distribution
        self._imu_latency_distribution = imu_latency_distribution

        # whether to use imu
        self._use_imu = use_imu

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """
        Sample random command with desired linear and angular velocity ranges.
        With a probability of self._zero_command_probability, return a near-zero
        command to ensure enough training data with near-zero velocity command.
        """
        lin_vel_x = self._linear_velocity_x_range  # min max [m/s]
        lin_vel_y = self._linear_velocity_y_range  # min max [m/s]
        ang_vel_yaw = self._angular_velocity_range  # min max [rad/s]

        rng, key1, key2, key3, key4, key5 = jax.random.split(rng, 6)
        lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])

        # X% probability to return command near [0, 0, 0]
        zero_cmd_prob = jax.random.uniform(key4, (1,))
        noisy_near_zero_command = jax.random.uniform(
            key5,
            (3,),
            minval=-self._stand_still_command_threshold,
            maxval=self._stand_still_command_threshold,
        )
        new_cmd = jp.where(zero_cmd_prob < self._zero_command_probability, noisy_near_zero_command, new_cmd)

        return new_cmd

    def sample_body_orientation(self, rng: jax.Array) -> jax.Array:
        """
        Sample random orientation with desired_world_z_in_body_frame as the mean.

        This method samples a random body orientation by generating random pitch and roll angles
        within the specified maximum limits. The desired_world_z_in_body_frame represents the
        desired orientation of the world z-axis in the body frame, which is used as the mean
        orientation. The method then rotates the z unit vector by the sampled pitch and roll
        angles to obtain the desired orientation.

        Args:
            rng (jax.Array): A random number generator array.

        Returns:
            jax.Array: The desired world z-axis orientation in the body frame.
        """

        rng, key_pitch, key_roll = jax.random.split(rng, 3)
        pitch = jax.random.uniform(key_pitch, (1,), minval=-1, maxval=1.0) * self._maximum_pitch_command
        roll = jax.random.uniform(key_roll, (1,), minval=-1, maxval=1.0) * self._maximum_roll_command
        # rotate the z unit vector by pitch and roll
        # euler_to_quat uses x-y'-z'' intrinsic convention so use roll, pitch, yaw
        euler_rotation = math.euler_to_quat(jp.array([roll[0], pitch[0], 0.0]))
        desired_world_z_in_body_frame = math.rotate(self._desired_world_z_in_body_frame, euler_rotation)
        return desired_world_z_in_body_frame

    def initial_action_buffer(self) -> jax.Array:
        return jp.zeros((12, self._latency_distribution.shape[0]), dtype=float)

    def initial_imu_buffer(self) -> jax.Array:
        """
        Initialize the IMU buffer which is shape (6, buffer_size).
        The order of elements in each column is:
            [angular_velocity_x, angular_velocity_y, angular_velocity_z,
            gravity_x, gravity_y, gravity_z].
        """
        buf = jp.zeros((6, self._imu_latency_distribution.shape[0]), dtype=float)
        buf = buf.at[5, :].set(-1.0)  # gravity is -1.0 in z
        return buf

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, sample_command_key, sample_orientation_key, randomize_pos_key, sample_start_time_key = jax.random.split(rng, 5)

        init_q = domain_randomization.randomize_qpos(self._init_q, self._start_position_config, rng=randomize_pos_key)

        pipeline_state = self.pipeline_init(init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12, dtype=float),
            "action_buffer": self.initial_action_buffer(),
            "imu_buffer": self.initial_imu_buffer(),
            "last_vel": jp.zeros(12, dtype=float),
            "command": self.sample_command(sample_command_key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4, dtype=float),
            # "rewards": {k: 0.0 for k in self._reward_config.rewards.scales.keys()},
            # "rewards": {k: 0.0 for k in ["torques", "joint_acceleration", "mechanical_work", "action_rate", "foot_slip", "knee_collision", "body_collision", "height", "balance", "pitch", "style", "foot"]},
            # "rewards": {k: 0.0 for k in ["height", "balance", "pitch", "style", "foot", "termination", "knee_collision", "body_collision", "torques", "joint_acceleration", "action_rate", "mechanical_work"]},
            "rewards": {k: 0.0 for k in ["height", "termination", "life"]},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
            "desired_world_z_in_body_frame": self.sample_body_orientation(sample_orientation_key),
            "start_time": jax.random.uniform(sample_start_time_key, minval=0.0, maxval=5.0),
            "is_half_turn": 0,
            "is_one_turn": 0,
            "stage": jp.array([1.0, 0, 0, 0, 0], dtype=float),
            "foot_contact_forces": jp.zeros(4, dtype=float),
        }

        obs_history = jp.zeros(
            self._observation_history * self.observation_dim, dtype=float
        )  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2, dtype=float)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types

        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        state.info["rng"], cmd_rng, kick_noise_2, kick_bernoulli, latency_key, cmd_binarylol_key = jax.random.split(state.info["rng"], 6)

        # Whether to kick and the kick velocity are both random
        kick = jax.random.uniform(kick_noise_2, shape=(2,), minval=-1.0, maxval=1.0) * self._kick_vel
        kick *= jax.random.bernoulli(kick_bernoulli, p=self._kick_probability, shape=(1,))
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # Sample an action with random latency
        lagged_action, state.info["action_buffer"] = utils.sample_lagged_value(
            latency_key, state.info["action_buffer"], action, self._latency_distribution
        )

        # Physics step
        motor_targets = self._default_pose + lagged_action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # Observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # Foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # Foot contact based on contact forces
        matching_contacts = state.pipeline_state.contact.geom[None, :, :] == self._floor_geom_id_pairs[:, None, :]
        matching_contacts = jp.all(matching_contacts, axis=2)
        contact_ids = jp.where(matching_contacts.any(axis=1), matching_contacts.argmax(axis=1), -1)
        forces_all = get_contact_forces(state.pipeline_state)
        safe_indices = jp.where(contact_ids == -1, 0, contact_ids)
        forces = forces_all[safe_indices, :]
        mask = (contact_ids != -1)[:, None]
        forces = jp.where(mask, forces, 0)
        rots = jp.transpose(state.pipeline_state.contact.frame[contact_ids], (0, 2, 1))
        world_forces = (rots @ forces[:, :3, None]).squeeze(-1)
        foot_contact_forces = world_forces[:, 2]
        # foot_contact_forces = pipeline_state.sensordata[self._feet_contact_sensor_adr]
        state.info["foot_contact_forces"] = foot_contact_forces

        # Done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        # done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < np.cos(self._terminal_body_angle)
        # done |= jp.any(joint_angles < self.lowers)
        done = jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self._terminal_body_z

        com_height = x.pos[self._torso_idx - 1, 2]
        # com_height = pipeline_state.subtree_com[0][2]
        # height_reward =  state.info["stage"][0]*(-jp.abs(com_height - 0.14))
        # height_reward += state.info["stage"][1]*(-jp.abs(com_height - 0.11))
        # height_reward += state.info["stage"][2]*(jp.where(com_height < 0.15, -20 * (0.15 - com_height) / 0.15, jp.exp(jp.log(501) / 0.2 * (com_height - 0.15)) - 1))
        # height_reward += state.info["stage"][3]*(jp.where(com_height < 0.15, -20 * (0.15 - com_height) / 0.15, jp.exp(jp.log(501) / 0.2 * (com_height - 0.15)) - 1))
        # # height_reward += state.info["stage"][2]*((20*jp.exp(10*com_height) - 20) * (com_height >= 0.25).astype('float32'))
        # # height_reward += state.info["stage"][3]*((20*jp.exp(10*com_height) - 20) * (com_height >= 0.25).astype('float32'))
        # height_reward += state.info["stage"][4]*(-jp.abs(com_height - 0.14))
        height_reward = 60*jp.exp(10*com_height) - 60
        # height_reward =  state.info["stage"][0]*(jp.where(com_height < 0.15, -2 * (0.15 - com_height) / 0.15, jp.exp(jp.log(501) / 0.2 * (com_height - 0.15)) - 1))
        # height_reward += state.info["stage"][1]*(10)

        # body balance
        world_z = jp.array([0.0, 0.0, 1.0])
        world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
        balance_reward = state.info["stage"][0]*(-jp.arccos(jp.clip(world_z_in_body_frame[2], -1.0, 1.0)))
        balance_reward += state.info["stage"][1]*(-jp.arccos(jp.clip(world_z_in_body_frame[2], -1.0, 1.0)))
        balance_reward += state.info["stage"][2]*(-jp.abs(jp.arccos(jp.clip(world_z_in_body_frame[1], -1.0, 1.0)) - jp.pi/2.0))
        balance_reward += state.info["stage"][3]*(-jp.abs(jp.arccos(jp.clip(world_z_in_body_frame[1], -1.0, 1.0)) - jp.pi/2.0))
        balance_reward += state.info["stage"][4]*(-jp.arccos(jp.clip(world_z_in_body_frame[2], -1.0, 1.0)))

        # pitch vel
        base_lin_vels = math.rotate(xd.vel[0, :], math.quat_inv(x.rot[0]))
        base_ang_vels = math.rotate(xd.ang[0, :], math.quat_inv(x.rot[0]))
        vel_penalty = jp.square(base_lin_vels[0]) + jp.square(base_lin_vels[1]) + jp.square(base_ang_vels[2])
        base_ang_vel_y = base_ang_vels[1]
        pitch_reward = state.info["stage"][0]*(-vel_penalty)
        pitch_reward += state.info["stage"][1]*(-vel_penalty)
        pitch_reward += state.info["stage"][2]*(1.0 - state.info["is_one_turn"])*(-base_ang_vel_y)
        pitch_reward += state.info["stage"][3]*(1.0 - state.info["is_one_turn"])*(-base_ang_vel_y)
        pitch_reward += state.info["stage"][4]*(-vel_penalty)

        # style
        style_reward = -jp.mean(jp.square(joint_angles - self._default_pose))

        # foot contact
        contact_via_force = foot_contact_forces > 0.25
        contact_float = contact_via_force.astype('float32')
        # contact_float = contact.astype('float32')
        foot_reward = state.info["stage"][0]*0.25*jp.sum(contact_float)
        foot_reward += state.info["stage"][1]*0.25*jp.sum(contact_float)
        foot_reward += state.info["stage"][2]*((contact_float[2] + contact_float[3])/2.0)
        foot_reward += state.info["stage"][3]*(1.0 - jp.sum(contact_float)/4.0)
        # reward += state.info["stage"][3]*0.25*jp.sum(contact_float)
        foot_reward += state.info["stage"][4]*0.25*jp.sum(contact_float)

        # switch = jax.random.bernoulli(cmd_binarylol_key).astype('float32')
        # state.info["command"] = jp.array([switch, switch, switch])
        ## from0_to1 = jp.logical_and(
        ##     state.info["stage"][0] == 1.0, jp.logical_and(
        ##         jp.any(contact_via_force),
        ##         state.info["is_half_turn"] == 1
        ##     )
        ## ).astype('float32')
        ## state.info["stage"] = state.info["stage"].at[0].set((1.0 - from0_to1)*state.info["stage"][0])
        ## state.info["stage"] = state.info["stage"].at[1].set(from0_to1 + (1.0 - from0_to1)*state.info["stage"][1])
        # from3_to4 = jp.logical_and(
        #     state.info["stage"][3] == 1.0, jp.logical_and(
        #         jp.any(contact_via_force),
        #         state.info["is_half_turn"] == 1
        #     )
        # ).astype('float32')
        # state.info["stage"] = state.info["stage"].at[3].set((1.0 - from3_to4)*state.info["stage"][3])
        # state.info["stage"] = state.info["stage"].at[4].set(from3_to4 + (1.0 - from3_to4)*state.info["stage"][4])
        # from2_to3 = jp.logical_and(
        #     state.info["stage"][2] == 1.0, 
        #     jp.all(~contact_via_force)
        # ).astype('float32')
        # state.info["stage"] = state.info["stage"].at[2].set((1.0 - from2_to3)*state.info["stage"][2])
        # state.info["stage"] = state.info["stage"].at[3].set(from2_to3 + (1.0 - from2_to3)*state.info["stage"][3])
        # from1_to2 = jp.logical_and(
        #     state.info["stage"][1] == 1.0, jp.logical_and(
        #         com_height <= 0.11, 
        #         jp.all(contact_via_force)
        #     )
        # ).astype('float32')
        # state.info["stage"] = state.info["stage"].at[1].set((1.0 - from1_to2)*state.info["stage"][1])
        # state.info["stage"] = state.info["stage"].at[2].set(from1_to2 + (1.0 - from1_to2)*state.info["stage"][2])
        # from0_to1 = jp.logical_and(
        #     state.info["stage"][0] == 1.0, jp.logical_and(
        #         state.info["step"]*self.dt > state.info["start_time"], jp.logical_and(
        #             com_height >= 0.135, 
        #             state.info["is_half_turn"] == 0
        #         )
        #     )
        # ).astype('float32')
        # state.info["stage"] = state.info["stage"].at[0].set((1.0 - from0_to1)*state.info["stage"][0])
        # state.info["stage"] = state.info["stage"].at[1].set(from0_to1 + (1.0 - from0_to1)*state.info["stage"][1])

        # check the robot tumbling
        state.info["is_half_turn"] = jp.logical_or(
            state.info["is_half_turn"], jp.logical_and(
                world_z_in_body_frame[0] < 0, world_z_in_body_frame[2] < 0)).astype(int)
        state.info["is_one_turn"] = jp.logical_or(
            state.info["is_one_turn"], jp.logical_and(
                state.info["is_half_turn"], jp.logical_and(
                    world_z_in_body_frame[0] >= 0, world_z_in_body_frame[2] >= 0))).astype(int)

        # landing_wo_turns = jp.logical_and(state.info["stage"][3] == 1.0, jp.logical_and(jp.any(contact_via_force), state.info["is_half_turn"] == 0))
        # done |= landing_wo_turns

        # jax.debug.print("com_height {com_height}, time {time}, from0to1 {from0to1}, start+time {start_time}, stage {stage}", com_height=com_height, time=(state.info["step"]*self.dt), from0to1=from0_to1, start_time=state.info["start_time"], stage=state.info["stage"])
        
        # Reward
        rewards_dict = {
            "height": height_reward,
            # "balance": balance_reward,
            # "pitch": pitch_reward,
            # "style": style_reward,
            # "foot": foot_reward,
            # "tracking_lin_vel": rewards.reward_tracking_lin_vel(
            #     state.info["command"],
            #     x,
            #     xd,
            #     tracking_sigma=self._reward_config.rewards.tracking_sigma,
            # ),
            # "tracking_ang_vel": rewards.reward_tracking_ang_vel(
            #     state.info["command"],
            #     x,
            #     xd,
            #     tracking_sigma=self._reward_config.rewards.tracking_sigma,
            # ),
            # "tracking_orientation": rewards.reward_tracking_orientation(
            #     state.info["desired_world_z_in_body_frame"],
            #     x,
            #     tracking_sigma=self._reward_config.rewards.tracking_sigma,
            # ),
            # "lin_vel_z": rewards.reward_lin_vel_z(xd),
            # "ang_vel_xy": rewards.reward_ang_vel_xy(xd),
            # "orientation": rewards.reward_orientation(x),
            # "torques": rewards.reward_torques(pipeline_state.qfrc_actuator),  # pytype: disable=attribute-error
            # "joint_acceleration": rewards.reward_joint_acceleration(joint_vel, state.info["last_vel"], dt=self._dt),
            # "mechanical_work": rewards.reward_mechanical_work(
            #     pipeline_state.qfrc_actuator[6:], pipeline_state.qvel[6:]
            # ),
            # "action_rate": rewards.reward_action_rate(action, state.info["last_act"]),
            # "stand_still": rewards.reward_stand_still(state.info["command"], joint_angles, self._default_pose, 0.1),
            # "stand_still_joint_velocity": rewards.reward_stand_still(
            #     state.info["command"], joint_vel, jp.zeros(12), self._stand_still_command_threshold
            # ),
            # "abduction_angle": rewards.reward_abduction_angle(
            #     joint_angles,
            #     desired_abduction_angles=self._desired_abduction_angles,
            # ),
            # "feet_air_time": rewards.reward_feet_air_time(
            #     state.info["feet_air_time"],
            #     first_contact,
            #     state.info["command"],
            # ),
            # "foot_slip": rewards.reward_foot_slip(
            #     pipeline_state,
            #     contact_filt_cm,
            #     feet_site_id=self._feet_site_id,
            #     lower_leg_body_id=self._lower_leg_body_id,
            # ),
            "termination": rewards.reward_termination(
                done,
                state.info["step"],
                step_threshold=self._early_termination_step_threshold,
            ),
            # "knee_collision": rewards.reward_geom_collision(pipeline_state, self._upper_leg_geom_ids),
            # "body_collision": rewards.reward_geom_collision(pipeline_state, self._torso_geom_ids),
            "life": -1,
        }
        rewards_dict = {k: v * self._reward_config.rewards.scales[k] for k, v in rewards_dict.items()}
        # reward = jp.clip(sum(rewards_dict.values()) * self.dt, 0.0, 10000.0)
        reward = jp.clip(sum(rewards_dict.values()) * self.dt, -100.0, 10000.0)

        # State management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards_dict
        state.info["step"] += 1

        # Sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > self._resample_velocity_step,
            self.sample_command(cmd_rng),
            state.info["command"],
        )

        # Resample new desired body orientation
        state.info["desired_world_z_in_body_frame"] = jp.where(
            state.info["step"] > self._resample_velocity_step,
            self.sample_body_orientation(cmd_rng),
            state.info["desired_world_z_in_body_frame"],
        )

        # Reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self._resample_velocity_step),
            0,
            state.info["step"],
        )
        # Log total displacement as a proxy metric
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
        if self._use_imu:
            inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
            local_body_angular_velocity = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        else:
            inv_torso_rot = jp.array([1, 0, 0, 0])
            local_body_angular_velocity = jp.zeros(3)

        # See https://arxiv.org/abs/2202.05481 as reference for noise addition
        (
            state_info["rng"],
            ang_key,
            gravity_key,
            motor_angle_key,
            last_action_key,
            imu_sample_key,
        ) = jax.random.split(state_info["rng"], 6)

        ang_vel_noise = jax.random.uniform(ang_key, (3,), minval=-1, maxval=1) * self._angular_velocity_noise
        gravity_noise = jax.random.uniform(gravity_key, (3,), minval=-1, maxval=1) * self._gravity_noise
        motor_ang_noise = jax.random.uniform(motor_angle_key, (12,), minval=-1, maxval=1) * self._motor_angle_noise
        last_action_noise = jax.random.uniform(last_action_key, (12,), minval=-1, maxval=1) * self._last_action_noise

        noised_gravity = math.rotate(jp.array([0, 0, -1]), inv_torso_rot) + gravity_noise
        noised_gravity = noised_gravity / jp.linalg.norm(noised_gravity)
        noised_ang_vel = local_body_angular_velocity + ang_vel_noise
        noised_imu_data = jp.concatenate([noised_ang_vel, noised_gravity])

        lagged_imu_data, state_info["imu_buffer"] = utils.sample_lagged_value(
            imu_sample_key,
            state_info["imu_buffer"],
            noised_imu_data,
            self._imu_latency_distribution,
        )

        # Construct observation and add noise
        obs = jp.concatenate([
            lagged_imu_data,  # noised angular velocity and gravity
            state_info["command"],  # command
            # state_info["stage"],
            # state_info["foot_contact_forces"],
            state_info["desired_world_z_in_body_frame"],  # desired body orientation
            pipeline_state.q[7:] - self._default_pose + motor_ang_noise,  # motor angles
            state_info["last_act"] + last_action_noise,  # last action
        ])

        assert self.observation_dim == obs.shape[0]

        # clip
        obs = jp.clip(obs, -100.0, 100.0)

        # stack observations through time
        # newest observation at the front
        new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return new_obs_history

    def render(self, trajectory: List[base.State], camera: Optional[str] = None) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera)
