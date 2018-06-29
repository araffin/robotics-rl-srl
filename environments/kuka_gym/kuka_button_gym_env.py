import os
import pybullet as p
import time

import numpy as np
import torch as th
import pybullet_data
from gym import spaces
from gym.utils import seeding

from environments.srl_env import SRLGymEnv
from state_representation.episode_saver import EpisodeSaver
from srl_zoo.preprocessing import getNChannels

from environments.kuka_gym import kuka

#  Number of steps before termination
MAX_STEPS = 1000  # WARNING: should be also change in __init__.py (timestep_limit)
N_CONTACTS_BEFORE_TERMINATION = 5
# Terminate the episode if the arm is outside the safety sphere during too much time
N_STEPS_OUTSIDE_SAFETY_SPHERE = 5000
RENDER_HEIGHT = 224
RENDER_WIDTH = 224
Z_TABLE = -0.2
N_DISCRETE_ACTIONS = 6
BUTTON_LINK_IDX = 1
BUTTON_GLIDER_IDX = 1  # Button glider joint
DELTA_V = 0.03  # velocity per physics step.
DELTA_V_CONTINUOUS = 0.0035  # velocity per physics step (for continuous actions).
DELTA_THETA = 0.1  # angular velocity per physics step.
RELATIVE_POS = True  # Use relative position for ground truth
# NOISE_STD = DELTA_V / 3 # Add noise to actions, so the env is not fully deterministic
NOISE_STD = 0.01
NOISE_STD_CONTINUOUS = 0.0001
NOISE_STD_JOINTS = 0.002
N_RANDOM_ACTIONS_AT_INIT = 5  # Randomize init arm pos: take 5 random actions
BUTTON_DISTANCE_HEIGHT = 0.28  # Extra height added to the buttons position in the distance calculation

CONNECTED_TO_SIMULATOR = False  # To avoid calling disconnect in the __del__ method when not needed


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


# TODO: improve the physics of the button

"""
Gym wrapper for Kuka arm RL
"""


class KukaButtonGymEnv(SRLGymEnv):
    """
    Gym wrapper for Kuka environment with a push button
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) Whether to use discrete or continuous actions
    :param multi_view :(bool) if TRUE -> returns stacked images of the scene on 6 channels (two cameras)
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    :param action_repeat: (int) Number of timesteps an action is repeated (here it is equivalent to frameskip)
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param action_joints: (bool) Set actions to apply to the joint space
    :param use_srl: (bool) Set to true, use srl_models
    :param srl_model_path: (str) Path to the srl model
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param use_ground_truth: (bool) Set to true, the observation will be the ground truth (arm position)
    :param use_joints: (bool) Set input to include the joint angles (only if not using SRL model)
    :param random_target: (bool) Set the button position to a random position on the table
    :param force_down: (bool) Set Down as the only vertical action allowed
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    :param env_rank: (int) the number ID of the environment
    :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
    """

    def __init__(self, urdf_root=pybullet_data.getDataPath(), renders=False, is_discrete=True, multi_view=False,
                 name="kuka_button_gym", max_distance=0.4, action_repeat=1, shape_reward=False, action_joints=False,
                 use_srl=False, srl_model_path=None, record_data=False, use_ground_truth=False, use_joints=False,
                 random_target=False, force_down=True, state_dim=-1, learn_states=False, verbose=False,
                 save_path='srl_zoo/data/', env_rank=0, srl_pipe=None):
        super(KukaButtonGymEnv, self).__init__(use_ground_truth=use_ground_truth,
                                               relative_pos=RELATIVE_POS,
                                               env_rank=env_rank,
                                               srl_pipe=srl_pipe)
        self._timestep = 1. / 240.
        self._urdf_root = urdf_root
        self._action_repeat = action_repeat
        self._observation = []
        self._env_step_counter = 0
        self._renders = renders
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self._cam_dist = 1.1
        self._cam_yaw = 145
        self._cam_pitch = -36
        self._cam_roll = 0
        self._max_distance = max_distance
        self._shape_reward = shape_reward
        self._random_target = random_target
        self._force_down = force_down
        self.camera_target_pos = (0.316, -0.2, -0.1)
        self._is_discrete = is_discrete
        self.terminated = False
        self.renderer = p.ER_TINY_RENDERER
        self.debug = False
        self.n_contacts = 0
        self.state_dim = state_dim
        self.use_srl = use_srl or use_ground_truth or use_joints
        self.use_ground_truth = use_ground_truth
        self.use_joints = use_joints
        self.action_joints = action_joints
        self.relative_pos = RELATIVE_POS
        self.cuda = th.cuda.is_available()
        self.saver = None
        self.multi_view = multi_view
        self.verbose = verbose
        self.max_steps = MAX_STEPS
        self.n_steps_outside = 0
        self.table_uid = None
        self.np_random = None
        self.button_pos = None
        self.button_uid = None
        self._kuka = None
        self.action = None

        if record_data:
            self.saver = EpisodeSaver(name, max_distance, state_dim, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=learn_states, path=save_path)

        if self._renders:
            client_id = p.connect(p.SHARED_MEMORY)
            if client_id < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

            # self.renderer = p.ER_BULLET_HARDWARE_OPENGL
            self.debug = True
            # Debug sliders for moving the camera
            self.x_slider = p.addUserDebugParameter("x_slider", -10, 10, self.camera_target_pos[0])
            self.y_slider = p.addUserDebugParameter("y_slider", -10, 10, self.camera_target_pos[1])
            self.z_slider = p.addUserDebugParameter("z_slider", -10, 10, self.camera_target_pos[2])
            self.dist_slider = p.addUserDebugParameter("cam_dist", 0, 10, self._cam_dist)
            self.yaw_slider = p.addUserDebugParameter("cam_yaw", -180, 180, self._cam_yaw)
            self.pitch_slider = p.addUserDebugParameter("cam_pitch", -180, 180, self._cam_pitch)

        else:
            p.connect(p.DIRECT)

        global CONNECTED_TO_SIMULATOR
        CONNECTED_TO_SIMULATOR = True

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            if self.action_joints:
                # 7 angles for the arm rotation, from -1 to 1
                action_dim = 7
                self._action_bound = 1
            else:
                # 3 direction for the arm movement, from -1 to 1
                action_dim = 3
                self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)

        if self.use_srl:
            if self.use_ground_truth and self.use_joints:
                self.state_dim = self.getGroundTruthDim() + self.getJointsDim()
            elif self.use_joints:
                self.state_dim = self.getJointsDim()
            elif self.use_ground_truth:
                self.state_dim = self.getGroundTruthDim()
                
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        # Create numpy random generator
        # This seed can be changed later
        self.seed(0)

    def getSRLState(self, observation):
        """
        get the SRL state for this environement with a given observation
        :param observation: (numpy float) image
        :return: (numpy float)
        """
        state = []
        if self.use_ground_truth:
            if self.relative_pos:
                state += list(self.getGroundTruth() - self.getTargetPos())
            else:
                state += list(self.getGroundTruth())
        if self.use_joints:
            state += list(self._kuka.joint_positions)

        if len(state) != 0:
            return np.array(state)
        else:
            self.srl_pipe[0].put((self.env_rank, observation))
            return self.srl_pipe[1][self.env_rank].get()

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        return self.button_pos

    @staticmethod
    def getJointsDim():
        """
        :return: (int)
        """
        return 14

    @staticmethod
    def getGroundTruthDim():
        """
        :return: (int)
        """
        return 3

    def getGroundTruth(self):
        """
        Alias for getArmPos for compatibility between envs
        :return: (numpy array)
        """
        return np.array(self.getArmPos())

    def getArmPos(self):
        """
        :return: ([float]) Position (x, y, z) of kuka gripper
        """
        return p.getLinkState(self._kuka.kuka_uid, self._kuka.kuka_gripper_index)[0]

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.terminated = False
        self.n_contacts = 0
        self.n_steps_outside = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])

        self.table_uid = p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                    0.000000, 0.000000, 0.0, 1.0)

        # Initialize button position
        x_pos = 0.5
        y_pos = 0
        if self._random_target:
            x_pos += 0.15 * self.np_random.uniform(-1, 1)
            y_pos += 0.3 * self.np_random.uniform(-1, 1)

        self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos = np.array([x_pos, y_pos, Z_TABLE])

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdf_root_path=self._urdf_root, timestep=self._timestep,
                               use_inverse_kinematics=(not self.action_joints),
                               small_constraints=(not self._random_target))
        self._env_step_counter = 0
        # Close the gripper and wait for the arm to be in rest position
        for _ in range(500):
            if self.action_joints:
                self._kuka.applyAction(list(np.array(self._kuka.joint_positions)[:7]) + [0, 0])
            else:
                self._kuka.applyAction([0, 0, 0, 0, 0])
            p.stepSimulation()

        # Randomize init arm pos: take 5 random actions
        for _ in range(N_RANDOM_ACTIONS_AT_INIT):
            if self._is_discrete:
                action = [0, 0, 0, 0, 0]
                sign = 1 if self.np_random.rand() > 0.5 else -1
                action_idx = self.np_random.randint(3)  # dx, dy or dz
                action[action_idx] += sign * DELTA_V
            else:
                if self.action_joints:
                    joints = np.array(self._kuka.joint_positions)[:7]
                    joints += DELTA_THETA * self.np_random.normal(joints.shape)
                    action = list(joints) + [0, 0]
                else:
                    action = np.zeros(5)
                    rand_direction = self.np_random.normal((3,))
                    # L2 normalize, so that the random direction is not too high or too low
                    rand_direction /= np.linalg.norm(rand_direction, 2)
                    action[:3] += DELTA_V_CONTINUOUS * rand_direction

            self._kuka.applyAction(list(action))
            p.stepSimulation()

        self._observation = self.getExtendedObservation()

        self.button_pos = np.array(p.getLinkState(self.button_uid, BUTTON_LINK_IDX)[0])
        self.button_pos[2] += BUTTON_DISTANCE_HEIGHT  # Set the target position on the top of the button
        if self.saver is not None:
            self.saver.reset(self._observation, self.getTargetPos(), self.getGroundTruth())

        if self.use_srl:
            # if len(self.saver.srl_model_path) > 0:
            # self.srl_model.load(self.saver.srl_model_path))
            return self.getSRLState(self._observation)

        return np.array(self._observation)

    def __del__(self):
        if CONNECTED_TO_SIMULATOR:
            p.disconnect()

    def seed(self, seed=None):
        """
        Seed random generator
        :param seed: (int)
        :return: ([int])
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        if getNChannels() > 3:
            self.multi_view = True
        self._observation = self.render("rgb_array")
        return self._observation

    def step(self, action):
        """
        :param action: (int)
        """
        # if you choose to do nothing
        if action is None:
            if self.action_joints:
                return self.step2(list(np.array(self._kuka.joint_positions)[:7]) + [0, 0])
            else:
                return self.step2([0, 0, 0, 0, 0])

        self.action = action  # For saver
        if self._is_discrete:
            dv = DELTA_V  # velocity per physics step.
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            if self._force_down:
                dz = [0, 0, 0, 0, -dv, -dv][action]  # Remove up action
            else:
                dz = [0, 0, 0, 0, -dv, dv][action]
            # da = [0, 0, 0, 0, 0, -0.1, 0.1][action]  # end effector angle
            finger_angle = 0.0  # Close the gripper
            # real_action = [dx, dy, -0.002, da, finger_angle]
            real_action = [dx, dy, dz, 0, finger_angle]
        else:
            if self.action_joints:
                arm_joints = np.array(self._kuka.joint_positions)[:7]
                d_theta = DELTA_THETA
                # Add noise to action
                d_theta += self.np_random.normal(0.0, scale=NOISE_STD_JOINTS)
                # append [0,0] for finger angles
                real_action = list(action * d_theta + arm_joints) + [0, 0]  # TODO remove up action
            else:
                dv = DELTA_V_CONTINUOUS
                # Add noise to action
                dv += self.np_random.normal(0.0, scale=NOISE_STD_CONTINUOUS)
                dx = action[0] * dv
                dy = action[1] * dv
                if self._force_down:
                    dz = -abs(action[2] * dv)  # Remove up action
                else:
                    dz = action[2] * dv
                finger_angle = 0.0  # Close the gripper
                real_action = [dx, dy, dz, 0, finger_angle]

        if self.verbose:
            print(np.array2string(np.array(real_action), precision=2))

        return self.step2(real_action)

    def step2(self, action):
        """
        :param action:([float])
        """
        # Apply force to the button
        p.setJointMotorControl2(self.button_uid, BUTTON_GLIDER_IDX, controlMode=p.POSITION_CONTROL, targetPosition=0.1)

        for i in range(self._action_repeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._env_step_counter += 1

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timestep)

        reward = self._reward()
        done = self._termination()
        if self.saver is not None:
            self.saver.step(self._observation, self.action, reward, done, self.getGroundTruth())

        if self.use_srl:
            return self.getSRLState(self._observation), reward, done, {}

        return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):

        if mode != "rgb_array":
            return np.array([])
        camera_target_pos = self.camera_target_pos

        if self.debug:
            self._cam_dist = p.readUserDebugParameter(self.dist_slider)
            self._cam_yaw = p.readUserDebugParameter(self.yaw_slider)
            self._cam_pitch = p.readUserDebugParameter(self.pitch_slider)
            x = p.readUserDebugParameter(self.x_slider)
            y = p.readUserDebugParameter(self.y_slider)
            z = p.readUserDebugParameter(self.z_slider)
            camera_target_pos = (x, y, z)
            # self._cam_roll = p.readUserDebugParameter(self.roll_slider)

        # TODO: recompute view_matrix and proj_matrix only in debug mode
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=self._cam_roll,
            upAxisIndex=2)
        proj_matrix1 = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px1, _, _) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix1,
            projectionMatrix=proj_matrix1, renderer=self.renderer)
        rgb_array1 = np.array(px1)

        if self.multi_view:
            # adding a second camera on the other side of the robot
            view_matrix2 = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.316, 0.316, -0.105),
                distance=1.05,
                yaw=32,
                pitch=-13,
                roll=0,
                upAxisIndex=2)
            proj_matrix2 = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                nearVal=0.1, farVal=100.0)
            (_, _, px2, _, _) = p.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix2,
                projectionMatrix=proj_matrix2, renderer=self.renderer)
            rgb_array2 = np.array(px2)
            rgb_array_res = np.concatenate((rgb_array1[:, :, :3], rgb_array2[:, :, :3]), axis=2)
        else:
            rgb_array_res = rgb_array1[:, :, :3]
        return rgb_array_res

    def close(self):
        # TODO: implement close function to close GUI
        pass

    def _termination(self):
        if self.terminated or self._env_step_counter > self.max_steps:
            self._observation = self.getExtendedObservation()
            return True
        return False

    def _reward(self):
        gripper_pos = self.getArmPos()
        distance = np.linalg.norm(self.button_pos - gripper_pos, 2)
        # print(distance)

        contact_points = p.getContactPoints(self.button_uid, self._kuka.kuka_uid, BUTTON_LINK_IDX)
        reward = int(len(contact_points) > 0)
        self.n_contacts += reward

        contact_with_table = len(p.getContactPoints(self.table_uid, self._kuka.kuka_uid)) > 0

        if distance > self._max_distance or contact_with_table:
            reward = -1
            self.n_steps_outside += 1
        else:
            self.n_steps_outside = 0

        if contact_with_table or self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION \
                or self.n_steps_outside >= N_STEPS_OUTSIDE_SAFETY_SPHERE:
            self.terminated = True

        if self._shape_reward:
            if self._is_discrete:
                return -distance
            else:
                # Button pushed
                if self.terminated and reward > 0:
                    return 50
                # out of bounds
                elif self.terminated and reward < 0:
                    return -250
                # anything else
                else:
                    return -distance

        return reward
