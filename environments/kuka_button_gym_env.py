import os
import pybullet as p
import time
import math

import gym
import numpy as np
import torch as th
import pybullet_data
from gym import spaces
from gym.utils import seeding

from state_representation.episode_saver import EpisodeSaver
from state_representation.models import loadSRLModel
from . import kuka

MAX_STEPS = 500
N_CONTACTS_BEFORE_TERMINATION = 5
# Terminate the episode if the arm is outside the safety sphere during too much time
N_STEPS_OUTSIDE_SAFETY_SPHERE = 50
RENDER_HEIGHT = 224
RENDER_WIDTH = 224
Z_TABLE = -0.2
MAX_DISTANCE = 0.70  # Max distance between end effector and the button (for negative reward)
N_DISCRETE_ACTIONS = 6
BUTTON_LINK_IDX = 1
BUTTON_GLIDER_IDX = 1  # Button glider joint
DELTA_V = 0.03  # velocity per physics step.
DELTA_THETA = 0.03  # angular velocity per physics step.
RELATIVE_POS = False  # number of timesteps an action is repeated (here it is equivalent to frameskip)
ACTION_REPEAT = 1
# NOISE_STD = DELTA_V / 3 # Add noise to actions, so the env is not fully deterministic
NOISE_STD = 0.01
SHAPE_REWARD = False  # Set to true, reward = -distance_to_goal
N_RANDOM_ACTIONS_AT_INIT = 5  # Randomize init arm pos: take 5 random actions
IS_DISCRETE = True
ACTION_JOINTS = False

# Parameters defined outside init because gym.make() doesn't allow arguments
FORCE_RENDER = False  # For enjoy script
STATE_DIM = -1  # When learning states
LEARN_STATES = False
USE_SRL = False
SRL_MODEL_PATH = None
RECORD_DATA = False
USE_GROUND_TRUTH = False
USE_JOINTS = False
VERBOSE = False  # Whether to print some debug info


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


# TODO: improve the physics of the button


class KukaButtonGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    """
    Gym wrapper for Kuka environment with a push button
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool)
    :param name: (str) name of the folder where recorded data will be stored
    """

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 renders=False,
                 is_discrete=True,
                 name="kuka_button_gym"):
        self._timestep = 1. / 240.
        self._urdf_root = urdf_root
        self._action_repeat = ACTION_REPEAT
        self._observation = []
        self._env_step_counter = 0
        self._renders = renders or FORCE_RENDER
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self._cam_dist = 1.1
        self._cam_yaw = 145
        self._cam_pitch = -36
        self._cam_roll = 0
        self.camera_target_pos = (0.316, -0.2, -0.1)
        self._is_discrete = is_discrete and IS_DISCRETE
        self.terminated = False
        self.renderer = p.ER_TINY_RENDERER
        self.debug = False
        self.n_contacts = 0
        self.state_dim = STATE_DIM
        self.use_srl = USE_SRL or USE_GROUND_TRUTH or USE_JOINTS
        self.use_ground_truth = USE_GROUND_TRUTH
        self.use_joints = USE_JOINTS
        self.action_joints = ACTION_JOINTS
        self.cuda = th.cuda.is_available()
        self.saver = None
        if RECORD_DATA:
            self.saver = EpisodeSaver(name, MAX_DISTANCE, STATE_DIM, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=LEARN_STATES)
        # SRL model
        if self.use_srl:
            env_object = self if USE_GROUND_TRUTH or USE_JOINTS else None
            self.srl_model = loadSRLModel(SRL_MODEL_PATH, self.cuda, STATE_DIM, env_object)
            self.state_dim = self.srl_model.state_dim

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
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

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            if self.action_joints:
                action_dim = 12
                self._action_bound = 1
            else:
                action_dim = 3
                self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)

        if self.use_srl:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        # Create numpy random generator
        # This seed can be changed later
        self.seed(0)

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
        x_pos = 0.5 + 0.0 * self.np_random.uniform(-1, 1)
        y_pos = 0 + 0.0 * self.np_random.uniform(-1, 1)
        self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos = np.array([x_pos, y_pos, Z_TABLE])

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdf_root_path=self._urdf_root, timestep=self._timestep, use_inverse_kinematics=(not self.action_joints))
        self._env_step_counter = 0
        # Close the gripper and wait for the arm to be in rest position
        for _ in range(500):
            self._kuka.applyAction([0, 0, 0, 0, 0])
            p.stepSimulation()

        # Randomize init arm pos: take 5 random actions
        for _ in range(N_RANDOM_ACTIONS_AT_INIT):
            action = [0, 0, 0, 0, 0]
            sign = 1 if self.np_random.rand() > 0.5 else -1
            action_idx = self.np_random.randint(3)  # dx, dy or dz
            action[action_idx] += sign * DELTA_V
            self._kuka.applyAction(action)
            p.stepSimulation()

        self._observation = self.getExtendedObservation()

        self.button_pos = np.array(p.getLinkState(self.button_uid, BUTTON_LINK_IDX)[0])
        if self.saver is not None:
            self.saver.reset(self._observation, self.button_pos, self.getArmPos())

        if self.use_srl:
            # if len(self.saver.srl_model_path) > 0:
            # self.srl_model.load(self.saver.srl_model_path))
            return self.srl_model.getState(self._observation)

        return np.array(self._observation)

    def __del__(self):
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
        self._observation = self.render("rgb_array")
        return self._observation

    def step(self, action):
        """
        :param action: (int)
        """
        self.action = action  # For saver
        if self._is_discrete:
            dv = DELTA_V  # velocity per physics step.
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, -dv][action]  # Remove up action
            # da = [0, 0, 0, 0, 0, -0.1, 0.1][action]  # end effector angle
            finger_angle = 0.0  # Close the gripper
            # real_action = [dx, dy, -0.002, da, finger_angle]
            real_action = [dx, dy, dz, 0, finger_angle]
        else:
            if self.action_joints:
                d_theta = DELTA_THETA
                # Add noise to action
                d_theta += self.np_random.normal(0.0, scale=NOISE_STD)
                real_action = list(action * d_theta) #TODO remove up action 
            else:
                dv = DELTA_V
                # Add noise to action
                dv += self.np_random.normal(0.0, scale=NOISE_STD)
                dx = action[0] * dv
                dy = action[1] * dv
                dz = -abs(action[2] * dv)  # Remove up action
                finger_angle = 0.0  # Close the gripper
                real_action = [dx, dy, dz, 0, finger_angle]

        if VERBOSE:
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

        done = self._termination()
        reward = self._reward()
        if self.saver is not None:
            self.saver.step(self._observation, self.action, reward, done, self.getArmPos())

        if self.use_srl:
            return self.srl_model.getState(self._observation), reward, done, {}

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
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=self._cam_roll,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self.renderer)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        # TODO: implement close function to close GUI
        pass

    def _termination(self):
        if self.terminated or self._env_step_counter > MAX_STEPS:
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

        if distance > MAX_DISTANCE or contact_with_table:
            reward = -1
            self.n_steps_outside += 1
        else:
            self.n_steps_outside = 0

        if contact_with_table or self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION - 1 \
                or self.n_steps_outside >= N_STEPS_OUTSIDE_SAFETY_SPHERE - 1:
            self.terminated = True
        if SHAPE_REWARD:
            return -distance
        return reward
