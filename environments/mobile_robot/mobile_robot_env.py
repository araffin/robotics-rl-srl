import os
import pybullet as p

import gym
import numpy as np
import torch as th
import pybullet_data
from gym import spaces
from gym.utils import seeding

from state_representation.episode_saver import EpisodeSaver
from state_representation.models import loadSRLModel

#  Number of steps before termination
MAX_STEPS = 250  # WARNING: should be also change in __init__.py (timestep_limit)
# Terminate the episode if the arm is outside the safety sphere during too much time
REWARD_DIST_THRESHOLD = 0.4  # Min distance to target before finishing an episode
RENDER_HEIGHT = 224
RENDER_WIDTH = 224
N_DISCRETE_ACTIONS = 4

DELTA_POS = 0.1  # DELTA_POS
RELATIVE_POS = True  # Use relative position for ground truth
NOISE_STD = 0.0

# To avoid calling disconnect in the __del__ method when not needed
CONNECTED_TO_SIMULATOR = False
# From urdf file, to create bounding box
ROBOT_WIDTH = 0.2
ROBOT_LENGTH = 0.325 * 2

def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class MobileRobotGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    """
    Gym wrapper for Mobile Robot environment
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) Whether to use discrete or continuous actions
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param use_srl: (bool) Set to true, use srl_models
    :param srl_model_path: (str) Path to the srl model
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param use_ground_truth: (bool) Set to true, the observation will be the ground truth (arm position)
    :param random_target: (bool) Set the target to a random position
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    """

    def __init__(self, urdf_root=pybullet_data.getDataPath(), renders=False, is_discrete=True,
                 name="kuka_button_gym", max_distance=1.6, shape_reward=False,
                 use_srl=False, srl_model_path=None, record_data=False, use_ground_truth=False,
                 random_target=False, force_down=True, state_dim=-1, learn_states=False, verbose=False,
                 save_path='srl_priors/data/', **kwargs):
        self._timestep = 1. / 240.
        self._urdf_root = urdf_root
        self._observation = []
        self._env_step_counter = 0
        self._renders = renders
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self._cam_dist = 4.4
        self._cam_yaw = 90
        self._cam_pitch = -90
        self._cam_roll = 0
        self._max_distance = max_distance
        self._shape_reward = shape_reward
        self._random_target = random_target
        self._force_down = force_down
        self.camera_target_pos = (2, 2, 0)
        self._is_discrete = is_discrete
        self.terminated = False
        self.renderer = p.ER_TINY_RENDERER
        self.debug = False
        self.n_contacts = 0
        self.state_dim = state_dim
        self.use_srl = use_srl or use_ground_truth
        self.use_ground_truth = use_ground_truth
        self.relative_pos = RELATIVE_POS
        self.cuda = th.cuda.is_available()
        self.saver = None
        self.verbose = verbose
        self.max_steps = MAX_STEPS
        self.robot_pos = np.zeros(3)
        self.robot_uid = None
        self.target_pos = np.zeros(3)
        self.target_uid = None
        self.np_random = None
        # Boundaries of the square env
        self._min_x, self._max_x = 0, 4
        self._min_y, self._max_y = 0, 4
        self.has_bumped = False  # Used for handling collisions
        self.collision_margin = 0.1
        self.walls = None
        self.use_joints = False  # For compatibility

        if record_data:
            self.saver = EpisodeSaver(name, max_distance, state_dim, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=learn_states, path=save_path)
        # SRL model
        if self.use_srl:
            env_object = self if use_ground_truth else None
            self.srl_model = loadSRLModel(srl_model_path, self.cuda, state_dim, env_object)
            self.state_dim = self.srl_model.state_dim

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
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
            raise ValueError("Only discrete actions is supported")

        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)

        if self.use_srl:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        # Create numpy random generator
        # This seed can be changed later
        self.seed(0)

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        # Return only the [x, y] coordinates
        return self.target_pos[:2]

    def getGroundTruthDim(self):
        """
        :return: (int)
        """
        return 2

    def getGroundTruth(self):
        """
        Alias for getArmPos for compatibility between envs
        :return: (numpy array)
        """
        # Return only the [x, y] coordinates
        return np.array(self.robot_pos)[:2]

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, 0])
        p.setGravity(0, 0, -10)

        # Init the robot randomly
        x_start = self._max_x / 2 + self.np_random.uniform(- self._max_x / 3, self._max_x / 3)
        y_start = self._max_y / 2 + self.np_random.uniform(- self._max_y / 3, self._max_y / 3)
        self.robot_pos = np.array([x_start, y_start, 0])
        # Initialize target position
        x_pos = 0.9 * self._max_x
        y_pos = self._max_y * 3 / 4
        if self._random_target:
            margin = 0.1 * self._max_x
            x_pos = self.np_random.uniform(self._min_x + margin, self._max_x - margin)
            y_pos = self.np_random.uniform(self._min_y + margin, self._max_y - margin)

        self.target_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, 0])
        self.target_pos = np.array([x_pos, y_pos, 0])

        # Add walls
        # rgba colors
        red, green, blue = [0.8, 0, 0, 1], [0, 0.8, 0, 1], [0, 0, 0.8, 1]

        wall_left = p.loadURDF("/urdf/wall.urdf", [self._max_x / 2, 0, 0])
        p.changeVisualShape(wall_left, -1, rgbaColor=red)

        wall_bottom = p.loadURDF("/urdf/wall.urdf", [self._max_x, self._max_y / 2, 0],
                                 p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        wall_right = p.loadURDF("/urdf/wall.urdf", [self._max_x / 2, self._max_y, 0])
        p.changeVisualShape(wall_right, -1, rgbaColor=green)

        wall_top = p.loadURDF("/urdf/wall.urdf", [self._min_x, self._max_y / 2, 0],
                              p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        p.changeVisualShape(wall_top, -1, rgbaColor=blue)

        self.walls = [wall_left, wall_bottom, wall_right, wall_top]

        # Add mobile robot
        self.robot_uid = p.loadURDF(os.path.join(self._urdf_root, "racecar/racecar.urdf"), self.robot_pos,
                                    useFixedBase=False)

        self._env_step_counter = 0
        for _ in range(50):
            p.stepSimulation()

        self._observation = self.getObservation()

        if self.saver is not None:
            self.saver.reset(self._observation, self.getTargetPos(), self.getGroundTruth())

        if self.use_srl:
            # if len(self.saver.srl_model_path) > 0:
            # self.srl_model.load(self.saver.srl_model_path))
            return self.srl_model.getState(self._observation)

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

    def getObservation(self):
        self._observation = self.render("rgb_array")
        return self._observation

    def step(self, action):
        """
        :param action: (int)
        """
        # True if it has bumped against a wall
        self.has_bumped = False
        if self._is_discrete:
            dv = DELTA_POS
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            dx = [-dv, dv, 0, 0][action]
            dy = [0, 0, -dv, dv][action]
            real_action = np.array([dx, dy])
        else:
            raise ValueError("Only discrete actions is supported")

        if self.verbose:
            print(np.array2string(np.array(real_action), precision=2))

        previous_pos = self.robot_pos.copy()
        self.robot_pos[:2] += real_action
        # Handle collisions
        for i, (limit, robot_dim) in enumerate(zip([self._max_x, self._max_y], [ROBOT_LENGTH, ROBOT_WIDTH])):
            margin = self.collision_margin + robot_dim / 2
            # If it has bumped against a wall, stay at the previous position
            if self.robot_pos[i] < margin or self.robot_pos[i] > limit - margin:
                self.has_bumped = True
                self.robot_pos = previous_pos
                break
        # Update mobile robot position
        p.resetBasePositionAndOrientation(self.robot_uid, self.robot_pos, [0, 0, 0, 1])

        p.stepSimulation()
        self._env_step_counter += 1

        self._observation = self.getObservation()

        reward = self._reward()
        done = self._termination()
        if self.saver is not None:
            self.saver.step(self._observation, action, reward, done, self.getGroundTruth())

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
        (_, _, px1, _, _) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self.renderer)
        rgb_array = np.array(px1)

        rgb_array_res = rgb_array[:, :, :3]
        return rgb_array_res

    def close(self):
        # TODO: implement close function to close GUI
        pass

    def _termination(self):
        if self.terminated or self._env_step_counter > self.max_steps:
            self._observation = self.getObservation()
            return True
        return False

    def _reward(self):
        distance = np.linalg.norm(self.getTargetPos() - self.robot_pos[:2], 2)
        reward = 0

        if distance <= REWARD_DIST_THRESHOLD:
            reward = 1
            # self.terminated = True

        # Negative reward when it bumps into a wall
        if self.has_bumped:
            reward = -1

        if self._shape_reward:
            return -distance
        return reward
