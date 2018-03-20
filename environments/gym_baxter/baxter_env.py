"""
This program allows to run Baxter Gym Environment as a module
"""

import numpy as np
import cv2
import zmq
import gym
from gym import spaces
from gym.utils import seeding
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns

# Baxter-Gazebo bridge specific
from gazebo.constants import SERVER_PORT, HOSTNAME
from gazebo.utils import recvMatrix
from state_representation.episode_saver import EpisodeSaver
from state_representation.models import loadSRLModel


MAX_STEPS = 500
RENDER_HEIGHT = 224
RENDER_WIDTH = 224
N_CONTACTS_BEFORE_TERMINATION = 5
MAX_DISTANCE = 0.8  # Max distance between end effector and the button (for negative reward)
THRESHOLD_DIST_TO_CONSIDER_BUTTON_TOUCHED = 0.01  # Min distance between effector and button
BUTTON_LINK_IDX = 1
RELATIVE_POS = False

# ==== CONSTANTS FOR BAXTER ROBOT ====
DELTA_POS = 0.05
# Each action array is [dx, dy, dz]: representing movements up, down, left, right,
# backward and forward from Baxter coordinate system center
action_dict = {
    0: [- DELTA_POS, 0, 0],
    1: [DELTA_POS, 0, 0],
    2: [0, - DELTA_POS, 0],
    3: [0, DELTA_POS, 0],
    4: [0, 0, - DELTA_POS],
    5: [0, 0, DELTA_POS]
}
N_DISCRETE_ACTIONS = len(action_dict)


# Parameters defined outside init because gym.make() doesn't allow arguments
STATE_DIM = -1  # When learning states
LEARN_STATES = False
USE_SRL = True  # False
SRL_MODEL_PATH = None
RECORD_DATA = True
USE_GROUND_TRUTH = False  # True #False
SHAPE_REWARD = False  # Set to true, reward = -distance_to_goal

# Init seaborn
sns.set()
TITLE_MAX_LENGTH = 60


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


def bgr2rgb(bgr_img):
    """
    Convert an image from BGR to RGB
    :param bgr_img:
    """
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


class BaxterEnv(gym.Env):
    """ Baxter robot arm Environment (Gym wrapper for Baxter Gazebo environment)
        The goal of the robotic arm is to push the button on the table
        :param renders: (bool) Whether to display the GUI or not
        :param is_discrete: (bool) true if action space is discrete vs continuous
        :param log_folder: (str) name of the folder where recorded data will be stored
    """

    def __init__(self,
                 renders=True,
                 is_discrete=True,
                 log_folder="baxter_log_folder"):
        self.n_contacts = 0
        self.use_srl = USE_SRL or USE_GROUND_TRUTH
        self._is_discrete = is_discrete
        self.observation = []
        # Start simulation with first observation
        self._env_step_counter = 0
        self.episode_terminated = False
        self.state_dim = STATE_DIM
        self._renders = renders
        self.np_random = None

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            action_dim = 3
            self._action_bound = 1
            action_bounds = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_bounds, action_bounds, dtype=np.float32)

        if self.use_srl:
            self.dtype = np.float32
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=self.dtype)
        else:
            self.dtype = np.uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=(RENDER_WIDTH, RENDER_HEIGHT, 3),
                                                dtype=self.dtype)

        self.cuda = th.cuda.is_available()
        self.button_pos = None
        self.saver = None

        if RECORD_DATA:
            self.saver = EpisodeSaver(log_folder, MAX_DISTANCE, self.state_dim, globals_=getGlobals(),
                                      relative_pos=RELATIVE_POS,
                                      learn_states=LEARN_STATES)

        # Initialize Baxter effector by connecting to the Gym bridge ROS node:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

        # note: if takes too long, run first client, then server
        print("Waiting for server connection...")
        msg = self.socket.recv_json()
        print("Connected to server (received message: {})".format(msg))

        self.action = [0, 0, 0]
        self.reward = -1
        self.arm_pos = np.array([0, 0, 0])
        self.seed(0)

        # SRL model
        if self.use_srl:
            env_object = self if USE_GROUND_TRUTH else None
            self.srl_model = loadSRLModel(SRL_MODEL_PATH, self.cuda, self.state_dim, env_object)
            self.state_dim = self.srl_model.state_dim

        # Initialize the state
        self.reset()
        if self._renders:
            self.image_plot = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :action: (int)
        """
        assert self.action_space.contains(action)
        # Convert int action to action in (x,y,z) space
        self.action = action_dict[action]
        self._env_step_counter += 1

        # Send the action to the server
        self.socket.send_json({"command": "action", "action": self.action})

        # Receive state data (position, etc), important to update state related values
        self.getEnvState()

        #  Receive a camera image from the server
        self.observation = self.getObservation()
        done = self._hasEpisodeTerminated()
        if self.saver is not None:
            self.saver.step(self.observation, self.action, self.reward, done, self.arm_pos)

        if self.use_srl:
            return self.srl_model.getState(self.observation), self.reward, done, {}
        else:
            return self.observation, self.reward, done, {}

    def getEnvState(self):
        """
        Returns a dictionary containing info about the environment state.
        It also sets the reward: the agent is rewarded for pushing the button
        and reward is discounted if the arm goes outside the bounding sphere
        surrounding the button.
        :return: (dict) state_data containing data related to the state: button_pos,
        arm_pos and reward.
        """
        state_data = self.socket.recv_json()
        self.reward = state_data["reward"]
        self.button_pos = np.array(state_data["button_pos"])
        self.arm_pos = np.array(state_data["position"])  # gripper_pos

        # Compute distance from Baxter left arm to goal (the button_pos)
        distance_to_goal = np.linalg.norm(self.button_pos - self.arm_pos, 2)

        # TODO: tune max distance
        self.n_contacts += self.reward

        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION - 1:
            self.episode_terminated = True

        if distance_to_goal > MAX_DISTANCE:  # outside sphere of proximity
            self.reward = -1

        if SHAPE_REWARD:
            self.reward = -distance_to_goal
        return state_data

    def getObservation(self):
        """
        Required method by gazebo
        """
        # Receive a camera image from the server
        self.observation = recvMatrix(self.socket)
        # Resize it:
        self.observation = cv2.resize(self.observation, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
        return self.observation

    def getArmPos(self):
        """
        :return: ([float])->  np.ndarray Position (x, y, z) of Baxter left gripper
        """
        return self.arm_pos

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.episode_terminated = False
        self.n_contacts = 0
        # Step count since episode start
        self._env_step_counter = 0
        self.socket.send_json({"command": "reset"})
        # Update state related variables, important step to get both data and
        # metadata that allow reading the observation image
        self.getEnvState()
        self.observation = self.getObservation()
        if self.saver is not None:
            self.saver.reset(self.observation, self.button_pos, self.arm_pos)
        if self.use_srl:
            return self.srl_model.getState(self.observation)
        else:
            return self.observation

    def _hasEpisodeTerminated(self):
        """
        Returns if the episode_saver terminated, not of the whole environment run
        """
        if self.episode_terminated or self._env_step_counter > MAX_STEPS:
            return True
        return False

    def closeServerConnection(self):
        """
        To be called at the end of running the program, externally
        """
        print("Baxter_env client exiting and closing socket...")
        self.socket.send_json({"command": "exit"})
        self.socket.close()

    def render(self, mode='rgb_array'):
        """
        :return: (numpy array) BGR image
        """
        if mode != "rgb_array":
            print('render in human mode not yet supported')
            return np.array([])

        plt.ion()  # needed for interactive update
        if self.image_plot is None:
            plt.figure('Baxter RL')
            self.image_plot = plt.imshow(bgr2rgb(self.observation), cmap='gray')
            self.image_plot.axes.grid(False)
        else:
            self.image_plot.set_data(bgr2rgb(self.observation))
        plt.draw()
        # Wait a bit, so that plot is visible
        plt.pause(0.0001)
        return self.observation

    def close(self):
        # TODO: implement close function to close GUI
        pass
