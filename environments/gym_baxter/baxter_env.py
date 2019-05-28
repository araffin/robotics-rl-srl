"""
This program allows to run Baxter Gym Environment as a module
"""

import numpy as np
import cv2
import zmq
from gym import spaces
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns

# Baxter-Gazebo bridge specific
from environments.srl_env import SRLGymEnv
from real_robots.constants import SERVER_PORT, HOSTNAME, Z_TABLE, DELTA_POS, MAX_DISTANCE, MAX_STEPS
from real_robots.utils import recvMatrix
from state_representation.episode_saver import EpisodeSaver

RENDER_HEIGHT = 224
RENDER_WIDTH = 224
N_CONTACTS_BEFORE_TERMINATION = 2
RELATIVE_POS = True

# ==== CONSTANTS FOR BAXTER ROBOT ====
# Each action array is [dx, dy, dz]: representing movements up, down, left, right,
# backward and forward from Baxter coordinate system center
action_dict = {
    0: [- DELTA_POS, 0, 0],
    1: [DELTA_POS, 0, 0],
    2: [0, - DELTA_POS, 0],
    3: [0, DELTA_POS, 0],
    4: [0, 0, - DELTA_POS],
    # Remove Up action
    # 5: [0, 0, DELTA_POS]
}
N_DISCRETE_ACTIONS = len(action_dict)

# Init seaborn
sns.set()


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


def bgr2rgb(bgr_img):
    """
    Convert an image from BGR to RGB
    :param bgr_img: np.ndarray
    :return: np.ndarray
    """
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


class BaxterEnv(SRLGymEnv):
    """
    Baxter robot arm Environment (Gym wrapper for Baxter Gazebo environment)
    The goal of the robotic arm is to push the button on the table
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) true if action space is discrete vs continuous
    :param log_folder: (str) name of the folder where recorded data will be stored
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param srl_model: (str) Path to the srl model
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param env_rank: (int) the number ID of the environment
    :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
    """

    def __init__(self, renders=False, is_discrete=True, log_folder="baxter_log_folder", state_dim=-1,
                 learn_states=False, record_data=False,
                 shape_reward=False, env_rank=0, srl_pipe=None, srl_model="raw_pixels", img_shape=None):
        super(BaxterEnv, self).__init__(srl_model=srl_model,
                                        relative_pos=RELATIVE_POS,
                                        env_rank=env_rank,
                                        srl_pipe=srl_pipe)
        self.n_contacts = 0
        use_ground_truth = srl_model == 'ground_truth'
        use_srl = srl_model != 'raw_pixels'
        self.use_srl = use_srl or use_ground_truth
        self.use_ground_truth = use_ground_truth
        self.use_joints = False
        self.relative_pos = RELATIVE_POS
        self._is_discrete = is_discrete
        self.observation = []
        # Start simulation with first observation
        self._env_step_counter = 0
        self.episode_terminated = False
        self.state_dim = state_dim
        self._renders = renders
        self._shape_reward = shape_reward
        self.cuda = th.cuda.is_available()
        self.button_pos = None
        self.saver = None
        if img_shape is None:
            self.img_shape = (3, RENDER_HEIGHT, RENDER_WIDTH)
        else:
            self.img_shape = img_shape

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            action_dim = 3
            self._action_bound = 1
            action_bounds = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_bounds, action_bounds, dtype=np.float32)
        # SRL model
        if self.srl_model != "raw_pixels":
            if self.srl_model == "ground_truth":
                self.state_dim = self.getGroundTruthDim()
            self.dtype = np.float32
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=self.dtype)
        else:
            self.dtype = np.uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.img_shape[2], self.img_shape[1], 3),
                                                dtype=self.dtype)

        if record_data:
            print("Recording data...")
            self.saver = EpisodeSaver(log_folder, MAX_DISTANCE, self.state_dim, globals_=getGlobals(),
                                      relative_pos=RELATIVE_POS,
                                      learn_states=learn_states)

        # Initialize Baxter effector by connecting to the Gym bridge ROS node:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

        # note: if takes too long, run first client, then server
        print("Waiting for server connection...")
        msg = self.socket.recv_json()
        print("Connected to server (received message: {})".format(msg))

        self.action = [0, 0, 0]
        self.reward = 0
        self.arm_pos = np.array([0, 0, 0])

        # Initialize the state
        if self._renders:
            self.image_plot = None

    def step(self, action):
        """
        :action: (int)
        :return: (tensor (np.ndarray)) observation, int reward, bool done, dict extras)
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
            self.saver.step(self.observation, action, self.reward, done, self.getGroundTruth())
        if self.use_srl:
            return self.getSRLState(self.observation), self.reward, done, {}
        else:
            return self.observation, self.reward, done, {}

    def getEnvState(self):
        """
        Returns a dictionary containing info about the environment state.
        It also sets the reward: the agent is rewarded for pushing the button
        and the reward value is negative if the arm goes outside the bounding sphere
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

        contact_with_table = self.arm_pos[2] < Z_TABLE - 0.01

        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION or contact_with_table:
            self.episode_terminated = True

        # print("dist=", distance_to_goal)

        if distance_to_goal > MAX_DISTANCE or contact_with_table:  # outside sphere of proximity
            self.reward = -1

        # print('state_data: {}'.format(state_data))

        if self._shape_reward:
            self.reward = -distance_to_goal
        return state_data

    def getObservation(self):
        """
        Receive the observation image using a socket (required method by gazebo)
        :return: np.ndarray (tensor) observation
        """
        # Receive a camera image from the server
        self.observation = recvMatrix(self.socket)
        # Resize it:
        self.observation = cv2.resize(self.observation, (self.img_shape[2], self.img_shape[1]), interpolation=cv2.INTER_AREA)
        return self.observation

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        return self.button_pos

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
        :return: ([float])->  np.ndarray Position (x, y, z) of Baxter left gripper
        """
        return self.arm_pos

    def reset(self):
        """
        Reset the environment
        :return: (numpy ndarray) first observation of the env
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
            self.saver.reset(self.observation, self.getTargetPos(), self.getGroundTruth())
        if self.use_srl:
            return self.getSRLState(self.observation)
        else:
            return self.observation

    def _hasEpisodeTerminated(self):
        """
        Returns True if the episode is over and False otherwise
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

        if self._renders:
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

