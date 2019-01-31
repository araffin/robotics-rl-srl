"""
This program allows to run Omnirobot Gym Environment as a module
"""

import numpy as np
import cv2
import zmq
from gym import spaces
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns

from environments.srl_env import SRLGymEnv
from real_robots.constants import SERVER_PORT, HOSTNAME, MAX_STEPS
from real_robots.utils import recvMatrix
from state_representation.episode_saver import EpisodeSaver

RENDER_HEIGHT = 480
RENDER_WIDTH = 480
RELATIVE_POS = False

N_DISCRETE_ACTIONS = 4

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


class OmniRobotEnv(SRLGymEnv):
    """
    OmniRobot robot Environment (Gym wrapper for OmniRobot environment)
    The goal of Omnirobot is to go to the location on the table
    (signaled with a circle sticker on the table)
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) true if action space is discrete vs continuous
    :param log_folder: (str) name of the folder where recorded data will be stored
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param env_rank: (int) the number ID of the environment
    :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
    """

    def __init__(self, renders=False, is_discrete=True, log_folder="omnirobot_log_folder", state_dim=-1,
                 learn_states=False, srl_model="raw_pixels", record_data=False, action_repeat=1,
                 shape_reward=False, env_rank=0, srl_pipe=None,**_):

        super(OmniRobotEnv, self).__init__(srl_model=srl_model,
                                        relative_pos=RELATIVE_POS,
                                        env_rank=env_rank,
                                        srl_pipe=srl_pipe)
        if action_repeat != 1:
            raise NotImplementedError
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
        self.target_pos = None
        self.saver = None

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            action_dim = 2
            self._action_bound = 1
            action_bounds = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_bounds, action_bounds, dtype=np.float32)
        # SRL model
        if self.use_srl:
            if use_ground_truth:
                self.state_dim = self.getGroundTruthDim()
            self.dtype = np.float32
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=self.dtype)
        else:
            self.dtype = np.uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=(RENDER_WIDTH, RENDER_HEIGHT, 3),
                                                dtype=self.dtype)

        if record_data:
            print("Recording data...")
            self.saver = EpisodeSaver(log_folder, 0, self.state_dim, globals_=getGlobals(),
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

        self.action = [0, 0]
        self.reward = 0
        self.robot_pos = np.array([0, 0])

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
        
        # serialize the action
        if isinstance(action, np.ndarray):
            self.action = action.tolist()
        elif hasattr(action, 'dtype'): # convert numpy type to python type
            self.action = action.item()
        else:
            self.action = action
            
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
        :return: (dict) state_data containing data related to the state: target_pos,
        robot_pos and reward.
        """
        state_data = self.socket.recv_json()
        self.reward = state_data["reward"]
        self.target_pos = np.array(state_data["target_pos"])
        self.robot_pos = np.array(state_data["position"])

        return state_data

    def getObservation(self):
        """
        Receive the observation image using a socket
        :return: (numpy ndarray) observation
        """
        # Receive a camera image from the server
        self.observation = recvMatrix(self.socket)
        # Resize it:
        self.observation = cv2.resize(self.observation, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
        return self.observation

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        return self.target_pos

    @staticmethod
    def getGroundTruthDim():
        """
        :return: (int)
        """
        return 2

    def getGroundTruth(self):
        """
        Alias for getRobotPos for compatibility between envs
        :return: (numpy array)
        """
        return np.array(self.getRobotPos())

    def getRobotPos(self):
        """
        :return: ([float])->  np.ndarray Position (x, y, z) of Baxter left gripper
        """
        return self.robot_pos

    def reset(self):
        """
        Reset the environment
        :return: (numpy ndarray) first observation of the env
        """
        self.episode_terminated = False
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
        print("Omnirobot_env client exiting and closing socket...")
        self.socket.send_json({"command": "exit"})
        self.socket.close()

    def render(self, mode='rgb_array'):
        """
        :param mode: (str)
        :return: (numpy array) BGR image
        """
        if mode != "rgb_array":
            print('render in human mode not yet supported')
            return np.array([])

        if self._renders:
            plt.ion()  # needed for interactive update
            if self.image_plot is None:
                plt.figure('Omnirobot RL')
                self.image_plot = plt.imshow(bgr2rgb(self.observation), cmap='gray')
                self.image_plot.axes.grid(False)
            else:
                self.image_plot.set_data(bgr2rgb(self.observation))
            plt.draw()
            # Wait a bit, so that plot is visible
            plt.pause(0.0001)
        return self.observation

