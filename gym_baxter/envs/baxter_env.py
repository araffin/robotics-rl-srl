import time
from textwrap import fill

import numpy as np
import cv2
import zmq
import gym
from gym import spaces
from gym.utils import seeding
import torch as th
import  matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Baxter-Gazebo bridge specific
from gazebo.constants import SERVER_PORT, HOSTNAME
from gazebo.utils import recvMatrix
from state_representation.episode_saver import EpisodeSaver
from state_representation.models import loadSRLModel

"""
This program allows to run Baxter Gym Environment as a module
"""

MAX_STEPS = 500
N_CONTACTS_BEFORE_TERMINATION = 5
RENDER_HEIGHT = 84  # 720 // 5   # camera image size
RENDER_WIDTH = 84  # 960 // 5
IMG_SHAPE = (3, RENDER_WIDTH, RENDER_HEIGHT)
Z_TABLE = -0.2
MAX_DISTANCE = 0.8  # Max distance between end effector and the button (for negative reward)
THRESHOLD_DIST_TO_CONSIDER_BUTTON_TOUCHED = 0.01  # Min distance between effector and button
BUTTON_LINK_IDX = 1
RELATIVE_POS = False  # number of timesteps an action is repeated (here it is equivalent to frameskip)

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
# logging anomalies
reward_counts = {}

# ROS Topics
IMAGE_TOPIC = "/cameras/head_camera_2/image"
ACTION_TOPIC = "/robot/limb/left/endpoint_action"
BUTTON_POS_TOPIC = "/button1/position"
EXIT_KEYS = [113, 27]  # Escape and q

# Parameters defined outside init because gym.make() doesn't allow arguments
STATE_DIM = 3 # When learning states
LEARN_STATES = False
USE_SRL = True #False
SRL_MODEL_PATH = "/home/natalia/srl-robotic-priors-pytorch/logs/staticButtonSimplest/18-03-15_17h00_18_custom_cnn_ProTemCauRep_ST_DIM6_SEED0_priors/srl_model.pth"
RECORD_DATA = True
USE_GROUND_TRUTH = False # True #False
SHAPE_REWARD = False  # Set to true, reward = -distance_to_goal


# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Init seaborn
sns.set()
TITLE_MAX_LENGTH = 60


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class BaxterEnv(gym.Env):
    """ Baxter robot arm Environment (Gym wrapper for Baxter Gazebo environment)
        The goal of the robotic arm is to push the button on the table
        :param renders: (bool) Whether to display the GUI or not
        :param is_discrete: (bool) true if action space is discrete vs continuous
        :param name: (str) name of the folder where recorded data will be stored
        :param data_log: (str) name of the folder where recorded data will be stored
        :state_dim: dimensionality of the states learned/to learn
    """

    def __init__(self,
                 renders=True,
                 is_discrete=True,
                 name="gym_baxter",  # This name should coincide with the module folder name
                 data_log="baxter_data_log",
                 state_dim= STATE_DIM):
        self.n_contacts = 0
        self.use_srl = USE_SRL or USE_GROUND_TRUTH
        self._is_discrete = is_discrete
        self.observation = []
        # Start simulation with first observation
        self._env_step_counter = 0
        self.episode_terminated = False
        self.state_dim = state_dim
        self._renders = renders
        if self._renders:
            self._width = RENDER_WIDTH
            self._height = RENDER_HEIGHT
            self._timestep = 1. / 240.
            #plt.ion() # necessary for real-time plotting and the figure to show when updating with fig.canvas.draw().
            # probably not need this if you're embedding things in a tkinter plot.

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
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=self.dtype)
        self.cuda = th.cuda.is_available()
        self.button_pos = None
        self.saver = None
        if RECORD_DATA:
            self.saver = EpisodeSaver(data_log, MAX_DISTANCE, self.state_dim, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=LEARN_STATES)

        # Initialize Baxter effector by connecting to the Gym bridge ROS node:
        self.context = zmq.Context()  # TODO: needed in order to apply later destroy?  zmq_ctx_new()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

        print("Waiting for server connection...")  # note: if takes too long, run first client, then server
        msg = self.socket.recv_json()
        print("Connected to server (received message: {})".format(msg))

        self.action = [0, 0, 0]
        self.reward = -1
        self.arm_pos = [0, 0, 0] # np.ndarray
        self.seed(0)

        # SRL model
        if self.use_srl:
            env_object = self if USE_GROUND_TRUTH else None
            self.srl_model = loadSRLModel(SRL_MODEL_PATH, self.cuda, self.state_dim, env_object)
            self.state_dim = self.srl_model.state_dim
            print('Using learned states with dim {}, rendering {} and model {}'.format(self.state_dim, self._renders, self.srl_model))

        # Initialize the state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :action: (int)
        """
        assert self.action_space.contains(action)
        self.action = action_dict[action]  # For saver
        self._env_step_counter += 1

        # Send the action to the server
        self.socket.send_json({"command": "action", "action": self.action})

        # Receive state data (position, etc), important to update state related values
        self.reward = self.getEnvState()['reward']
        reward_counts[self.reward] = reward_counts.get(self.reward, 0) + 1

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
        print('state_data {}'.format(state_data))
        self.reward = state_data["reward"]
        self.button_pos = np.array(state_data["button_pos"])
        self.arm_pos = np.array(state_data["position"])  # gripper_pos
        # Compute distance from Baxter left arm to goal (the button_pos)
        distance_to_goal = np.linalg.norm(self.button_pos - self.arm_pos, 2)
        # print('Distance and MAX_DISTANCE {}, {} (TODO: tune max 0.8?)'.format(distance_to_goal, MAX_DISTANCE))
        self.n_contacts += self.reward
        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION - 1:
            self.episode_terminated = True
        if distance_to_goal > MAX_DISTANCE: # outside sphere of proximity
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
        self.socket.send_json({
            "command": "reset"
        })
        # update state related variables, important step to get both data and
        # metadata that allow reading the observation image
        state = self.getEnvState()
        self.observation = self.getObservation()
        # close plots GUI
        #plt.close()
        #fig.canvas.flush_events()
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
        print('\nStep counter reached MAX_STEPS: {}. Summary of reward counts:{}'.format(self._env_step_counter, reward_counts))
        print("Baxter_env client exiting and closing socket...")
        self.socket.send_json({"command": "exit"})
        cv2.destroyAllWindows()
        self.socket.close()

    def render(self, mode='rgb_array'):
        """
        Method required by OpenAI Gym.
        Gets from x,y,z sliders, proj_matrix
        Returns an rgb np.array.
        Note: should be called only from within the environment, not to hang
        the program, since it requires previous to render, to call recv_json()
        """
        if mode != "rgb_array":
            print('render in human mode not yet supported')
            return np.array([])
        #print('rendering image of length: {}'.format(self.observation.shape))

        # SIMPLE WAY 1 NOT interactivele
        # plt.ion()
        # cv2.imshow("Image", self.observation) # this option does not render image
        # plt.imshow(self.observation, cmap='gray')  # this one does
        # plt.show()

        # SECOND WAY UPDATING NOT CLOSING WINDOWS AND REUSING ONE
        # fig = plt.figure()
        # timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
        # timer.add_callback(self.close_event)
        # plt.imshow(self.observation, cmap='gray')
        # plt.ylabel('Baxter camera')
        #
        # fig.canvas.draw()
        # plt.pause(0.05)
        # timer.start()
        # #plt.show()
        # time.sleep(1e-6) #unnecessary, but useful   #time.sleep(1) #time.sleep(self._timestep)
        # # produces seg fault? : fig.canvas.flush_events()


        # ANTONIN WAY NOT SHOWING Image
        plt.ion()
        fig = plt.figure('Baxter reinforcement learning')
        plt.clf()
        #ax = fig.add_subplot(111)
        # im = ax.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
        plt.imshow(self.observation, cmap='gray')
        fig.canvas.draw()
        # ax.set_xlabel('State dimension 1')
        # ax.set_ylabel('State dimension 2')
        #ax.set_title(fill("Baxter", TITLE_MAX_LENGTH))
        #fig.tight_layout()
        createInteractivePlot(fig, self.observation)
        plt.show()

        return self.observation

    def close_event(self):
        """ A timer calls this function after 3 secs and closes the rendering Window"""
        plt.close()

def createInteractivePlot(fig, img_array):
    fig2 = plt.figure("Image")
    image_plot = plt.imshow(img_array)
    # Disable seaborn grid
    image_plot.axes.grid(False)
    callback = ImageSetter(fig, img_array)
    fig.canvas.mpl_connect('button_release_event', callback)

class ImageSetter(object):
    """
    Callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.
    """

    def __init__(self, image_plot, img):

        self.image_plot = image_plot
        self.img = img

    def __call__(self, event):
        if event.inaxes:
            click_x = event.xdata
            click_y = event.ydata
        print('Updating Image Setter info... THIS IS NEVER CALLED{}'.format(self.img))
        title = "Baxter RL"
            #self.image_plot.axes.set_title(title)
            # Load the image that corresponds to the clicked point in the space
        self.image_plot.set_data(self.img)
