import time
import logging
import sys

import zmq
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch as th
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt


"""
To run this program, that calls to gazebo server:
1) Start ROS + Gazebo modules
(outside conda env):
natalia@natalia:~$ roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

2) Then start the server:
~/robotics-rl-srl$ python -m gazebo.gazebo_server

3) Test this program in the main repo directory level:
python -m gym_baxter.test_baxter_env

"""

logger = logging.getLogger(__name__)

if sys.version_info > (3,):
    # representing objects in Py3-compatibility
    buffer = memoryview


#from srl_priors.models import SRLCustomCNN
# from srl_priors.preprocessing import preprocessImage
#from state_representation.episode_saver import EpisodeSaver

MAX_STEPS = 5 # TODO 500
N_CONTACTS_BEFORE_TERMINATION = 5
RENDER_HEIGHT = 84  # 720 // 5   # camera image size
RENDER_WIDTH = 84  # 960 // 5
IMG_SHAPE = (3, RENDER_WIDTH, RENDER_HEIGHT)
Z_TABLE = -0.2
MAX_DISTANCE = 0.8  # Max distance between end effector and the button (for negative reward)
THRESHOLD_DIST_TO_CONSIDER_BUTTON_TOUCHED = 0.01 # Min distance between effector and button
FORCE_RENDER = False  # For enjoy script
BUTTON_LINK_IDX = 1
BUTTON_GLIDER_IDX = 1  # Button glider joint
NOISE_STD = 1e-6  # To avoid NaN for SRL

# Baxter-Gazebo bridge specific
# TODO: FIX relative import in other subfolder? or move following import files
#to upper level so this module can also use them?
#from ..gazebo.constants import SERVER_PORT, HOSTNAME or move these to main level is the only choice?
#from .utils import recvMatrix

# ==== CONSTANTS FOR BAXTER ROBOT ====
# Socket port
SERVER_PORT = 7777
REF_POINT = [0.6, 0.30, 0.20]
# ['left_e0', 'left_e1', 'left_s0', 'left_s1', 'left_w0', 'left_w1', 'left_w2']
IK_SEED_POSITIONS = [-1.535, 1.491, -0.038, 0.194, 1.546, 1.497, -0.520]
HOSTNAME = 'localhost'
DELTA_POS = 0.05

UP, DOWN, LEFT, RIGHT, BACKWARD, FORWARD = 1,2,3,4,5,6
action_dict = {
    LEFT: [- DELTA_POS, 0, 0],
    RIGHT: [DELTA_POS, 0, 0],
    DOWN: [0, - DELTA_POS, 0],
    UP: [0, DELTA_POS, 0],
    BACKWARD: [0, 0, - DELTA_POS],
    FORWARD: [0, 0, DELTA_POS]
}
N_DISCRETE_ACTIONS = len(action_dict)

# ROS Topics
IMAGE_TOPIC = "/cameras/head_camera_2/image"
ACTION_TOPIC = "/robot/limb/left/endpoint_action"
BUTTON_POS_TOPIC = "/button1/position"
EXIT_KEYS = [113, 27]  # Escape and q

# Parameters defined outside init because gym.make() doesn't allow arguments
STATE_DIM = -1  # When learning states
LEARN_STATES = False
USE_SRL = False
SRL_MODEL_PATH = None
RECORD_DATA = False
USE_GROUND_TRUTH = False

def recvMatrix(socket):
    """
    Receive a numpy array over zmq
    :param socket: (zmq socket)
    :return: (Numpy matrix)
    """
    metadata = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    A = np.frombuffer(buffer(msg), dtype=metadata['dtype'])
    return A.reshape(metadata['shape'])

class BaxterEnv(gym.Env):
    """ Baxter robot arm Environment"
    The goal of the robotic arm is to push the button on the table
    """

    metadata = {'render.modes': ['human', 'rgb_array'],  # human is for human teleoperation (see teleop_cliente example)
        'video.frames_per_second': 50}

    def __init__(self,
                 action_repeat=1,
                 renders=False,
                 is_discrete=True,
                 name="gym_baxter",  #TODO this name should coincide with the module folder name
                 state_dim=3):
        self.use_srl = USE_SRL or USE_GROUND_TRUTH
        self.range = 1000  # +/- value the randomly select number can be between
        self._is_discrete = is_discrete
        self.reward_range = (-1, 1)
        self.max_actions = 5  # 50
        self.observation = []  # shall it be an integer or a list of action's values per dimension?

        self._timestep = 1. / 240.
        self._action_repeat = action_repeat
        # Start simulation with first observation
        self._env_step_counter = 0
        self._renders = renders or FORCE_RENDER
        self.contact_points = 0
        self.episode_terminated = False
        self.state_dim = state_dim
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            action_dim = 3
            self._action_bound = 1
            action_bounds = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_bounds, action_bounds, dtype=np.float32)
            #self.bounds = 2000  # Action space bounds #self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]), dtype=np.float32)
        if self.use_srl:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        self.cuda = th.cuda.is_available()

        # TODO indicate in readme how to use
        self.saver = None
        #self.saver = EpisodeSaver(name, MAX_DISTANCE, state_dim, relative_pos=False)
        # SRL model
        if self.use_srl:
            self.srl_model = SRLCustomCNN(state_dim, self.cuda, noise_std=NOISE_STD)
            if self.cuda:
                self.srl_model.cuda()

        # Initialize button position
        x_pos = 0.5 + 0.0 * np.random.uniform(-1, 1)
        y_pos = 0 + 0.0 * np.random.uniform(-1, 1)
        #TODO: Get position of the button -> randomly relocate in each episode
        self.button_pos = np.array([x_pos, y_pos, Z_TABLE])
        # Initialize Baxter effector by connecting to the Gym bridge ROS node:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

        print("Waiting for server connection...") # note: if takes too long, run first client, then server
        msg = self.socket.recv_json()
        print("Connected to server")

        # Initialize the state
        self.reset()
        self.action = [0, 0, 0]
        self.reward = -100
        self.arm_pos = [0, 0, 0]
        if self.use_srl:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ), dtype=np.float32)
        else:  # raw images
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        # Create numpy random generator,  This seed can be changed later
        self.seed(0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.action = action  # For saver
        self._env_step_counter += 1
        if action in action_dict.keys():
            action = action_dict[action]
        else:
            print("Unknown action: {}".format(action))
            action = [0, 0, 0]

        # TODO Load SRL model
        # if self.use_srl:
        #     env_object = self if USE_GROUND_TRUTH else None
        #     self.srl_model = loadSRLModel(SRL_MODEL_PATH, self.cuda, STATE_DIM, env_object)
        #     self.state_dim = self.srl_model.state_dim

        # Send the action to the server
        print("sending action to server {}...".format(action))
        self.socket.send_json({"command": "action", "action": action})

        # Receive state data (position, etc)
        env_state = self.getEnvState()

        # Receive a camera image from the server
        self.observation = self.getObservation()

        done = self._hasEpisodeTerminated()
        print("env_state: {}".format(env_state))
        return self.observation, self.reward, done, {}  #np.array(self.observation)

        # if self.saver is not None:
        #     self.saver.step(self.observation, self.action, reward, done, self.arm_pos)
        # if self.use_srl:
        #     return self.getState(self.observation), reward, done, {}
        # else:
        #     return self.observation, reward, done, {}  #np.array(self.observation)

    def getEnvState(self):
        """
        Agent is rewarded for pushing the button and reward is discounted if the
        arm goes outside the bounding sphere surrounding the button.
        """
        state_data = self.socket.recv_json()
        self.reward = state_data["reward"]
        self.button_pos = np.array(state_data["button_pos"])
        self.arm_pos =  np.array(state_data["position"]) # gripper_pos
        # Compute distance from Baxter left arm to the button_pos
        distance = np.linalg.norm(self.button_pos - self.arm_pos, 2)
        print('Distance and MAX_DISTANCE {}, {}'.format(distance, MAX_DISTANCE))
        self.n_contacts += self.reward
        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION -1:
            self.episode_terminated = True
        if distance > MAX_DISTANCE:
            self.reward = -1
        return state_data

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.episode_terminated = False
        self.n_contacts = 0
        # TODO? Randomize init arm pos: take 5 random actions?
        # Step count since episode start
        self._env_step_counter = 0
        self.socket.send_json({
            "command":"reset"
        })
        # if self.use_srl: # TODO: WHY: AttributeError: 'BaxterEnv' object has no attribute 'use_srl'
        #     return self.srl_model.getState(self.getObservation())
        # else:
        self.getEnvState()
        return self.getObservation() # np.array(self.observation)# return super(BaxterEnv, self)._reset()

    def getObservation(self):
        self.observation = self.render("rgb_array")
        return self.observation

    def render(self, mode='rgb_array', close=False):
        """
        Method required by OpenAI Gym.
        Gets from x,y,z sliders, proj_matrix
        Returns an rgb np.array
        """
        if mode != "rgb_array":
            print ('render in human mode not yet supported')
            return np.array([])
        # Receive a camera image from the server
        self.img = recvMatrix(self.socket) # required by gazebo
        return self.img

    def _hasEpisodeTerminated(self):
        """
        Returns if the episode_saver terminated, not of the whole environment run
        """
        if self.episode_terminated or self._env_step_counter > MAX_STEPS:
            print('Episode Terminated: step counter reached MAX_STEPS: {}'.format(self._env_step_counter))
            return True
        return False

    def closeServerConnection(self):
        """
        To be called at the end of running the program, externally
        """
        print("Baxter_env client exiting and closing socket...")
        self.socket.send_json({"command": "exit"})
        cv2.destroyAllWindows()
        self.socket.close()
        # Terminate the context to close the socket
        self.context.term() # TODO THESE NEEDED?
        #zmq_ctx_destroy() # destroys context so program exits gracefully
        # if self.socket is not None: # socket.close should do this
        #     os.kill(self.socket.pid, signal.SIGKILL)
