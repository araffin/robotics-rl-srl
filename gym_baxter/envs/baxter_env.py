import numpy as np
import cv2

import zmq
import gym
from gym import spaces
from gym.utils import seeding
import torch as th

# Baxter-Gazebo bridge specific
from gazebo.constants import SERVER_PORT, HOSTNAME
from gazebo.utils import recvMatrix

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
FORCE_RENDER = False  # For enjoy script
BUTTON_LINK_IDX = 1

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
STATE_DIM = -1  # When learning states
LEARN_STATES = False
USE_SRL = False
SRL_MODEL_PATH = None
RECORD_DATA = False
USE_GROUND_TRUTH = False


class BaxterEnv(gym.Env):
    """ Baxter robot arm Environment"
    The goal of the robotic arm is to push the button on the table
    """

    def __init__(self,
                 renders=True,
                 is_discrete=True,
                 name="gym_baxter",  # This name should coincide with the module folder name
                 state_dim=3):
        self.n_contacts = 0
        self.use_srl = USE_SRL or USE_GROUND_TRUTH
        self._is_discrete = is_discrete
        self.observation = []
        # Start simulation with first observation
        self._env_step_counter = 0
        self._renders = renders or FORCE_RENDER
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
        if self.use_srl:
            print('Using learned states')
            self.dtype = np.float32
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=self.dtype)
        else:
            self.dtype = np.uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=self.dtype)
        self.cuda = th.cuda.is_available()
        self.button_pos = None
        self.saver = None
        if RECORD_DATA:
            self.saver = EpisodeSaver(name, MAX_DISTANCE, STATE_DIM, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=LEARN_STATES)
        # SRL model
        if self.use_srl:
            env_object = self if USE_GROUND_TRUTH else None
            self.srl_model = loadSRLModel(SRL_MODEL_PATH, self.cuda, STATE_DIM, env_object)
            self.state_dim = self.srl_model.state_dim

        # Initialize Baxter effector by connecting to the Gym bridge ROS node:
        self.context = zmq.Context()  # TODO: needed in order to apply later destroy?  zmq_ctx_new()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

        print("Waiting for server connection...")  # note: if takes too long, run first client, then server
        msg = self.socket.recv_json()
        print("Connected to server (received message: {})".format(msg))

        # Initialize the state
        self.reset()
        self.action = [0, 0, 0]
        self.reward = -1
        self.arm_pos = [0, 0, 0]
        self.seed(0)

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
        self.socket.send_json({"command": "action", "action": action})

        # Receive state data (position, etc), important to update state related values
        env_state = self.getEnvState()
        #  Receive a camera image from the server
        self.observation = self.getObservation()
        done = self._hasEpisodeTerminated()
        if self.saver is not None:
            self.saver.step(self.observation, self.action, self.reward, done, self.arm_pos)

        if self.use_srl:
            return self.srl_model.getState(self.observation), reward, done, {}
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
        # Compute distance from Baxter left arm to the button_pos
        distance = np.linalg.norm(self.button_pos - self.arm_pos, 2)
        # print('Distance and MAX_DISTANCE {}, {} (TODO: tune max 0.8?)'.format(distance, MAX_DISTANCE))
        self.n_contacts += self.reward
        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION - 1:
            self.episode_terminated = True
        if distance > MAX_DISTANCE:
            self.reward = -1
        # TODO: support reward shaping
        reward_counts[self.reward] = reward_counts.get(self.reward, 0) + 1
        return state_data

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
        # metadata in the image that allow reading an observation
        state = self.getEnvState()
        if self.saver is not None:
            self.saver.reset(self.observation, self.button_pos, self.arm_pos)
        if self.use_srl:
            return self.srl_model.getState(self.observation)
        else:
            return self.getObservation()

    def getObservation(self):
        self.observation = self.render("rgb_array")
        # print('observationa; {}'.format(self.observation))
        return self.observation

    def render(self, mode='rgb_array'):
        """
        Method required by OpenAI Gym.
        Gets from x,y,z sliders, proj_matrix
        Returns an rgb np.array.
        Note: should not be called before sampling an action, or will never return
        """
        if mode != "rgb_array":
            print('render in human mode not yet supported')
            return np.array([])
        # Receive a camera image from the server
        self.img = recvMatrix(self.socket)  # required by gazebo
        if self._renders:
            cv2.imshow("Image", self.img)
        return self.img

    def _hasEpisodeTerminated(self):
        """
        Returns if the episode_saver terminated, not of the whole environment run
        """
        if self.episode_terminated or self._env_step_counter > MAX_STEPS:
            print('Episode Terminated: step counter reached MAX_STEPS: {}. Summary of Reward counts:{}'.format(
                self._env_step_counter, reward_counts))
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
