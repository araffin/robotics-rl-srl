import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
import torch as th
from torch.autograd import Variable

import logging
logger = logging.getLogger(__name__)

# from srl_priors.models import SRLCustomCNN
# from srl_priors.preprocessing import preprocessImage
# from state_representation.episode_saver import EpisodeSaver

"""
To test this program, in the main repo directory level:
python -m environments.test_baxter_env

"""
MAX_STEPS = 500
N_CONTACTS_BEFORE_TERMINATION = 5
RENDER_HEIGHT = 84  # 720 // 5
RENDER_WIDTH = 84  # 960 // 5
Z_TABLE = -0.2
MAX_DISTANCE = 0.5  # Max distance between end effector and the button (for negative reward)
THRESHOLD_DIST_TO_CONSIDER_BUTTON_TOUCHED = 0.01 # Min distance between effector and button
FORCE_RENDER = False  # For enjoy script
BUTTON_LINK_IDX = 1
BUTTON_GLIDER_IDX = 1  # Button glider joint
NOISE_STD = 1e-6  # To avoid NaN for SRL


# Baxter-Gazebo bridge specific
import zmq, time
# TODO: FIX relative import in other subfolder? from ..gazebo.constants import SERVER_PORT, HOSTNAME or move these to main level is the only choice?
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


class BaxterEnv(gym.Env):
    """ Baxter robot arm Environment"
    The goal of the robotic arm is to push the button on the table
    """

    metadata = {'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50}

    def __init__(self,
                 action_repeat=1,
                 renders=False,
                 is_discrete=True,
                 name="kuka_button_gym",
                 state_dim=3):
        self.range = 1000  # +/- value the randomly select number can be between
        self.bounds = 2000  # Action space bounds

        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]))
        self.observation_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.max_actions = 5
        self.observation = 0
        self.seed()
        self.reset()

        self._timestep = 1. / 240.
        #self._urdf_root = urdf_root
        self._action_repeat = action_repeat
        self._observation = []
        self._env_step_counter = 0
        self.contact_points = 0
        self.terminated = False
        self.state_dim = state_dim
        self.use_srl = state_dim > 0
        self.cuda = th.cuda.is_available()
        self.saver = EpisodeSaver(name, MAX_DISTANCE, state_dim, relative_pos=False)
        # SRL model
        self.srl_model = SRLCustomCNN(state_dim, self.cuda, noise_std=NOISE_STD)
        if self.cuda:
            self.srl_model.cuda()

        #loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)
        # Initialize button position
        x_pos = 0.5 + 0.0 * np.random.uniform(-1, 1)
        y_pos = 0 + 0.0 * np.random.uniform(-1, 1)
        #TODO: Get position of the button     #self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos = np.array([x_pos, y_pos, Z_TABLE])
        # Initialize Baxter effector by connecting to the Gym bridge ROS node:   TODO: can we reuse any URDF root path?
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://{}:{}".format(HOSTNAME, SERVER_PORT))

        print("Waiting for server...")
        msg = self.socket.recv_json()
        print("Connected to server")
        # Start simulation with first observation
        self._env_step_counter = 0
        self.times = []
        self.action = [0, 0, 0]
        cv2.imshow("Image", np.zeros((10, 10, 3), dtype=np.uint8))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.action = action  # For saver
        #if self._is_discrete: # TODO having a program parameter that sets it in order to avoid a check in each action performed
        #p.setJointMotorControl2(self.button_uid, BUTTON_GLIDER_IDX, controlMode=p.POSITION_CONTROL, targetPosition=0.1)
        self._env_step_counter += 1
        # TODO self._observation =
        if self._renders:
            time.sleep(self._timestep)
        if action in action_dict.keys():
            action = action_dict[action]
        elif action == EXIT_KEYS:
            self._termination()
        else:
            print("Unknown action: {}".format(action))
            action = [0, 0, 0]

        start_time = time.time()
        self.socket.send_json({"command": "action", "action": action})

        # Receive state data (position, etc)
        state_data = self.socket.recv_json()
        print('state: '.format(state_data))

        # Receive a camera image from the server
        img = recvMatrix(self.socket)
        cv2.imshow("Image", img)

        self.times.append(time.time() - start_time)
        done = self._termination()
        reward = self._reward()
        self.saver.step(self._observation, self.action, reward, done, self.getArmPos())

        if self.use_srl:
            return self.getState(self._observation), reward, done, {}
        return np.array(self._observation), reward, done, {}

    def getArmPos():
        state_data = self.socket.recv_json()
        start_time = time.time()
        self.socket.send_json({"command": "action", "action": action})
        # Receive state data (position, etc)
        state_data = self.socket.recv_json()
        print('state: '.format(state_data))
        # TODO: extract arm position
        return arm_pos

    def _get_reward(self): #get_reward(self):
        """
        Agent is rewarded for pushing the button and reward is discounted if the
        arm goes outside the bounding sphere surrounding the button.
        """
        current_state = self._env.getState()
        # Compute distance from Baxter left arm to the button_pos
        gripper_pos = None#p.getLinkState(self._kuka.kuka_uid, self._kuka.kuka_end_effector_index)[0]
        distance = np.linalg.norm(self.button_pos - gripper_pos, 2)
        print('Distance '.format(distance))

        if distance > MAX_DISTANCE:
            reward = -1
        elif distance < THRESHOLD_DIST_TO_CONSIDER_BUTTON_TOUCHED:
            reward = 1
        else:
            reward = 0

        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION:
            self.terminated = True

        return reward

    def reset(self):
        self.terminated = False
        self.button_pos = (0,0,0)#self.np_random.uniform(-self.range, self.range)
        self.observation = 0
        self.got_reward = False
        self.n_contacts = 0

        return np.array(self.observation) # return super(BaxterEnv, self)._reset()

    def _termination(self):
        if self.terminated or self._env_step_counter > MAX_STEPS:
            return True
        return False

    # def __del__(self): # TODO: why it doesnt recognize the self member vars? is this the wrong method?
    #     #self._env.step()
    #     self.socket.send_json({"command": "exit"})
    #     cv2.destroyAllWindows()
    #     # TODO necessary? quit gazebo, as in self.env.act(hfo_py.QUIT)
    #     # if self.socket is not None:
    #     #     os.kill(self.socket.pid, signal.SIGKILL)
    #     print("baxter_env client exiting...")
    #     print("{:.2f} FPS".format(len(self.times) / np.sum(self.times)))
    #     self.socket.close()

    # OPTIONAL / MORE SPECIALIZED METHODS
    def getState(self, obs):
        obs = preprocessImage(obs)
        # Create 4D Tensor
        obs = obs.reshape(1, *obs.shape)
        # Channel first
        obs = np.transpose(obs, (0, 3, 2, 1))
        obs = Variable(th.from_numpy(obs), volatile=True)
        if self.cuda:
            obs = obs.cuda()
        self._state = self.srl_model(obs)
        if self.cuda:
            self._state = self._state.cpu()
        self._state = self._state.data.numpy()
        return self._state

    #def _render(self, mode='human', close=False):
