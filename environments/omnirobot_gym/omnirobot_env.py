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
import yaml
from scipy.spatial.transform import Rotation as R

from environments.srl_env import SRLGymEnv
from real_robots.constants import *
from real_robots.omnirobot_utils.utils import RingBox, PosTransformer
from state_representation.episode_saver import EpisodeSaver

if USING_OMNIROBOT_SIMULATOR:
    from real_robots.omnirobot_simulator_server import OmniRobotSimulatorSocket

    def recvMatrix(socket):
        return socket.recv_image()

else:
    from real_robots.utils import recvMatrix

RENDER_HEIGHT = 224
RENDER_WIDTH = 224
RELATIVE_POS = True
N_CONTACTS_BEFORE_TERMINATION = 15 #10

DELTA_POS = 0.1  # DELTA_POS for continuous actions
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
    :param save_path: (str) name of the folder where recorded data will be stored
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param env_rank: (int) the number ID of the environment
    :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
    """

    def __init__(self, renders=False, name="Omnirobot", is_discrete=True, save_path='srl_zoo/data/', state_dim=-1,
                 learn_states=False, srl_model="raw_pixels", record_data=False, action_repeat=1, random_target=True,
                 shape_reward=False, simple_continual_target=False, circular_continual_move=False,
                 escape_continual_move=False, square_continual_move=False, eight_continual_move=False,
                 chasing_continual_move=False, short_episodes=False,
                 state_init_override=None, env_rank=0, srl_pipe=None, **_):

        super(OmniRobotEnv, self).__init__(srl_model=srl_model,
                                           relative_pos=RELATIVE_POS,
                                           env_rank=env_rank,
                                           srl_pipe=srl_pipe)
        if action_repeat != 1:
            raise NotImplementedError
        self.server_port = SERVER_PORT + env_rank
        self.n_contacts = 0
        use_ground_truth = srl_model == 'ground_truth'
        use_srl = srl_model != 'raw_pixels'
        self.use_srl = use_srl or use_ground_truth
        self.use_ground_truth = use_ground_truth
        self.use_joints = False
        self.relative_pos = RELATIVE_POS and (not escape_continual_move)
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
        self._random_target = random_target
        self.simple_continual_target = simple_continual_target
        self.circular_continual_move = circular_continual_move
        self.square_continual_move = square_continual_move
        self.eight_continual_move = eight_continual_move
        self.chasing_continual_move = chasing_continual_move
        self.escape_continual_move = escape_continual_move
        self.short_episodes = short_episodes

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            action_dim = 2
            self.action_space = RingBox(positive_low=ACTION_POSITIVE_LOW, positive_high=ACTION_POSITIVE_HIGH,
                                        negative_low=ACTION_NEGATIVE_LOW, negative_high=ACTION_NEGATIVE_HIGH,
                                        shape=np.array([action_dim]), dtype=np.float32)
        # SRL model
        if self.use_srl:
            if use_ground_truth:
                self.state_dim = self.getGroundTruthDim()
            self.dtype = np.float32
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=self.dtype)
        else:
            self.dtype = np.uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=(RENDER_WIDTH, RENDER_HEIGHT, 3),
                                                dtype=self.dtype)

        if record_data:
            print("Recording data...")
            self.saver = EpisodeSaver(name, 0, self.state_dim, globals_=getGlobals(),
                                      relative_pos=RELATIVE_POS,
                                      learn_states=learn_states, path=save_path)

        if USING_OMNIROBOT_SIMULATOR:
            self.socket = OmniRobotSimulatorSocket(simple_continual_target=simple_continual_target,
                                                   circular_continual_move=circular_continual_move,
                                                   square_continual_move=square_continual_move,
                                                   eight_continual_move=eight_continual_move,
                                                   chasing_continual_move=chasing_continual_move,
                                                   escape_continual_move=escape_continual_move,
                                                   output_size=[RENDER_WIDTH, RENDER_HEIGHT],
                                                   random_target=self._random_target,
                                                   state_init_override=state_init_override)
        else:
            # Initialize Baxter effector by connecting to the Gym bridge ROS node:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.connect(
                "tcp://{}:{}".format(HOSTNAME, self.server_port))

            # note: if takes too long, run first client, then server
            print("Waiting for server connection at port {}...".format(
                self.server_port))

            # hide the output of server
            msg = self.socket.recv_json()
            print("Connected to server on port {} (received message: {})".format(
                self.server_port, msg))

        self.action = [0, 0]
        self.reward = 0
        self.robot_pos = np.array([0, 0])

        # Initialize the state
        if self._renders:
            self.image_plot = None

    def actionPolicyTowardTarget(self):
        """
        :return: (int) action
        """
        if abs(self.robot_pos[0] - self.target_pos[0]) > abs(self.robot_pos[1] - self.target_pos[1]):

            if self._is_discrete:
                return int(Move.FORWARD) if self.robot_pos[0] < self.target_pos[0] else int(Move.BACKWARD)
                # forward                                        # backward
            else:
                return DELTA_POS if self.robot_pos[0] < self.target_pos[0] else -DELTA_POS
        else:
            if self._is_discrete:
                # left                                          # right
                return int(Move.LEFT) if self.robot_pos[1] < self.target_pos[1] else int(Move.RIGHT)
            else:
                return DELTA_POS if self.robot_pos[1] < self.target_pos[1] else -DELTA_POS

    def actionPolicyAwayTarget(self):
        """
        :return: (int) action
        """
        if abs(self.robot_pos[0] - self.target_pos[0]) > abs(self.robot_pos[1] - self.target_pos[1]):

            if self._is_discrete:
                return int(Move.BACKWARD) if self.robot_pos[0] < self.target_pos[0] else int(Move.FORWARD)
                # forward                                        # backward
            else:
                return -DELTA_POS if self.robot_pos[0] < self.target_pos[0] else +DELTA_POS
        else:
            if self._is_discrete:
                # left                                          # right
                return int(Move.RIGHT) if self.robot_pos[1] < self.target_pos[1] else int(Move.LEFT)
            else:
                return -DELTA_POS if self.robot_pos[1] < self.target_pos[1] else +DELTA_POS

    def step(self, action, generated_observation=None, action_proba=None, action_grid_walker=None):
        """
        :param :action: (int)
        :param generated_observation:
        :param action_proba:
        :param action_grid_walker: # Whether or not we want to override the action with the one from a grid walker
        :return: (tensor (np.ndarray)) observation, int reward, bool done, dict extras)
        """
        if action_grid_walker is None:
            action_to_step = action
            action_from_teacher = None
        else:
            action_to_step = action_grid_walker
            action_from_teacher = action
            assert self.action_space.contains(action_from_teacher)

        if not self._is_discrete:
            action_to_step = np.array(action_to_step)
        assert self.action_space.contains(action_to_step)

        # Convert int action to action in (x,y,z) space
        # serialize the action
        if isinstance(action_to_step, np.ndarray):
            self.action = action_to_step.tolist()
        elif hasattr(action_to_step, 'dtype'):  # convert numpy type to python type
            self.action = action.item()
        else:
            self.action = action_to_step

        self._env_step_counter += 1

        # Send the action to the server
        self.socket.send_json(
            {"command": "action", "action": self.action, "is_discrete": self._is_discrete,
             "step_counter": self._env_step_counter})

        # Receive state data (position, etc), important to update state related values
        self.getEnvState()

        #  Receive a camera image from the server
        self.observation = self.getObservation() if generated_observation is None else generated_observation * 255
        done = self._hasEpisodeTerminated()

        self.render()

        if self.saver is not None:
            # Dynamic environment
            if self.chasing_continual_move or self.escape_continual_move:
                self.saver.step(self.observation, action_from_teacher if action_grid_walker is not None else
                                action_to_step,
                                self.reward, done, self.getGroundTruth(),
                                action_proba=action_proba, target_pos=self.getTargetPos())
            else:
                self.saver.step(self.observation, action_from_teacher if action_grid_walker is not None else
                                action_to_step,
                                self.reward, done, self.getGroundTruth(),
                                action_proba=action_proba)
        old_observation = self.getObservation()
        if self.use_srl:
            return self.getSRLState(self.observation
                                    if generated_observation is None else old_observation), self.reward, done, {}
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
        self.observation = cv2.resize(
            self.observation, (RENDER_WIDTH, RENDER_HEIGHT), interpolation=cv2.INTER_AREA)
        return self.observation

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        return self.target_pos

    @staticmethod
    def getGroundTruthDim():
        """
        The convergence is slow for escape task with dimension 2
        :return: (int)
        """
        # if(not self.escape_continual_move):
        #     return 2
        # else:
        #     #The position of the robot, target
        #     return 4
        return 2

    def getGroundTruth(self):
        """
        Alias for getRobotPos for compatibility between envs
        :return: (numpy array)
        """
        #
        # if(not self.escape_continual_move):
        #     return np.array(self.getRobotPos())
        # else:
        #     return np.append(self.getRobotPos(),self.getTargetPos())
        return np.array(self.getRobotPos())

    def getRobotPos(self):
        """
        :return: ([float])->  np.ndarray Position (x, y, z) of Baxter left gripper
        """
        return self.robot_pos

    def reset(self, generated_observation=None, state_override=None):
        """
        Reset the environment
        :param generated_observation:
        :param state_override:
        :return: (numpy ndarray) first observation of the env
        """
        self.episode_terminated = False
        # Step count since episode start
        self._env_step_counter = 0
        # set n contact count
        self.n_contacts = 0
        self.socket.send_json({"command": "reset", "step_counter": self._env_step_counter})
        # Update state related variables, important step to get both data and
        # metadata that allow reading the observation image
        self.getEnvState()
        self.robot_pos = np.array([0, 0]) if state_override is None else state_override
        self.observation = self.getObservation() if generated_observation is None else generated_observation * 255
        if self.saver is not None:
            self.saver.reset(self.observation,
                             self.getTargetPos(), self.getGroundTruth())
        old_observation = self.getObservation()

        if self.use_srl:
            return self.getSRLState(self.observation if generated_observation is None else old_observation)
        else:
            return self.observation

    def _hasEpisodeTerminated(self):
        """
        Returns True if the episode is over and False otherwise
        """
        if (self.episode_terminated or self._env_step_counter > MAX_STEPS) or \
                (self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION and self.short_episodes and
                 self.simple_continual_target) or \
                (self._env_step_counter > MAX_STEPS_CIRCULAR_TASK_SHORT_EPISODES and self.short_episodes and
                 self.circular_continual_move):
            return True

        if np.abs(self.reward - REWARD_TARGET_REACH) < 0.000001:  # reach the target
            self.n_contacts += 1
        else:
            self.n_contacts += 0
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
        if self._renders:
            if mode != "rgb_array":
                print('render in human mode not yet supported')
                return np.array([])

            plt.ion()  # needed for interactive update
            if self.image_plot is None:
                plt.figure('Omnirobot RL')
                self.initVisualizeBoundary()
                self.visualizeBoundary()
                self.image_plot = plt.imshow(self.observation_with_boundary, cmap='gray')
                self.image_plot.axes.grid(False)

            else:
                self.visualizeBoundary()
                self.image_plot.set_data(self.observation_with_boundary)
            plt.draw()
            # Wait a bit, so that plot is visible
            plt.pause(0.0001)
        return self.observation

    def initVisualizeBoundary(self):
        with open(CAMERA_INFO_PATH, 'r') as stream:
            try:
                contents = yaml.load(stream)
                camera_matrix = np.array(contents['camera_matrix']['data']).reshape((3, 3))
                distortion_coefficients = np.array(contents['distortion_coefficients']['data']).reshape((1, 5))
            except yaml.YAMLError as exc:
                print(exc)

        # camera installation info
        r = R.from_euler('xyz', CAMERA_ROT_EULER_COORD_GROUND, degrees=True)
        camera_rot_mat_coord_ground = r.as_dcm()

        pos_transformer = PosTransformer(camera_matrix, distortion_coefficients, CAMERA_POS_COORD_GROUND,
                                         camera_rot_mat_coord_ground)

        self.boundary_coner_pixel_pos = np.zeros((2, 4))
        # assume that image is undistorted
        self.boundary_coner_pixel_pos[:, 0] = \
            pos_transformer.phyPosGround2PixelPos([MIN_X, MIN_Y], return_distort_image_pos=False).squeeze()
        self.boundary_coner_pixel_pos[:, 1] = \
            pos_transformer.phyPosGround2PixelPos([MAX_X, MIN_Y], return_distort_image_pos=False).squeeze()
        self.boundary_coner_pixel_pos[:, 2] = \
            pos_transformer.phyPosGround2PixelPos([MAX_X, MAX_Y], return_distort_image_pos=False).squeeze()
        self.boundary_coner_pixel_pos[:, 3] = \
            pos_transformer.phyPosGround2PixelPos([MIN_X, MAX_Y], return_distort_image_pos=False).squeeze()

        # transform the corresponding points into cropped image
        self.boundary_coner_pixel_pos = self.boundary_coner_pixel_pos - (np.array(ORIGIN_SIZE) -
                                                                         np.array(CROPPED_SIZE)).reshape(2, 1) / 2.0

        # transform the corresponding points into resized image (RENDER_WIDHT, RENDER_HEIGHT)
        self.boundary_coner_pixel_pos[0, :] *= RENDER_WIDTH/CROPPED_SIZE[0]
        self.boundary_coner_pixel_pos[1, :] *= RENDER_HEIGHT/CROPPED_SIZE[1]

        self.boundary_coner_pixel_pos = np.around(self.boundary_coner_pixel_pos).astype(np.int)

        # Create square for vizu of objective in continual square task
        if self.square_continual_move:

            self.boundary_coner_pixel_pos_continual = np.zeros((2, 4))
            # assume that image is undistorted
            self.boundary_coner_pixel_pos_continual[:, 0] = \
                pos_transformer.phyPosGround2PixelPos([-RADIUS, -RADIUS], return_distort_image_pos=False).squeeze()
            self.boundary_coner_pixel_pos_continual[:, 1] = \
                pos_transformer.phyPosGround2PixelPos([RADIUS, -RADIUS],  return_distort_image_pos=False).squeeze()
            self.boundary_coner_pixel_pos_continual[:, 2] = \
                pos_transformer.phyPosGround2PixelPos([RADIUS, RADIUS], return_distort_image_pos=False).squeeze()
            self.boundary_coner_pixel_pos_continual[:, 3] = \
                pos_transformer.phyPosGround2PixelPos([-RADIUS, RADIUS], return_distort_image_pos=False).squeeze()

            # transform the corresponding points into cropped image
            self.boundary_coner_pixel_pos_continual = self.boundary_coner_pixel_pos_continual - (
                        np.array(ORIGIN_SIZE) - np.array(CROPPED_SIZE)).reshape(2, 1) / 2.0

            # transform the corresponding points into resized image (RENDER_WIDHT, RENDER_HEIGHT)
            self.boundary_coner_pixel_pos_continual[0, :] *= RENDER_WIDTH / CROPPED_SIZE[0]
            self.boundary_coner_pixel_pos_continual[1, :] *= RENDER_HEIGHT / CROPPED_SIZE[1]

            self.boundary_coner_pixel_pos_continual = np.around(self.boundary_coner_pixel_pos_continual).astype(np.int)

        elif self.circular_continual_move:
            self.center_coordinates = \
                pos_transformer.phyPosGround2PixelPos([0, 0], return_distort_image_pos=False).squeeze()
            self.center_coordinates = self.center_coordinates - (
                np.array(ORIGIN_SIZE) - np.array(CROPPED_SIZE)) / 2.0
            # transform the corresponding points into resized image (RENDER_WIDHT, RENDER_HEIGHT)
            self.center_coordinates[0] *= RENDER_WIDTH / CROPPED_SIZE[0]
            self.center_coordinates[1] *= RENDER_HEIGHT / CROPPED_SIZE[1]

            self.center_coordinates = np.around(self.center_coordinates).astype(np.int)

            # Points to convert radisu in env space
            self.boundary_coner_pixel_pos_continual = \
                pos_transformer.phyPosGround2PixelPos([0, RADIUS], return_distort_image_pos=False).squeeze()

            # transform the corresponding points into cropped image
            self.boundary_coner_pixel_pos_continual = self.boundary_coner_pixel_pos_continual - (
                        np.array(ORIGIN_SIZE) - np.array(CROPPED_SIZE)) / 2.0

            # transform the corresponding points into resized image (RENDER_WIDHT, RENDER_HEIGHT)
            self.boundary_coner_pixel_pos_continual[0] *= RENDER_WIDTH / CROPPED_SIZE[0]
            self.boundary_coner_pixel_pos_continual[1] *= RENDER_HEIGHT / CROPPED_SIZE[1]

            self.boundary_coner_pixel_pos_continual = np.around(self.boundary_coner_pixel_pos_continual).astype(np.int)

    def visualizeBoundary(self):
        """
        visualize the unvisible boundary, should call initVisualizeBoundary first
        """
        self.observation_with_boundary = self.observation.copy()

        # Add boundary continual
        if self.square_continual_move:
            for idx in range(4):
                idx_next = idx + 1
                cv2.line(self.observation_with_boundary, tuple(self.boundary_coner_pixel_pos_continual[:, idx]),
                         tuple(self.boundary_coner_pixel_pos_continual[:, idx_next % 4]), (0, 0, 200), 2)

        elif self.circular_continual_move:
            radius_converted = np.linalg.norm(self.center_coordinates - self.boundary_coner_pixel_pos_continual)
            cv2.circle(self.observation_with_boundary, tuple(self.center_coordinates), np.float32(radius_converted),
                       (0, 0, 200), 2)

        # Add boundary of env
        for idx in range(4):
            idx_next = idx + 1
            cv2.line(self.observation_with_boundary, tuple(self.boundary_coner_pixel_pos[:, idx]),
                     tuple(self.boundary_coner_pixel_pos[:, idx_next % 4]), (200, 0, 0), 3)
