from __future__ import division, print_function, absolute_import
from gym import spaces
from gym import logger
import gym
from .constants import *


class OmnirobotManagerBase(object):
    def __init__(self, second_cam_topic=None):
        """
        This class is the basic class for omnirobot server, and omnirobot simulator's server.
        This class takes omnirobot position at instant t, and takes the action at instant t,
        to determinate the position it should go at instant t+1, and the immediate reward it can get at instant t
        """
        super(OmnirobotManagerBase, self).__init__()
        self.second_cam_topic = SECOND_CAM_TOPIC
        self.episode_idx = 0

        # the abstract object for robot,
        # can be the real robot (Omnirobot class)
        #  or the robot simulator (OmniRobotEnvRender class)
        self.robot = None

    def rightAction(self):
        """
        Let robot excute right action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[1] > MIN_Y:
            self.robot.right()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def leftAction(self):
        """
        Let robot excute left action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[1] < MAX_Y:
            self.robot.left()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def forwardAction(self):
        """
        Let robot excute forward action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[0] < MAX_X:
            self.robot.forward()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def backwardAction(self):
        """
        Let robot excute backward action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[0] > MIN_X:
            self.robot.backward()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def moveContinousAction(self, msg):
        """
        Let robot excute continous action, and checking the boudary
        :return has_bumped: (bool) 
        """
        if MIN_X < self.robot.robot_pos[0] + msg['action'][0] < MAX_X and \
                MIN_Y < self.robot.robot_pos[1] + msg['action'][1] < MAX_Y:
            self.robot.moveContinous(msg['action'])
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def sampleRobotInitalPosition(self):
        random_init_x = np.random.random_sample() * (INIT_MAX_X - INIT_MIN_X) + INIT_MIN_X
        random_init_y = np.random.random_sample() * (INIT_MAX_Y - INIT_MIN_Y) + INIT_MIN_Y
        return [random_init_x, random_init_y]
    
    def resetEpisode(self):
        """
        Give the correct sequance of commands to the robot 
        to rest environment between the different episodes
        """
        if self.second_cam_topic is not None:
            assert NotImplementedError
        # Env reset
        random_init_position = self.sampleRobotInitalPosition()
        self.robot.setRobotCmd(random_init_position[0], random_init_position[1], 0)

    def processMsg(self, msg):
        """
        Using this steps' msg command the determinate the correct position that the robot should be at next step,
        and to determinate the reward of this step.
        This function also takes care of the environment's reset.
        :param msg: (dict)
        """
        command = msg.get('command', '')
        if command == 'reset':
            action = None
            self.episode_idx += 1
            self.resetEpisode()

        elif command == 'action':
            if msg.get('is_discrete', False):
                action = Move(msg['action'])
            else:
                action = 'Continuous'

        elif command == "exit":
            print("recive exit signal, quit...")
            exit(0)
        else:
            raise ValueError("Unknown command: {}".format(msg))

        has_bumped = False
        # We are always facing North
        if action == Move.FORWARD:
            has_bumped = self.forwardAction()
        elif action == Move.STOP:
            pass
        elif action == Move.RIGHT:
            has_bumped = self.rightAction()
        elif action == Move.LEFT:
            has_bumped = self.leftAction()
        elif action == Move.BACKWARD:
            has_bumped = self.backwardAction()
        elif action == 'Continuous':
            has_bumped = self.moveContinousAction(msg)
        elif action == None:
            pass
        else:
            print("Unsupported action: ", action)

        # Determinate the reward for this step
        
        # Consider that we reached the target if we are close enough
        # we detect that computing the difference in area between TARGET_INITIAL_AREA
        # current detected area of the target
        if np.linalg.norm(np.array(self.robot.robot_pos) - np.array(self.robot.target_pos)) \
                < DIST_TO_TARGET_THRESHOLD:
            self.reward = REWARD_TARGET_REACH
        elif has_bumped:
            self.reward = REWARD_BUMP_WALL
        else:
            self.reward = REWARD_NOTHING
<<<<<<< HEAD



class RingBox(gym.Space):
    """
    A ring box in R^n.
    I.e., each coordinate is bounded.
    there are minimum constrains (absolute) on all of the coordinates 
    """
    def __init__(self, positive_low=None, positive_high=None, negative_low=None, negative_high=None, shape=None, dtype=None):
        """
        for each coordinate
        the value will be sampled from [positive_low, positive_hight] or [negative_low, negative_high]        
        """

        if shape is None:
            assert positive_low.shape == positive_high.shape == negative_low.shape == negative_high.shape
            shape = positive_low.shape
        else:
            assert np.isscalar(positive_low) and np.isscalar(positive_high) and np.isscalar(negative_low) and np.isscalar(negative_high)
            positive_low = positive_low + np.zeros(shape)
            positive_high = positive_high + np.zeros(shape)
            negative_low = negative_low + np.zeros(shape)
            negative_high = negative_high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (positive_high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            logger.warn("Ring Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))
        self.positive_low = positive_low.astype(dtype)
        self.positive_high = positive_high.astype(dtype)
        self.negative_low = negative_low.astype(dtype)
        self.negative_high = negative_high.astype(dtype)
        self.length_positive = self.positive_high - self.positive_low 
        self.length_negative = self.negative_high - self.negative_low
        super(RingBox, self).__init__(shape, dtype)
        self.np_random = np.random.RandomState()
    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        length_positive = self.length_positive if self.dtype.kind == 'f' else self.length_positive.astype('int64') + 1
        origin_sample = self.np_random.uniform(low=-self.length_negative, high=length_positive, size=self.negative_high.shape).astype(self.dtype)
        origin_sample = origin_sample + self.positive_low * (origin_sample >= 0) + self.negative_high * (origin_sample <= 0)
        return origin_sample

    def contains(self, x):
        return x.shape == self.shape and np.logical_or(np.logical_and(x >= self.positive_low, x <= self.positive_high),
                np.logical_and(x <= self.negative_high,  x >= self.negative_low)).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.positive_low, other.positive_low) and np.allclose(self.positive_high, other.positive_high) \
            and np.allclose(self.negative_low, other.negative_low) and np.allclose(self.negative_high, other.negative_high)
=======
>>>>>>> 3a82827e9d0531b1dd83632eda83c1e0351ac058
