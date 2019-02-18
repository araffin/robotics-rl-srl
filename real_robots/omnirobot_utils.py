from __future__ import division, print_function, absolute_import
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
        self.robot = None # the abstract object for robot, 
                          # can be the real robot (Omnirobot class)
                          #  or the robot simulator (OmniRobotEnvRender class)
    
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
        else:
            print("Unsupported action")

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
            self.reward = REWARD_BUMP_WALL