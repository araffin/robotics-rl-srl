from __future__ import division, print_function, absolute_import

from real_robots.constants import *


class OmnirobotManagerBase(object):
    def __init__(self, simple_continual_target=False, circular_continual_move=False, square_continual_move=False,
                 lambda_c=1.0):
        """
        This class is the basic class for omnirobot server, and omnirobot simulator's server.
        This class takes omnirobot position at instant t, and takes the action at instant t,
        to determinate the position it should go at instant t+1, and the immediate reward it can get at instant t
        """
        super(OmnirobotManagerBase, self).__init__()
        self.second_cam_topic = SECOND_CAM_TOPIC
        self.episode_idx = 0
        self.simple_continual_target = simple_continual_target
        self.circular_continual_move = circular_continual_move
        self.square_continual_move =  square_continual_move
        self.lambda_c = lambda_c

        # the abstract object for robot,
        # can be the real robot (Omnirobot class)
        #  or the robot simulator (OmniRobotEnvRender class)
        self.robot = None

    def rightAction(self):
        """
        Let robot execute right action, and checking the boundary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[1] - STEP_DISTANCE > MIN_Y:
            self.robot.right()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def leftAction(self):
        """
        Let robot execute left action, and checking the boundary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[1] + STEP_DISTANCE < MAX_Y:
            self.robot.left()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def forwardAction(self):
        """
        Let robot execute forward action, and checking the boundary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[0] + STEP_DISTANCE < MAX_X:
            self.robot.forward()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def backwardAction(self):
        """
        Let robot execute backward action, and checking the boundary
        :return has_bumped: (bool) 
        """
        if self.robot.robot_pos[0] - STEP_DISTANCE > MIN_X:
            self.robot.backward()
            has_bumped = False
        else:
            has_bumped = True
        return has_bumped

    def moveContinousAction(self, msg):
        """
        Let robot execute continous action, and checking the boundary
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

        if self.circular_continual_move or self.square_continual_move:
            step_counter = msg.get("step_counter", None)
            assert step_counter is not None

            self.robot.appendToHistory(self.robot.robot_pos)

            ord = None
            if self.square_continual_move:
                ord = np.inf
            self.reward = 1 - 100 * (np.linalg.norm(self.robot.robot_pos, ord=ord) - RADIUS) ** 2
            print(self.reward, 'REWARD SQUARE/CIRCLE')

            if step_counter < self.robot.getHistorySize():
                pass
            else:
                self.robot.popOfHistory()
                self.reward += \
                    self.lambda_c * np.linalg.norm(self.robot.robot_pos - self.robot.robot_pos_past_k_steps[0])
                print(self.lambda_c * np.linalg.norm(self.robot.robot_pos - self.robot.robot_pos_past_k_steps[0]), 'ADDITIONAL REWARD')

        else:
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
