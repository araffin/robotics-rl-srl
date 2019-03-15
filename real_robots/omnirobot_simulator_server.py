from __future__ import division, print_function, absolute_import

from multiprocessing import Process, Pipe
import yaml
import cv2
# Konwn issue: - No module named 'scipy.spatial.transform', To resolve, try pip3 install scipy==1.2
from scipy.spatial.transform import Rotation as R

from real_robots.constants import *
from real_robots.omnirobot_utils.marker_finder import MakerFinder
from real_robots.omnirobot_utils.marker_render import MarkerRender
from real_robots.omnirobot_utils.omnirobot_manager_base import OmnirobotManagerBase
from real_robots.omnirobot_utils.utils import PosTransformer

assert USING_OMNIROBOT_SIMULATOR, "Please set USING_OMNIROBOT_SIMULATOR to True in real_robots/constants.py"
NOISE_VAR_ROBOT_POS = 0.01  # meter
NOISE_VAR_ROBOT_YAW = np.pi/180 * 2.5  # 5 Deg
NOISE_VAR_TARGET_PIXEL = 2  # pixel noise on target marker
NOISE_VAR_ROBOT_PIXEL = 2
NOISE_VAR_ENVIRONMENT = 0.03  # pixel noise of final image on LAB space
NOISE_VAR_ROBOT_SIZE_PROPOTION = 0.05  # noise of robot size propotion
NOISE_VAR_TARGET_SIZE_PROPOTION = 0.05


class OmniRobotEnvRender():
    def __init__(self, init_x, init_y, init_yaw, origin_size, cropped_size,
                 back_ground_path, camera_info_path,
                 robot_marker_path, robot_marker_margin, target_marker_path, target_marker_margin,
                 robot_marker_code, target_marker_code,
                 robot_marker_length, target_marker_length, output_size, history_size=10, **_):
        """
        Class for rendering Omnirobot environment
        :param init_x: (float) initial x position of robot
        :param init_y: (float) initial y position of robot
        :param init_yaw: (float) initial yaw position of robot
        :param origin_size: (list of int) original camera's size (eg. [640,480]), the camera matrix should be corresponding to this size
        :param cropped_size: (list of int) cropped image's size (eg. [480,480])
        :param back_ground_path: (str) back ground image's path, the image should be undistorted.
        :param camera_info_path: (str) camera info file's path (containing camera matrix)
        :param robot_marker_path: (str) robot maker's path, the marker should have a margin with several pixels 
        :param robot_marker_margin: (list of int) marker's margin (eg. [3,3,3,3])
        :param target_marker_path: (str) target maker's path, the marker should have a margin with several pixels 
        :param target_marker_margin: (list of int) marker's margin (eg. [3,3,3,3])
        :param robot_marker_code: (currently not supported, should be "None" by default) (numpy ndarray) optional, the code of robot marker, only used for detecting position directly from the image.
        :param target_marker_code: (currently not supported, should be "None" by default) (numpy ndarray) optional, the code of target marker, only used for detecting position directly from the image.
        :param robot_marker_length: (float) the physical length of the marker (in meter)
        :param target_marker_length: (float) the physical length of the marker (in meter)
        :param output_size: (list of int) the output image's size (eg. [224,224])
        :param **_: other input params not used, so they are dropped
        """
        super(OmniRobotEnvRender, self).__init__()

        self.output_size = output_size

        # store the size of robot marker
        self.robot_marker_size_proprotion = 1.0

        # Initialize the direction
        self.init_pos = [init_x, init_y]
        self.init_yaw = init_yaw

        # OmniRobot's real position on the grid
        self.robot_pos = np.float32([0, 0])
        self.robot_yaw = 0  # in rad

        self.history_size = history_size
        self.robot_pos_past_k_steps = []

        # Last velocity command, used for simulating the controlling of velocity directly
        self.last_linear_velocity_cmd = np.float32(
            [0, 0])  # in m/s, in robot local frame
        self.last_rot_velocity_cmd = 0  # in rad/s

        # last wheel speeds command, used for simulating the controlling of wheel speed directly
        # [left_speed, front_speed, right_speed]
        self.last_wheel_speeds_cmd = np.float32([0, 0, 0])

        # OmniRobot's position command on the grid
        self.robot_pos_cmd = np.float32(self.init_pos[:])
        self.robot_yaw_cmd = self.init_yaw

        # Target's set position on the grid
        self.target_pos_cmd = np.float32([0, 0])
        self.target_yaw_cmd = 0.0

        # Target's real position on the grid
        self.target_pos = np.float32([0, 0])
        self.target_yaw = 0

        # status of moving
        self.move_finished = False
        self.target_pos_changed = False

        # Distance for each step
        self.step_distance = STEP_DISTANCE

        with open(camera_info_path, 'r') as stream:
            try:
                contents = yaml.load(stream)
                camera_matrix = np.array(contents['camera_matrix']['data'])
                self.origin_size = np.array(
                    [contents['image_height'], contents['image_width']])
                self.camera_matrix = np.reshape(camera_matrix, (3, 3))
                self.dist_coeffs = np.array(
                    contents["distortion_coefficients"]["data"]).reshape((1, 5))
            except yaml.YAMLError as exc:
                print(exc)
        self.cropped_size = [np.min(self.origin_size), np.min(
            self.origin_size)]  # size after being cropped

        # restore the image before being cropped
        self.bg_img = np.zeros([*self.origin_size, 3], np.uint8)

        self.cropped_margin = (self.origin_size - self.cropped_size)/2.0
        self.cropped_range = np.array([self.cropped_margin[0], self.cropped_margin[0]+self.cropped_size[0],
                                       self.cropped_margin[1],
                                       self.cropped_margin[1]+self.cropped_size[1]]).astype(np.int)

        back_ground_img = cv2.imread(back_ground_path)
        if(back_ground_img.shape[0:2] != self.cropped_size):
            print("input back ground image's size: ", back_ground_img.shape)
            print("resize to ", self.cropped_size)
            self.bg_img[self.cropped_range[0]:self.cropped_range[1], self.cropped_range[2]:self.cropped_range[3], :] \
                = cv2.resize(back_ground_img, tuple(self.cropped_size))  # background image
        else:
            self.bg_img[self.cropped_range[0]:self.cropped_range[1], self.cropped_range[2]:self.cropped_range[3], :] \
                = back_ground_img  # background image

        self.bg_img = cv2.undistort(
            self.bg_img, self.camera_matrix, self.dist_coeffs)
        # Currently cannot find a solution to re-distort a image...

        self.target_bg_img = self.bg_img  # background image with target.
        self.image = self.bg_img  # image with robot and target

        # camera installation info
        r = R.from_euler('xyz', CAMERA_ROT_EULER_COORD_GROUND, degrees=True)
        camera_rot_mat_coord_ground = r.as_dcm()

        self.pos_transformer = PosTransformer(self.camera_matrix, self.dist_coeffs,
                                              CAMERA_POS_COORD_GROUND, camera_rot_mat_coord_ground)

        self.target_render = MarkerRender(noise_var=NOISE_VAR_TARGET_PIXEL)
        self.robot_render = MarkerRender(noise_var=NOISE_VAR_ROBOT_PIXEL)
        self.robot_render.setMarkerImage(cv2.imread(
            robot_marker_path, cv2.IMREAD_COLOR), robot_marker_margin)
        self.target_render.setMarkerImage(cv2.imread(
            target_marker_path, cv2.IMREAD_COLOR), target_marker_margin)

        if robot_marker_code is not None and target_marker_code is not None:
            self.marker_finder = MakerFinder(camera_info_path)
            self.marker_finder.setMarkerCode(
                'robot', robot_marker_code, robot_marker_length)
            self.marker_finder.setMarkerCode(
                'target', target_marker_code, target_marker_length)

    def renderEnvLuminosityNoise(self, origin_image, noise_var=0.1, in_RGB=False, out_RGB=False):
        """
        render the different environment luminosity
        """
        # variate luminosity and color
        origin_image_LAB = cv2.cvtColor(
            origin_image, cv2.COLOR_RGB2LAB if in_RGB else cv2.COLOR_BGR2LAB, cv2.CV_32F)
        origin_image_LAB[:, :, 0] = origin_image_LAB[:,
                                                     :, 0] * (np.random.randn() * noise_var + 1.0)
        origin_image_LAB[:, :, 1] = origin_image_LAB[:,
                                                     :, 1] * (np.random.randn() * noise_var + 1.0)
        origin_image_LAB[:, :, 2] = origin_image_LAB[:,
                                                     :, 2] * (np.random.randn() * noise_var + 1.0)
        out_image = cv2.cvtColor(
            origin_image_LAB, cv2.COLOR_LAB2RGB if out_RGB else cv2.COLOR_LAB2BGR, cv2.CV_8UC3)
        return out_image

    def renderTarget(self):
        """
        render the target
        """
        self.target_bg_img = self.target_render.addMarker(self.bg_img,
                                                          self.pos_transformer.phyPosGround2PixelPos(
                                                              self.target_pos.reshape(2, 1)),
                                                          self.target_yaw, np.random.randn() * NOISE_VAR_TARGET_SIZE_PROPOTION + 1.0)

    def renderRobot(self):
        """
        render the image.
        """
        self.image = self.robot_render.addMarker(self.target_bg_img,
                                                 self.pos_transformer.phyPosGround2PixelPos(
                                                     self.robot_pos.reshape(2, 1)),
                                                 self.robot_yaw, self.robot_marker_size_proprotion)
    def getHistorySize(self):
        return self.history_size

    def appendToHistory(self, pos):
        self.robot_pos_past_k_steps.append(pos)

    def popOfHistory(self):
        self.robot_pos_past_k_steps.pop(0)

    def getCroppedImage(self):
        return self.image[self.cropped_range[0]:self.cropped_range[1], self.cropped_range[2]:self.cropped_range[3], :]

    def findMarkers(self):
        assert NotImplementedError
        # this is not tested
        tags_trans_coord_cam, tags_rot_coord_cam = self.marker_finder.getMarkerPose(
            self.image, ['robot', 'target'], True)
        if 'robot' in tags_trans_coord_cam.keys():
            self.robot_pos = self.pos_transformer.phyPosCam2PhyPosGround(
                tags_trans_coord_cam['robot'])
            tag_rot_coord_ground = np.matmul(
                self.pos_transformer.camera_2_ground_trans[0:3, 0:3], tags_rot_coord_cam['robot'])[0:3, 0:3]
            self.robot_yaw = R.from_dcm(
                tag_rot_coord_ground).as_euler('zyx', degree=False)
            print("robot_error: ". self.robot_pos - self.robot_pos_cmd)
            print("robot_yaw_error: ". self.robot_yaw - self.robot_yaw_cmd)

        if 'target' in tags_trans_coord_cam.keys():
            self.target_pos = self.pos_transformer.phyPosCam2PhyPosGround(
                tags_trans_coord_cam['target'])
            tag_rot_coord_ground = np.matmul(self.pos_transformer.camera_2_ground_trans[0:3, 0:3],
                                             tags_rot_coord_cam['target'])[0:3, 0:3]
            self.target_yaw = R.from_dcm(
                tag_rot_coord_ground).as_euler('zyx', degree=False)
            print("target_error: ", self.target_pos - self.target_pos_cmd)
            print("target_yaw_error: ", self.target_yaw - self.target_yaw_cmd)

    def setRobotCmdConstrained(self, x, y, yaw):
        self.robot_pos_cmd[0] = max(x, MIN_X)
        self.robot_pos_cmd[0] = min(x, MAX_X)

        self.robot_pos_cmd[1] = max(y, MIN_Y)
        self.robot_pos_cmd[1] = min(y, MAX_Y)
        self.robot_yaw_cmd = self.normalizeAngle(yaw)

    def setRobotCmd(self, x, y, yaw):
        self.robot_pos_cmd[0] = x
        self.robot_pos_cmd[1] = y
        self.robot_yaw_cmd = self.normalizeAngle(yaw)

        self.robot_pos = self.robot_pos_cmd + \
            np.random.randn(2) * NOISE_VAR_ROBOT_POS  # add noise
        self.robot_yaw = self.normalizeAngle(
            self.robot_yaw_cmd + np.random.randn() * NOISE_VAR_ROBOT_YAW)  # add noise

    def setTargetCmd(self, x, y, yaw):
        self.target_pos_cmd[0] = x
        self.target_pos_cmd[1] = y
        self.target_yaw_cmd = self.normalizeAngle(yaw)

        self.target_pos = self.target_pos_cmd
        self.target_yaw = self.normalizeAngle(self.target_yaw_cmd)

    def forward(self, action=None):
        """
        Move one step forward (Translation)
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0] + self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)

    def backward(self, action=None):
        """
        Move one step backward
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0] - self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)

    def left(self, action=None):
        """
        Translate to the left
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0], self.robot_pos_cmd[1] + self.step_distance, self.robot_yaw_cmd)

    def right(self, action=None):
        """
        Translate to the right
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0], self.robot_pos_cmd[1] - self.step_distance, self.robot_yaw_cmd)

    def moveContinous(self, action):
        """
        Perform a continuous displacement of dx, dy
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0] + action[0], self.robot_pos_cmd[1] + action[1], self.robot_yaw_cmd)

    def moveByVelocityCmd(self, speed_x, speed_y, speed_yaw):
        """
        simuate the robot moved by velocity command
        This function is assumed to be called at a frequency RL_CONTROL_FREQ in the simulation world

        :param speed_x: (float) linear speed along x-axis (m/s) (forward-backward), in robot local coordinate
        :param speed_y: (float) linear speed along y-axis (m/s) (left-right), in robot local coordinate
        :param speed_yaw: (float) rotation speed of robot around z-axis (rad/s), in robot local coordinate
        """
        # calculate the robot position that it should be at this moment, so it should be driven by last command
        # Assume in 1/RL_CONTROL_FREQ, the heading remains the same (not true,
        #   but should be approximately work if RL_CONTROL_FREQ is high enough)
        # translate the last velocity cmd in robot local coordiante to position cmd in gound coordiante
        cos_direction = np.cos(self.robot_yaw)
        sin_direction = np.sin(self.robot_yaw)

        ground_pos_cmd_x = self.robot_pos[0] + (self.last_linear_velocity_cmd[0] *
                                                cos_direction - self.last_linear_velocity_cmd[1] * sin_direction)/RL_CONTROL_FREQ
        ground_pos_cmd_y = self.robot_pos[1] + (self.last_linear_velocity_cmd[1] *
                                                cos_direction + self.last_linear_velocity_cmd[0] * sin_direction)/RL_CONTROL_FREQ
        ground_yaw_cmd = self.robot_yaw + self.last_rot_velocity_cmd/RL_CONTROL_FREQ
        self.setRobotCmd(ground_pos_cmd_x, ground_pos_cmd_y, ground_yaw_cmd)

        # save the command of this moment
        self.last_linear_velocity_cmd[0] = speed_x
        self.last_linear_velocity_cmd[1] = speed_y
        self.last_rot_velocity_cmd = speed_yaw

    def moveByWheelsCmd(self, left_speed, front_speed, right_speed):
        """
        simuate the robot moved by wheel speed command
        This function is assumed to be called at a frequency RL_CONTROL_FREQ in the simulation world

        :param left_speed: (float) linear speed of left wheel (meter/s)
        :param front_speed: (float) linear speed of front wheel (meter/s)
        :param right_speed: (float) linear speed of right wheel (meter/s)
        """

        # calculate the robot position by omnirobot's kinematic equations
        # Assume in 1/RL_CONTROL_FREQ, the heading remains the same (not true,
        # but should be approximately work if RL_CONTROL_FREQ is high enough)

        # translate the last wheel speeds cmd in last velocity cmd
        local_speed_x = self.last_wheel_speeds_cmd[0] / np.sqrt(3.0) \
            - self.last_wheel_speeds_cmd[2] / np.sqrt(3.0)
        local_speed_y = - self.last_wheel_speeds_cmd[1] / 1.5 + \
            self.last_wheel_speeds_cmd[0] / 3.0 + \
            self.last_wheel_speeds_cmd[2] / 3.0
        local_rot_speed = - self.last_wheel_speeds_cmd[1] / (3.0 * OMNIROBOT_L) \
            - self.last_wheel_speeds_cmd[0] / (3.0 * OMNIROBOT_L) \
            - self.last_wheel_speeds_cmd[2] / (3.0 * OMNIROBOT_L)
            
        # translate the last velocity cmd in robot local coordiante to position cmd in gound coordiante
        cos_direction = np.cos(self.robot_yaw)
        sin_direction = np.sin(self.robot_yaw)

        ground_pos_cmd_x = self.robot_pos[0] + (local_speed_x *
                                                cos_direction - local_speed_y * sin_direction)/RL_CONTROL_FREQ
        ground_pos_cmd_y = self.robot_pos[1] + (local_speed_y *
                                                cos_direction + local_speed_x * sin_direction)/RL_CONTROL_FREQ
        ground_yaw_cmd = self.robot_yaw + local_rot_speed/RL_CONTROL_FREQ
        self.setRobotCmd(ground_pos_cmd_x, ground_pos_cmd_y, ground_yaw_cmd)

        self.last_wheel_speeds_cmd = np.float32(
            [left_speed, front_speed, right_speed])

    @staticmethod
    def normalizeAngle(angle):
        """
        :param angle: (float) (in rad)
        :return: (float) the angle in [-pi, pi] (in rad)
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class OmniRobotSimulatorSocket(OmnirobotManagerBase):
    def __init__(self, **args):
        '''
        Simulate the zmq socket like real omnirobot server 
        :param **args  arguments 

        '''
        default_args = {
            "back_ground_path": "real_robots/omnirobot_utils/back_ground.jpg",
            "camera_info_path": CAMERA_INFO_PATH,
            "robot_marker_path": "real_robots/omnirobot_utils/robot_margin3_pixel_only_tag.png",
            "robot_marker_margin": [3, 3, 3, 3],
            "target_marker_margin": [4, 4, 4, 4],
            "robot_marker_code": None,
            "target_marker_code": None,
            "robot_marker_length": 0.18,
            "target_marker_length": 0.18,
            "output_size": [224, 224],
            "init_x": 0,
            "init_y": 0,
            "init_yaw": 0,
            "origin_size": ORIGIN_SIZE,
            "cropped_size": CROPPED_SIZE,
            "circular_move": False
        }
        # overwrite the args if it exists
        self.new_args = {**default_args, **args}

        if self.new_args["simple_continual_target"]:
            self.new_args["target_marker_path"] = "real_robots/omnirobot_utils/red_square.png"

        elif self.new_args["circular_continual_move"]:
            self.new_args["target_marker_path"] = "real_robots/omnirobot_utils/blue_square.png"

        elif self.new_args["square_continual_move"]:
            self.new_args["target_marker_path"] = "real_robots/omnirobot_utils/green_square.png"
        else:
            # for black target, use target_margin4_pixel.png",
            self.new_args["target_marker_path"] = "real_robots/omnirobot_utils/red_target_margin4_pixel_480x480.png"

        super(OmniRobotSimulatorSocket, self).__init__(simple_continual_target=self.new_args["simple_continual_target"],
                                                   circular_continual_move=self.new_args["circular_continual_move"],
                                                   square_continual_move=self.new_args["square_continual_move"])

        assert len(self.new_args['robot_marker_margin']) == 4
        assert len(self.new_args['target_marker_margin']) == 4
        assert len(self.new_args['output_size']) == 2

        self.robot = OmniRobotEnvRender(**self.new_args)
        self.episode_idx = 0
        self._random_target = self.new_args["random_target"]
        self.resetEpisode()  # for a random target initial position

    def resetEpisode(self):
        """
        override the original method
        Give the correct sequance of commands to the robot 
        to rest environment between the different episodes
        """
        if self.second_cam_topic is not None:
            assert NotImplementedError
        # Env reset
        random_init_position = self.sampleRobotInitalPosition()
        self.robot.setRobotCmd(
            random_init_position[0], random_init_position[1], 0)

        self.robot_marker_size_proprotion = np.random.randn(
        ) * NOISE_VAR_ROBOT_SIZE_PROPOTION + 1.0

        # target reset
        if self._random_target or self.episode_idx == 0:
            random_init_x = np.random.random_sample() * (TARGET_MAX_X - TARGET_MIN_X) + \
                TARGET_MIN_X
            random_init_y = np.random.random_sample() * (TARGET_MAX_Y - TARGET_MIN_Y) + \
                TARGET_MIN_Y
            self.robot.setTargetCmd(
                random_init_x, random_init_y, 2 * np.pi * np.random.rand() - np.pi)

        # render the target and robot
        self.robot.renderTarget()
        self.robot.renderRobot()

    def send_json(self, msg):
        # env send msg to render
        self.processMsg(msg)

        self.robot.renderRobot()

        self.img = self.robot.getCroppedImage()
        self.img = self.robot.renderEnvLuminosityNoise(self.img, noise_var=NOISE_VAR_ENVIRONMENT, in_RGB=False,
                                                       out_RGB=True)
        self.img = cv2.resize(self.img, tuple(self.robot.output_size))

    def recv_json(self):
        msg = {
            # XYZ position
            "position": self.robot.robot_pos.tolist(),
            "reward": self.reward,
            "target_pos": self.robot.target_pos.tolist()
        }
        return msg

    def recv_image(self):
        return self.img
