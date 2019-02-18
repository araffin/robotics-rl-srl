#!/usr/bin/env python

from __future__ import division, print_function, absolute_import
import rospy

import os
import time


import numpy as np
import zmq
import argparse
import yaml
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, PoseStamped
from std_msgs.msg import Bool


from tf.transformations import euler_from_quaternion

from real_robots.constants import *
from real_robots.utils import sendMatrix
from real_robots.omnirobot_utils.omnirobot_manager_base import OmnirobotManagerBase
assert USING_OMNIROBOT, "Please set USING_OMNIROBOT to True in real_robots/constants.py"

NO_TARGET_MODE = False

bridge = CvBridge()
should_exit = [False]


class OmniRobot(object):
    def __init__(self, init_x, init_y, init_yaw):
        """
        Class for controlling omnirobot based on topic mechanism of ros
        """

        super(OmniRobot, self).__init__()

        # Initialize the direction
        self.init_pos = [init_x, init_y]
        self.init_yaw = init_yaw

        # OmniRobot's real position on the grid
        self.robot_pos = [0, 0]
        self.robot_yaw = 0  # in rad

        # OmniRobot's position command on the grid
        self.robot_pos_cmd = self.init_pos[:]
        self.robot_yaw_cmd = self.init_yaw

        # Target's real position on the grid
        self.target_pos = [0, 0]
        self.target_yaw = 0

        # status of moving
        self.move_finished = False
        self.target_pos_changed = False

        # Distance for each step
        self.step_distance = STEP_DISTANCE

        self.visual_robot_sub = rospy.Subscriber(
            "/visual_robot_pose", PoseStamped, self.visualRobotCallback, queue_size=10)
        self.visual_target_sub = rospy.Subscriber(
            "/visual_target_pose", PoseStamped, self.visualTargetCallback, queue_size=10)

        self.pos_cmd_pub = rospy.Publisher(
            "/position_commands", Vector3, queue_size=10)
        self.move_finished_sub = rospy.Subscriber(
            "/finished", Bool, self.moveFinishedCallback, queue_size=10)
        self.stop_signal_pub = rospy.Publisher("/stop", Bool, queue_size=10)
        self.reset_odom_pub = rospy.Publisher(
            "/reset_odom", Vector3, queue_size=10)
        self.reset_signal_pub = rospy.Publisher("/reset", Bool, queue_size=10)

        # known issues, without sleep 1 second, publishers could not been setup
        rospy.sleep(1)
        # https://answers.ros.org/question/9665/test-for-when-a-rospy-publisher-become-available/?answer=14125#post-id-14125

    def setRobotCmdConstrained(self, x, y, yaw):
        """
        set the position command for the robot, the command will be automatically constrained
        x, y, yaw are in the global frame
        Note: the command will be not published until pubPosCmd() is called
        """
        self.robot_pos_cmd[0] = max(x, MIN_X)
        self.robot_pos_cmd[0] = min(x, MAX_X)

        self.robot_pos_cmd[1] = max(y, MIN_Y)
        self.robot_pos_cmd[1] = min(y, MAX_Y)
        self.robot_yaw_cmd = self.normalizeAngle(yaw)

    def setRobotCmd(self, x, y, yaw):
        """
        set the position command for the robot
        x, y, yaw are in the global frame
        Note: the command will be not published until pubPosCmd() is called
        """
        self.robot_pos_cmd[0] = x
        self.robot_pos_cmd[1] = y
        self.robot_yaw_cmd = self.normalizeAngle(yaw)

    def pubPosCmd(self):
        """
        Publish the position command for the robot
        x, y, yaw are in the global frame
        """
        msg = Vector3(
            self.robot_pos_cmd[0], self.robot_pos_cmd[1], self.robot_yaw_cmd)
        self.pos_cmd_pub.publish(msg)
        self.move_finished = False
        print("move to x: {:.4f} y:{:.4f} yaw: {:.4f}".format(
            msg.x, msg.y, msg.z))

    def resetOdom(self, x, y, yaw):
        """
        The odometry of robot will be reset to x, y, yaw (in global frame)
        """
        msg = Vector3()
        msg.x = x
        msg.y = y
        msg.z = yaw
        self.reset_odom_pub.publish(msg)

    def reset(self):
        """
        Publish the reset signal to robot (quit the stop state)
        The odometry will not be reset automatically
        """
        msg = Bool()
        self.reset_signal_pub.publish(msg)

    def stop(self):
        """
        Publish the stop signal to robot
        """
        msg = Bool()
        msg.data = True
        self.stop_signal_pub.publish(msg)
        self.move_finished = True

    def visualRobotCallback(self, pose_stamped_msg):
        """
        Callback for ROS topic
        Get the new updated position of robot from camera
        :param pose_stamped_msg: (PoseStamped ROS message)
        """
        self.robot_pos[0] = pose_stamped_msg.pose.position.x
        self.robot_pos[1] = pose_stamped_msg.pose.position.y
        self.robot_yaw = euler_from_quaternion([pose_stamped_msg.pose.orientation.x, pose_stamped_msg.pose.orientation.y,
                                                pose_stamped_msg.pose.orientation.z, pose_stamped_msg.pose.orientation.w])[2]

        if NO_TARGET_MODE and self.target_pos_changed:
            # simulate the target's position update
            self.target_pos[0] = 0.0
            self.target_pos[1] = 0.0
            self.target_yaw = 0.0
            self.target_pos_changed = False

    def visualTargetCallback(self, pose_stamped_msg):
        """
        Callback for ROS topic
        Get the new updated position of robot from camera
        Only update when target position should have been moved (eg. reset env)
        :param pose_stamped_msg: (PoseStamped ROS message)
        """

        if self.target_pos_changed:
            if pose_stamped_msg.pose.position.x < TARGET_MAX_X and pose_stamped_msg.pose.position.x > TARGET_MIN_X  \
                    and pose_stamped_msg.pose.position.y > TARGET_MIN_Y and pose_stamped_msg.pose.position.y < TARGET_MAX_Y:
                self.target_pos[0] = pose_stamped_msg.pose.position.x
                self.target_pos[1] = pose_stamped_msg.pose.position.y
                self.target_yaw = euler_from_quaternion([pose_stamped_msg.pose.orientation.x, pose_stamped_msg.pose.orientation.y,
                                                         pose_stamped_msg.pose.orientation.z, pose_stamped_msg.pose.orientation.w])[2]
                self.target_pos_changed = False

    def moveFinishedCallback(self, move_finished_msg):
        """
        Callback for ROS topic
        receive 'finished' signal when robot moves to the target
        """
        self.move_finished = move_finished_msg.data

    def forward(self):
        """
        Move one step forward (Translation)
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0] + self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)
        self.pubPosCmd()

    def backward(self):
        """
        Move one step backward
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0] - self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)
        self.pubPosCmd()

    def left(self):
        """
        Translate to the left
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0], self.robot_pos_cmd[1] + self.step_distance, self.robot_yaw_cmd)
        self.pubPosCmd()

    def right(self):
        """
        Translate to the right
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0], self.robot_pos_cmd[1] - self.step_distance, self.robot_yaw_cmd)
        self.pubPosCmd()

    def moveContinous(self, action):
        """
        Perform a continuous displacement of dx, dy
        """
        self.setRobotCmd(
            self.robot_pos_cmd[0] + action[0], self.robot_pos_cmd[1] + action[1], self.robot_yaw_cmd)
        self.pubPosCmd()

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


class ImageCallback(object):
    """
    Image callback for ROS
    """

    def __init__(self, camera_matrix, distortion_coefficients):
        super(ImageCallback, self).__init__()
        self.valid_img = None
        self.valid_box = None
        self.first_msg = True
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

    def imageCallback(self, msg):
        try:    
            # Convert your ROS Image message to OpenCV
            cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
            
            if self.first_msg:
                shape = cv2_img.shape
                min_length = min(shape[0], shape[1])
                up_margin = int((shape[0] - min_length) / 2)  # row
                left_margin = int((shape[1] - min_length) / 2)  # col
                self.valid_box = [up_margin, up_margin + min_length, left_margin, left_margin + min_length]
                print("origin size: {}x{}".format(shape[0],shape[1]))
                print("crop each image to a square image, cropped size: {}x{}".format(min_length, min_length))
                self.first_msg = False
            
            undistort_image = cv2.undistort(cv2_img, self.camera_matrix, self.distortion_coefficients)
            self.valid_img = undistort_image[self.valid_box[0]:self.valid_box[1], self.valid_box[2]:self.valid_box[3]]

        except CvBridgeError as e:
            print("CvBridgeError:", e)

def saveSecondCamImage(im, episode_folder, episode_step, path="omnirobot_2nd_cam"):
    """
    Write an image to disk
    :param im: (numpy matrix) BGR image
    :param episode_folder: (str)
    :param episode_step: (int)
    :param path: (str)
    """
    image_path = "{}/{}/frame{:06d}.jpg".format(
        path, episode_folder, episode_step)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite("srl_zoo/data/{}".format(image_path), im)


def waitTargetUpdate(omni_robot, timeout):
    """
    Wait for target updating
    :param omni_robot: (OmniRobot)
    :param timeout: (time to wait for updating)
    """
    omni_robot.target_pos_changed = True
    time = 0.0  # second
    while time < timeout and not rospy.is_shutdown():
        if not omni_robot.target_pos_changed:  # updated
            return True
        else:
            rospy.sleep(0.1)
            time += 0.1
    return False

class OmnirobotManager(OmnirobotManagerBase):
    """
    Omnirobot magager for real robot
    """
    def __init__(self):
        super(OmnirobotManager, self).__init__(second_cam_topic=SECOND_CAM_TOPIC)
        self.robot = OmniRobot(0,0,0) # assign the real robot object to manager
        self.episode_idx = 0
        self.episode_step = 0
    def resetEpisode(self):
        """
        override orignal method
        Give the correct sequance of commands to the robot 
        to rest environment between the different episodes
        """
        if self.second_cam_topic is not None:
            assert NotImplementedError
            episode_folder = "record_{:03d}".format(episode_idx)
            try:
                os.makedirs(
                    "srl_zoo/data/{}/{}".format(DATA_FOLDER_SECOND_CAM, episode_folder))
            except OSError:
                pass

        print('Environment reset, choose random position')
        self.episode_idx += 1
        self.episode_step = 0
        self.robot.reset()

        # Env reset
        random_init_position = self.sampleRobotInitalPosition()
        self.robot.setRobotCmd(random_init_position[0], random_init_position[1], 0)
        self.robot.pubPosCmd()

        while True:  # check the new target can be seen
            if not NO_TARGET_MODE:
                raw_input(
                    "please set the target position, then press 'enter' !")

            if waitTargetUpdate(self.robot, timeout=0.5):
                break
            else:
                print("Can't see the target, please move it into the boundary!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Server for omnirobot")
    parser.add_argument('--camera-info-path', default="real_robots/omnirobot_utils/cam_calib_info.yaml", type=str,
                        help="camera calibration file generated by ros calibrate")
    args = parser.parse_args()
    with open(args.camera_info_path, 'r') as stream:
        try:
            contents = yaml.load(stream)
            camera_matrix = np.array(contents['camera_matrix']['data']).reshape((3,3))
            distortion_coefficients = np.array(
                contents['distortion_coefficients']['data']).reshape((1, 5))
        except yaml.YAMLError as exc:
            print(exc)
    rospy.init_node('omni_robot_server', anonymous=True)
    # warning for no target mode
    if NO_TARGET_MODE:
        rospy.logwarn(
            "ATTENTION: This script is running under NO TARGET mode!!!")
    # Connect to ROS Topics
    if IMAGE_TOPIC is not None:
        image_cb_wrapper = ImageCallback(camera_matrix, distortion_coefficients)
        img_sub = rospy.Subscriber(
            IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback, queue_size=1)

    if SECOND_CAM_TOPIC is not None:
        assert NotImplementedError
        image_cb_wrapper_2 = ImageCallback(camera_matrix, distortion_coefficients)
        img_2_sub = rospy.Subscriber(
            SECOND_CAM_TOPIC, Image, image_cb_wrapper_2.imageCallback, queue_size=1)

    print('Starting up on port number {}'.format(SERVER_PORT))
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    socket.bind("tcp://*:{}".format(SERVER_PORT))

    print("Waiting for client...")
    socket.send_json({'msg': 'hello'})
    print("Connected to client")

    action = 0
    episode_step = 0
    episode_idx = -1
    episode_folder = None

    omnirobot_manager = OmnirobotManager()
    omnirobot_manager.robot = OmniRobot(0, 0, 0)  # yaw is in rad
    omnirobot_manager.robot.stop()  # after stop, the robot need to be reset
    omnirobot_manager.robot.resetOdom(0, 0, 0)
    omnirobot_manager.robot.reset()

    omnirobot_manager.robot.pubPosCmd()
    r = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        print("wait for new command")
        msg = socket.recv_json()

        print("msg: {}".format(msg))
        omnirobot_manager.processMsg(msg)

        # wait for robot to finish the action, timeout 30s
        timeout = 30  # second
        for i in range(timeout):
            readable_list, _, _ = zmq.select([socket], [], [], 0)

            if len(readable_list) > 0:
                print("New command incomes, ignore the current command")
                continue
            if omnirobot_manager.robot.move_finished:
                print("action done")
                break

            elif i == timeout - 1:
                print("Error: timeout for action finished signal")
                exit()
            time.sleep(1)

        if IMAGE_TOPIC is not None:
            # Retrieve last image from image topic
            original_image = np.copy(image_cb_wrapper.valid_img)

        print("reward: {}".format(omnirobot_manager.reward))
        print("omni_robot position", omnirobot_manager.robot.robot_pos)
        print("target position", omnirobot_manager.robot.target_pos)
        socket.send_json(
            {
                # XYZ position
                "position": omnirobot_manager.robot.robot_pos,
                "reward": omnirobot_manager.reward,
                "target_pos": omnirobot_manager.robot.target_pos
            },
            flags=zmq.SNDMORE if IMAGE_TOPIC is not None else 0
        )

        if SECOND_CAM_TOPIC is not None:
            saveSecondCamImage(image_cb_wrapper_2.valid_img,
                               episode_folder, episode_step, DATA_FOLDER_SECOND_CAM)
            episode_step += 1

        if IMAGE_TOPIC is not None:
            # to contiguous, otherwise ZMQ will complain
            img = np.ascontiguousarray(original_image, dtype=np.uint8)
            sendMatrix(socket, img)
        r.sleep()

    print("Exiting server - closing socket...")
    socket.close()
