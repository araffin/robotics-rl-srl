#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import os
import subprocess
import signal
from enum import Enum

import numpy as np
import rospy
import zmq
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from com_mytechia_robobo_ros_msgs.srv import Command
from com_mytechia_robobo_ros_msgs.msg import KeyValue, Status

from .constants import *
from .utils import sendMatrix

assert USING_ROBOBO, "Please set USING_ROBOBO to True in gazebo/constants.py"

bridge = CvBridge()
should_exit = [False]

class Move(Enum):
    STOP = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4


# exit the script on ctrl+c
def ctrl_c(signum, frame):
    should_exit[0] = True


signal.signal(signal.SIGINT, ctrl_c)


class ImageCallback(object):
    """
    Image callback for ROS
    """
    def __init__(self):
        super(ImageCallback, self).__init__()
        self.valid_img = None

    def imageCallback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV
            cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
            self.valid_img = cv2_img
        except CvBridgeError as e:
            print("CvBridgeError:", e)



def saveSecondCamImage(im, episode_folder, episode_step, path="robobo_2nd_cam"):
    """
    Write an image to disk
    :param im: (numpy matrix) BGR image
    :param episode_folder: (str)
    :param episode_step: (int)
    :param path: (str)
    """
    image_path = "{}/{}/frame{:06d}.jpg".format(path, episode_folder, episode_step)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite("srl_zoo/data/{}".format(image_path), im)


rospy.init_node('robobo_server', anonymous=True)

# Connect to ROS Topics
image_cb_wrapper = ImageCallback()
img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback)

if SECOND_CAM_TOPIC is not None:
    DATA_FOLDER_SECOND_CAM = "real_baxter_2nd_cam"
    image_cb_wrapper_2 = ImageCallback()
    img_2_sub = rospy.Subscriber(SECOND_CAM_TOPIC, Image, image_cb_wrapper_2.imageCallback)


print("Initializing robot...")
# Init robot pose

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
robot_position = np.array([0, 0])


class Robobo(object):
    def __init__(self):
        super(Robobo, self).__init__()
        self.Kp = 1
        self.Kp_angle = 1
        self.max_speed = 20
        self.max_angular_speed = 10
        self.error_threshold = 2
        self.left_encoder_pos, self.right_encoder_pos = 0, 0
        self.yaw = 0
        try:
            self.robobo_command = rospy.ServiceProxy('/command', Command)
        except rospy.ServiceException, e:
            print("Service exception", str(e))
            exit(1)

        self.status_sub = rospy.Subscriber("/status", Status, self.statusCallback)

    def moveForever(self, lspeed, rspeed, speed):
        command_name = 'MOVE-FOREVER'
        command_parameters = []
        command_parameters.append(KeyValue('lspeed', lspeed))
        command_parameters.append(KeyValue('rspeed', rspeed))
        command_parameters.append(KeyValue('speed', str(speed)))
        self.robobo_command(command_name, 0, command_parameters)


    def statusCallback(self):
        if status.name == 'ORIENTATION':
            print(status.value)
            self.yaw = 0
        if status.name == 'WHEELS':
            print(status.value)
            # encoderPosR
            self.left_encoder = 0
            self.right_encoder = 0


    def stop(self):
        self.moveForever('forward', 'forward', 0)

    def rotate(self):
        left_start, right_start = self.left_encoder_pos, self.right_encoder_pos
        while (error_left + error_right) < self.error_threshold:
            error_left = DELTA_TICS - (self.left_encoder_pos - left_start)
            error_right = -1 * (DELTA_TICS - (self.right_encoder_pos - right_start))
            speed_left = np.clip(self.Kp_angle * error_left, -self.max_angular_speed, self.max_angular_speed)
            speed_right = np.clip(self.Kp_angle * error_right, -self.max_angular_speed, self.max_angular_speed)
            self.moveForever('forward', 'off', speed_left)
            self.moveForever('off', 'forward', speed_right)
        self.stop()

    def forward(self):
        left_start, right_start = self.left_encoder_pos, self.right_encoder_pos
        while (error_left + error_right) < self.error_threshold:
            error_left = DELTA_TICS - (self.left_encoder_pos - left_start)
            error_right = DELTA_TICS - (self.right_encoder_pos - right_start)
            speed_left = np.clip(self.Kp * error_left, -self.max_speed, self.max_speed)
            speed_right = np.clip(self.Kp * error_right, -self.max_speed, self.max_speed)
            self.moveForever('forward', 'off', speed_left)
            self.moveForever('off', 'forward', speed_right)
        self.stop()

robobo = Robobo()

while not should_exit[0]:
    msg = socket.recv_json()
    command = msg.get('command', '')

    if command == 'reset':
        print('Environment reset')
        action = None
        episode_idx += 1
        episode_step = 0

        if SECOND_CAM_TOPIC is not None:
            episode_folder = "record_{:03d}".format(episode_idx)
            try:
                os.makedirs("srl_zoo/data/{}/{}".format(DATA_FOLDER_SECOND_CAM, episode_folder))
            except OSError:
                pass

    elif command == 'action':
        print("action (int)", msg['action'])
        action = Move(msg['action'])
        print("action (move):", action)

    elif command == "exit":
        break
    else:
        raise ValueError("Unknown command: {}".format(msg))

    # dx = [-dv, dv, 0, 0][action]
    # dy = [0, 0, -dv, dv][action]
    # real_action = np.array([dx, dy])
    # TODO: update robot position
    if action == FORWARD:
        robobo.forward()
    elif action == BACKWARD:
        robobo.backward()        #
    elif action == LEFT:
        robobo.turnLeft()
        robobo.forward()
        robobo.turnRight()
    elif action == RIGHT:
        robobo.turnRight()
        robobo.forward()
        robobo.turnLeft()
    elif action == STOP:
        pass
    else:
        print("Unsupported action")

    reward = 0
    # Consider that we touched the button if we are close enough
    # if np.linalg.norm(target_pos - robot_position, 2) < DIST_TO_TARGET_THRESHOLD:
    #     reward = 1
    #     print("Target reached!")

    # Send arm position, button position, ...
    socket.send_json(
        {
            # XYZ position
            "position": list(robot_position),
            "reward": reward,
            "target_pos": list([0, 0])
        },
        flags=zmq.SNDMORE
    )
    # Retrieve last image from image topic
    img = image_cb_wrapper.valid_img

    if SECOND_CAM_TOPIC is not None:
        saveSecondCamImage(image_cb_wrapper_2.valid_img, episode_folder, episode_step, DATA_FOLDER_SECOND_CAM)
        episode_step += 1
    # to contiguous, otherwise ZMQ will complain
    img = np.ascontiguousarray(img, dtype=np.uint8)
    sendMatrix(socket, img)

print(" Exiting server - closing socket...")
socket.close()
