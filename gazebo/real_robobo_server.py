#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import os
import subprocess
import signal
import time

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

#  rosrun image_transport republish compressed in:=/camera/image raw out:=/camera/image_repub

bridge = CvBridge()
should_exit = [False]

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
if IMAGE_TOPIC is not None:
    image_cb_wrapper = ImageCallback()
    img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback)

if SECOND_CAM_TOPIC is not None:
    DATA_FOLDER_SECOND_CAM = "real_baxter_2nd_cam"
    image_cb_wrapper_2 = ImageCallback()
    img_2_sub = rospy.Subscriber(SECOND_CAM_TOPIC, Image, image_cb_wrapper_2.imageCallback)


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

t_turn = 2.04 # 90 degrees
t_forward = 1.7


class Robobo(object):
    def __init__(self):
        super(Robobo, self).__init__()
        self.Kp = 1
        self.Ki = 0.0
        self.Kp_angle = 1
        self.time_forward = 1.7

        self.speed = 10
        self.alpha_init = 0
        self.alpha_target = 0
        self.angle_offset = 38
        self.angle_coeff = 50

        self.directions = {'left': 90, 'right': -90}
        self.faces = ['west', 'north', 'east']
        self.yaw_error = 0
        self.current_face_idx = 1
        self.yaw_target = 0
        self.yaw_north = 0

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

    def statusCallback(self, status):
        if status.name == 'ORIENTATION':
            for KeyV in status.value:
                if KeyV.key == 'yaw':
                    self.yaw = float(KeyV.value)
        if status.name == 'WHEELS':
            for KeyV in status.value:
                if KeyV.key == 'wheelPosL':
                    self.left_encoder_pos = float(KeyV.value)
                elif KeyV.key == 'wheelPosR':
                    self.right_encoder_pos = float(KeyV.value)

    def stop(self):
        self.moveForever('forward', 'forward', 0)

    def initYawNorth(self):
        self.yaw_north = self.yaw
        self.angles = {
            'north': self.yaw_north,
            'east': self.normalizeAngle(self.yaw_north - 90),
            'west':  self.normalizeAngle(self.yaw_north + 90)
            }
        self.current_face_idx = 1
        self.yaw_target = self.yaw_north
        self.yaw_error = 0

    def forward(self):
        self.move(self.time_forward, self.speed)
        time.sleep(1.1 * self.time_forward)

    def backward(self):
        self.move(self.time_forward, -self.speed)

    def turnLeft(self):
        turn_time = self.computeTime('left')
        assert self.current_face_idx > 0
        self.current_face_idx -= 1
        self.updateTarget()
        self.turn(turn_time, -self.speed)
        self.updateError()

    def turnRight(self):
        turn_time = self.computeTime('right')
        assert self.current_face_idx < len(self.faces)
        self.current_face_idx += 1
        self.updateTarget()
        self.turn(turn_time, self.speed)
        self.updateError()

    def updateError(self):
        self.yaw_error = self.normalizeAngle(self.yaw_target - self.yaw)
        print("yaw_error", self.yaw_error)

    def updateTarget(self):
        print("face_idx", self.current_face_idx, self.faces[self.current_face_idx])
        self.yaw_target = self.angles[self.faces[self.current_face_idx]]
        print("yaw_target", self.yaw_target)

    def computeTime(self, direction):
        """
        :param direction: (str) "left" or "right"
        :return: (float)
        """
        print("yaw", self.yaw, "current_face", self.faces[self.current_face_idx])
        print("yaw_north", self.yaw_north, 'direction', direction)
        time = (abs(self.directions[direction] + self.yaw_error) - self.angle_offset) / self.angle_coeff + 1
        return time

    def move(self, t, speed):
        command_parameters = []
       	command_parameters.append(KeyValue('lspeed', str(speed)))
        command_parameters.append(KeyValue('rspeed', str(speed)))
        command_parameters.append(KeyValue('time', str(t)))
        self.robobo_command("MOVE", 0, command_parameters)

    def turn(self, t, speed):
        command_parameters = []
        command_parameters.append(KeyValue('lspeed', str(speed)))
        command_parameters.append(KeyValue('rspeed', str(-speed)))
        command_parameters.append(KeyValue('time', str(t)))
        self.robobo_command("MOVE", 0, command_parameters)
        time.sleep(1.1 * t + 2)
        print("MOVED")

    @staticmethod
    def normalizeAngle(angle):
        while angle > 180:
            angle -= 2 * 180
        while angle < -180:
            angle += 2 * 180
        return angle


robobo = Robobo()
# Initialize encoders
robobo.stop()

robobo.turn(robobo.computeTime('left'), -robobo.speed)
robobo.turn(robobo.computeTime('right'), robobo.speed)
robobo.initYawNorth()

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
    if action == Move.FORWARD:
        robobo.forward()
    elif action == Move.STOP:
        robobo.stop()
    elif action == Move.RIGHT:
        robobo.turnRight()
        robobo.forward()
        robobo.turnLeft()
    elif action == Move.LEFT:
        robobo.turnLeft()
        robobo.forward()
        robobo.turnRight()
    elif action == Move.BACKWARD:
        robobo.backward()
    else:
        print("Unsupported action")
    # print(robobo.yaw)

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
        flags=zmq.SNDMORE if IMAGE_TOPIC is not None else 0
    )

    if SECOND_CAM_TOPIC is not None:
        saveSecondCamImage(image_cb_wrapper_2.valid_img, episode_folder, episode_step, DATA_FOLDER_SECOND_CAM)
        episode_step += 1

    if IMAGE_TOPIC is not None:
        # Retrieve last image from image topic
        img = image_cb_wrapper.valid_img
        # to contiguous, otherwise ZMQ will complain
        img = np.ascontiguousarray(img, dtype=np.uint8)
        sendMatrix(socket, img)

print(" Exiting server - closing socket...")
socket.close()
