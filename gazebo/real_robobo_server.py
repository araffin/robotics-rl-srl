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

def processImageWithColorMask(image, debug=False):
    """
    :param image: (bgr image)
    :param debug: (bool)
    :return: (int, int)
    """
    error = False
    r = [0, 0, image.shape[1], image.shape[0]]
    margin_left, margin_top, _, _ = r

    im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    image = im_cropped
    # if debug:
    #     cv2.imshow('crop', im_cropped)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_red = np.array([120, 130, 0])
    upper_red = np.array([135, 255, 255])

    # lower_red = np.array([148, 137, 0])
    # upper_red = np.array([176, 255, 255])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel_erode = np.ones((4, 4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=2)

    kernel_dilate = np.ones((6, 6), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=2)

    if debug:
        cv2.imshow('mask', mask)
        cv2.imshow('eroded', eroded_mask)
        cv2.imshow('dilated', dilated_mask)

    # cv2.RETR_CCOMP  instead of cv2.RETR_TREE
    im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    if debug:
        # Draw biggest
        # cv2.drawContours(image, contours, 0, (0,255,0), 3)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    if len(contours) > 0:
        M = cv2.moments(contours[0])
        # Centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contours[0])
    else:
        cx, cy = 0, 0
        area = 0
        error = True

    if debug:
        if error:
            print("No centroid found")
        else:
            print("Found centroid at ({}, {})".format(cx, cy))
        cv2.circle(image, (cx, cy), radius=10, color=(0, 0, 255),
                   thickness=1, lineType=8, shift=0)
        cv2.imshow('result', image)
    return cx, cy, area, error

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
    img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback, queue_size=1)

if SECOND_CAM_TOPIC is not None:
    DATA_FOLDER_SECOND_CAM = "real_robobo_2nd_cam"
    image_cb_wrapper_2 = ImageCallback()
    img_2_sub = rospy.Subscriber(SECOND_CAM_TOPIC, Image, image_cb_wrapper_2.imageCallback, queue_size=1)


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
        self.position = [0, 0]
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
        time.sleep(1.1 * self.time_forward)

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
        # print("face_idx", self.current_face_idx, self.faces[self.current_face_idx])
        self.yaw_target = self.angles[self.faces[self.current_face_idx]]
        # print("yaw_target", self.yaw_target)

    def computeTime(self, direction):
        """
        :param direction: (str) "left" or "right"
        :return: (float)
        """
        self.yaw_error = 0
        t = (abs(self.directions[direction] + self.yaw_error) - self.angle_offset) / self.angle_coeff + 1
        # print("yaw", self.yaw, "current_face", self.faces[self.current_face_idx])
        # print("yaw_north", self.yaw_north, 'direction', direction)
        # print("time:", t)
        print("yaw", self.yaw, "yaw_north", self.yaw_north)
        return t

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
robobo.stop()

# Init robot yaw angle
robobo.turn(robobo.computeTime('left'), -robobo.speed)
robobo.turn(robobo.computeTime('right'), robobo.speed)
robobo.initYawNorth()

initial_area = 3700

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

    has_bumped = False
    if action == Move.FORWARD:
        if robobo.position[1] < MAX_Y:
            robobo.forward()
            robobo.position[1] += 1
        else:
            has_bumped = True
    elif action == Move.STOP:
        robobo.stop()
    elif action == Move.RIGHT:
        if robobo.position[0] < MAX_X:
            robobo.turnRight()
            robobo.forward()
            robobo.turnLeft()
            robobo.position[0] += 1
        else:
            has_bumped = True
    elif action == Move.LEFT:
        if robobo.position[0] > MIN_X:
            robobo.turnLeft()
            robobo.forward()
            robobo.turnRight()
            robobo.position[0] -= 1
        else:
            has_bumped = True
    elif action == Move.BACKWARD:
        if robobo.position[1] > MIN_Y:
            robobo.backward()
            robobo.position[1] -= 1
        else:
            has_bumped = True
    elif action == None:
        # Env reset
        pass
    else:
        print("Unsupported action")

    print("Updating image")
    original_image = np.copy(image_cb_wrapper.valid_img)
    cx, cy, area, error = processImageWithColorMask(original_image.copy(), debug=False)

    # cv2.imshow('image', original_image)
    # cv2.waitKey(1000)

    delta_area_rate = (initial_area - area) / initial_area

    print("Image processing", cx, cy, area, error)
    reward = 0
    # Consider that we reached the target if we are close enough
    if delta_area_rate > 0.2:
        reward = 1
        print("Target reached!")

    if has_bumped:
        reward = -1
        print("Bumped into wall")
        print()

    print("Robobo position", robobo.position)
    socket.send_json(
        {
            # XYZ position
            "position": list(robobo.position),
            "reward": reward,
            "target_pos": list([cx, cy])
        },
        flags=zmq.SNDMORE if IMAGE_TOPIC is not None else 0
    )

    if SECOND_CAM_TOPIC is not None:
        saveSecondCamImage(image_cb_wrapper_2.valid_img, episode_folder, episode_step, DATA_FOLDER_SECOND_CAM)
        episode_step += 1

    if IMAGE_TOPIC is not None:
        # # Retrieve last image from image topic
        # img = image_cb_wrapper.valid_img
        # to contiguous, otherwise ZMQ will complain
        img = np.ascontiguousarray(original_image, dtype=np.uint8)
        sendMatrix(socket, img)

print(" Exiting server - closing socket...")
socket.close()
