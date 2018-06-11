#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import os
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


def findTarget(image, debug=False):
    """
    Find the target in the image using color thresholds
    :param image: (bgr image)
    :param debug: (bool) Whether to display the image or not
    :return: (int, int, float, bool)
    """
    error = False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_red = np.array([120, 130, 0])
    upper_red = np.array([135, 255, 255])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Remove noise
    kernel_erode = np.ones((4, 4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=2)

    kernel_dilate = np.ones((6, 6), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=2)

    if debug:
        cv2.imshow('mask', mask)
        cv2.imshow('eroded', eroded_mask)
        cv2.imshow('dilated', dilated_mask)

    # Retrieve contours
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


class Robobo(object):
    """
    Class for controlling Robobo
    """

    def __init__(self):
        super(Robobo, self).__init__()
        # Duration of the "FORWARD" action
        self.time_forward = 1.7

        self.speed = 10
        # Angle that robobo achieve in one second
        # at a given speed
        self.angle_offset = 38
        # Degree per s when turning after the 1st second
        # at a given speed
        # From calibration
        self.angle_coeff = 50
        # Robobo's position on the grid
        self.position = [0, 0]

        self.directions = {'left': 90, 'right': -90}
        self.faces = ['west', 'north', 'east']
        self.current_face_idx = 1
        self.yaw_error = 0
        self.yaw_target = 0
        self.yaw_north = 0
        self.yaw = 0
        self.left_encoder_pos = 0
        self.right_encoder_pos = 0
        self.angles = {}

        # Attempt connection to Robobo's service
        try:
            self.robobo_command = rospy.ServiceProxy('/command', Command)
        except rospy.ServiceException as e:
            print("Service exception", str(e))
            exit(1)

        self.status_sub = rospy.Subscriber("/status", Status, self.statusCallback)

    def moveForever(self, lspeed, rspeed, speed):
        """
        :param lspeed: (str) "forward" or "backward"
        :param rspeed: (str)
        :param speed: (int)
        """
        command_name = 'MOVE-FOREVER'
        command_parameters = [KeyValue('lspeed', lspeed), KeyValue('rspeed', rspeed), KeyValue('speed', str(speed))]
        self.robobo_command(command_name, 0, command_parameters)

    def statusCallback(self, status):
        """
        Callback for ROS topic
        :param status: (Status ROS message)
        """
        # Update the current yaw using phone gyroscope
        # NOTE: this may not work depending on the phone
        if status.name == 'ORIENTATION':
            for KeyV in status.value:
                if KeyV.key == 'yaw':
                    self.yaw = float(KeyV.value)
        # Update position of the two encoders
        if status.name == 'WHEELS':
            for KeyV in status.value:
                if KeyV.key == 'wheelPosL':
                    self.left_encoder_pos = float(KeyV.value)
                elif KeyV.key == 'wheelPosR':
                    self.right_encoder_pos = float(KeyV.value)

    def stop(self):
        """
        Stop robobo
        """
        self.moveForever('forward', 'forward', 0)

    def initYawNorth(self):
        """
        Initialize the reference yaw that represents the North
        """
        self.yaw_north = self.yaw
        self.angles = {
            'north': self.yaw_north,
            'east': self.normalizeAngle(self.yaw_north - 90),
            'west': self.normalizeAngle(self.yaw_north + 90)
        }
        self.current_face_idx = 1
        self.yaw_target = self.yaw_north
        self.yaw_error = 0

    def forward(self):
        """
        Move one step forward (Translation)
        """
        self.move(self.time_forward, self.speed)
        time.sleep(1.1 * self.time_forward)

    def backward(self):
        """
        Move one step backward
        """
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
        """
        Update the error between desired yaw and current one
        """
        self.yaw_error = self.normalizeAngle(self.yaw_target - self.yaw)
        print("yaw_error", self.yaw_error)

    def updateTarget(self):
        """
        Update the target angle
        """
        self.yaw_target = self.angles[self.faces[self.current_face_idx]]
        # print("face_idx", self.current_face_idx, self.faces[self.current_face_idx])
        # print("yaw_target", self.yaw_target)

    def computeTime(self, direction):
        """
        Compute the time needed for a rotation to face a given direction
        It is meant to correct the previous error
        however this does not seems to work for now
        :param direction: (str) "left" or "right"
        :return: (float)
        """
        # Cancel the error, gives better performance (less drift)
        self.yaw_error = 0
        t = (abs(self.directions[direction] + self.yaw_error) - self.angle_offset) / self.angle_coeff + 1
        # print("yaw", self.yaw, "current_face", self.faces[self.current_face_idx])
        # print("yaw_north", self.yaw_north, 'direction', direction)
        # print("time:", t)
        print("yaw", self.yaw, "yaw_north", self.yaw_north)
        return t

    def move(self, t, speed):
        """
        Translation move
        :param t: (float) duration of Translation
        :param speed: (int)
        """
        command_parameters = [KeyValue('lspeed', str(speed)), KeyValue('rspeed', str(speed)), KeyValue('time', str(t))]
        self.robobo_command("MOVE", 0, command_parameters)

    def turn(self, t, speed):
        """
        Rotation move
        :param t: (float) duration of Rotation
        :param speed: (int)
        """
        command_parameters = [KeyValue('lspeed', str(speed)), KeyValue('rspeed', str(-speed)), KeyValue('time', str(t))]
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
    # We are always facing North
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
    elif action is None:
        # Env reset
        pass
    else:
        print("Unsupported action")

    original_image = np.copy(image_cb_wrapper.valid_img)
    # Find the target in the image using color thresholds
    cx, cy, area, error = findTarget(original_image.copy(), debug=False)

    delta_area_rate = (TARGET_INITIAL_AREA - area) / TARGET_INITIAL_AREA

    print("Image processing:", cx, cy, area, error)
    reward = 0
    # Consider that we reached the target if we are close enough
    # we detect that computing the difference in area between TARGET_INITIAL_AREA
    # current detected area of the target
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

print("Exiting server - closing socket...")
socket.close()
