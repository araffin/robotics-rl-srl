#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import os
import subprocess
import signal

import baxter_interface
import numpy as np
import rospy
import zmq
import cv2
from arm_scenario_experiments import baxter_utils
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from .constants import *
from .utils import sendMatrix

assert USING_REAL_BAXTER, "Please set USING_REAL_BAXTER to True in gazebo/constants.py"

bridge = CvBridge()
should_exit = [False]


# exit the script on ctrl+c
def ctrl_c(signum, frame):
    should_exit[0] = True


signal.signal(signal.SIGINT, ctrl_c)


def resetPose():
    """
    Enable Baxter robot (if necessary) and reset the lefy arm position
    """
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    if rs.state().enabled:
        print("Robot already enabled")
    else:
        print("Enabling robot... ")
        rs.enable()
        # Untuck arms
        subprocess.call(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])
    print("Moving left arm to init")
    moveLeftArmToInit()


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


def moveLeftArmToInit():
    """
    Initialize robot left arm to starting position (hardcoded)
    :return: ([float])
    """
    joints = None
    position = LEFT_ARM_INIT_POS
    while not joints:
        try:
            joints = baxter_utils.IK(left_arm, position, LEFT_ARM_ORIENTATION, IK_SEED_POSITIONS)
        except Exception:
            try:
                joints = baxter_utils.IK(left_arm, position, LEFT_ARM_ORIENTATION, IK_SEED_POSITIONS)
            except Exception:
                raise
    left_arm.move_to_joint_positions(joints)
    return position


def saveSecondCamImage(im, episode_folder, episode_step, path="real_baxter_2nd_cam"):
    """
    Write an image to disk
    :param im: (numpy matrix) BGR image
    :param episode_folder: (str)
    :param episode_step: (int)
    :param path: (str)
    """
    image_path = "{}/{}/frame{:06d}.jpg".format(path, episode_folder, episode_step)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite("srl_priors/data/{}".format(image_path), im)


rospy.init_node('real_baxter_server', anonymous=True)

# Connect to ROS Topics
image_cb_wrapper = ImageCallback()
img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback)

if SECOND_CAM_TOPIC is not None:
    DATA_FOLDER_SECOND_CAM = "real_baxter_2nd_cam"
    image_cb_wrapper_2 = ImageCallback()
    img_2_sub = rospy.Subscriber(SECOND_CAM_TOPIC, Image, image_cb_wrapper_2.imageCallback)

# Retrieve the different gazebo objects
left_arm = baxter_interface.Limb('left')
right_arm = baxter_interface.Limb('right')

print("Initializing robot...")
# Init robot pose
resetPose()
print("Init Robot pose over")
end_point_position = baxter_utils.get_ee_position(left_arm)
ee_orientation = baxter_utils.get_ee_orientation(left_arm)

print('Starting up on port number {}'.format(SERVER_PORT))
context = zmq.Context()
socket = context.socket(zmq.PAIR)

socket.bind("tcp://*:{}".format(SERVER_PORT))

print("Waiting for client...")
socket.send_json({'msg': 'hello'})
print("Connected to client")

action = [0, 0, 0]
joints = None
episode_step = 0
episode_idx = -1
episode_folder = None

while not should_exit[0]:
    msg = socket.recv_json()
    command = msg.get('command', '')

    if command == 'reset':
        resetPose()
        end_point_position = baxter_utils.get_ee_position(left_arm)
        print('Environment reset')
        action = [0, 0, 0]
        episode_idx += 1
        episode_step = 0

        if SECOND_CAM_TOPIC is not None:
            episode_folder = "record_{:03d}".format(episode_idx)
            try:
                os.makedirs("srl_priors/data/{}/{}".format(DATA_FOLDER_SECOND_CAM, episode_folder))
            except OSError:
                pass

    elif command == 'action':
        action = np.array(msg['action'])
        print("action:", action)

    elif command == "exit":
        break
    else:
        raise ValueError("Unknown command: {}".format(msg))

    end_point_position_candidate = end_point_position + action

    print("End-effector Position:", end_point_position_candidate)
    joints = None
    try:
        joints = baxter_utils.IK(left_arm, end_point_position_candidate, ee_orientation)
    except Exception as e:
        print("[ERROR] no joints position returned by the Inverse Kinematic fn")
        print("end_point_position_candidate:{}".format(end_point_position_candidate))
        print(e)

    if joints:
        end_point_position = end_point_position_candidate
        left_arm.move_to_joint_positions(joints, timeout=3)
    else:
        print("No joints position, returning previous one")

    reward = 0
    # Consider that we touched the button if we are close enough
    if np.linalg.norm(BUTTON_POS - end_point_position, 2) < DIST_TO_TARGET_THRESHOLD:
        reward = 1
        print("Button touched!")

    # Send arm position, button position, ...
    socket.send_json(
        {
            # XYZ position
            "position": list(end_point_position),
            "reward": reward,
            "button_pos": list(BUTTON_POS)
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
