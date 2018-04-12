#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import subprocess
import signal

import arm_scenario_simulator as arm_sim
import baxter_interface
import numpy as np
import rospy
import zmq
from baxter_interface import Limb, Head, Gripper, CHECK_VERSION
from arm_scenario_experiments import baxter_utils
from arm_scenario_experiments import utils as arm_utils
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Vector3, Vector3Stamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .constants import DELTA_POS, SERVER_PORT, IMAGE_TOPIC, \
    ACTION_TOPIC
from .utils import sendMatrix, getActions

REF_POINT_LEFT_ARM = [ 0.69850099,  0.14505832,  0.08032852]
LEFT_ARM_ORIENTATION = [ 0.99893116, -0.04207143, -0.00574656, -0.01826233]

BUTTON_POS = [ 0.7090276,   0.13833109, -0.11170768]

IK_SEED_POSITIONS = None

bridge = CvBridge()

should_exit = [False]

# exit the script on ctrl+c
def ctrl_c(signum, frame):
    should_exit[0] = True

signal.signal(signal.SIGINT, ctrl_c)


def resetPose():
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    if rs.state().enabled:
        print("Robot already enabled")
    else:
        print("Enabling robot... ")
        rs.enable()
        # Untuck arms
        subprocess.call(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])
    print("Moving left arm to init")
    move_left_arm_to_init()

class ImageCallback(object):
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


def move_left_arm_to_init():
    """
    Initialize robot left arm to starting position (hardcoded)
    :return: ([float])
    """
    joints = None
    position = REF_POINT_LEFT_ARM
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


rospy.init_node('real_baxter_server', anonymous=True)

# Connect to ROS Topics
image_cb_wrapper = ImageCallback()
img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback)
action_pub = rospy.Publisher(ACTION_TOPIC, Vector3Stamped, queue_size=1)

# Retrieve the different gazebo objects
left_arm = baxter_interface.Limb('left')
right_arm = baxter_interface.Limb('right')
ee_orientation = baxter_utils.get_ee_orientation(left_arm)

# baxter_position = arm_utils.point2array(baxter_pose.position)
# baxter_orientation = arm_utils.quat2array(baxter_pose.orientation)

# ===== Get list of allowed actions ====
possible_actions = getActions(DELTA_POS, n_actions=6)
rospy.sleep(1)

print("Initializing robot...")
# Init robot pose
resetPose()
print("Init Robot pose over")
end_point_position = baxter_utils.get_ee_position(left_arm)

print('Starting up on port number {}'.format(SERVER_PORT))
context = zmq.Context()
socket = context.socket(zmq.PAIR)

socket.bind("tcp://*:{}".format(SERVER_PORT))

print("Waiting for client...")
socket.send_json({'msg': 'hello'})
print("Connected to client")

action = [0, 0, 0]
joints = None

while not should_exit[0]:
    msg = socket.recv_json()
    command = msg.get('command', '')
    if command == 'reset':
        resetPose()
        end_point_position = baxter_utils.get_ee_position(left_arm)
        print('Environment reset')
        action = [0, 0, 0]

    elif command == 'action':
        action = np.array(msg['action'])
        print("action:", action)

    elif command == "exit":
        break
    else:
        raise ValueError("Unknown command: {}".format(msg))

    # action = randomAction(possible_actions)
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
        action_pub.publish(Vector3Stamped(Header(stamp=rospy.Time.now()), Vector3(*action)))
        end_point_position = end_point_position_candidate
        left_arm.move_to_joint_positions(joints, timeout=3)
    else:
        print("No joints position, returning previous one")

    reward = 0
    if np.linalg.norm(BUTTON_POS - end_point_position, 2) < 0.035:
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

    img = image_cb_wrapper.valid_img
    # to contiguous, otherwise ZMQ will complain
    img = np.ascontiguousarray(img, dtype=np.uint8)
    sendMatrix(socket, img)


# TODO:  avoid socket pid running and 'Address already in use' error relaunching, this is not enough
print(" Exiting server - closing socket...")
socket.close()
