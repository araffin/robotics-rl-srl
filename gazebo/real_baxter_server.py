#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import subprocess

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

REF_POINT_LEFT_ARM = [0.6, 0.30, 0.20]  # TODO: Calibrate
BUTTON_POS = [0, 0, 0]  # TODO: Calibrate
# Are they the right ones ?
# ['left_e0', 'left_e1', 'left_s0', 'left_s1', 'left_w0', 'left_w1', 'left_w2']
IK_SEED_POSITIONS = [-1.535, 1.491, -0.038, 0.194, 1.546, 1.497, -0.520]

bridge = CvBridge()


def resetPose():
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    if rs.state().enabled:
        print("Robot already enabled")
    else:
        print("Enabling robot... ")
        rs.enable()
        # Untuck arms
        subprocess.call(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])
        # TODO: use IK to reset to fixed pose
        # or calibrate:
        # head = Head()
        # left_arm = Limb('left')
        # right_arm = Limb('right')
        # grip_left = Gripper('left', CHECK_VERSION)
        # grip_right = Gripper('right', CHECK_VERSION)
        #
        # names = ['head_pan', 'l_gripper_l_finger_joint', 'l_gripper_r_finger_joint',
        #         'left_e0', 'left_e1', 'left_s0',
        #         'left_s1', 'left_w0', 'left_w1',
        #         'left_w2', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint',
        #         'right_e0', 'right_e1', 'right_s0',
        #         'right_s1', 'right_w0', 'right_w1', 'right_w2']
        #
        # positions = [1.9175123711079323e-09, 3.0089229734974557e-05, 1.1656136997545379e-08,
        #             -1.5557922490972862, 1.4869254432037105, 0.2966753816741825,
        #             -0.043254170670461, 1.4459875320633593, 1.4934273103021356,
        #             -0.5197388002153112,0.020833031933134405, 3.920833833842966e-08,
        #             1.1897546738059388, 1.9397502577790355, -1.25925592718432,
        #             -0.9998100343641312, -0.6698868022939237, 1.029853661574463, 0.4999199143249742]
        #
        # positions_dico = {names[i]: positions[i] for i in range(len(names))}
        #
        # left_arm.move_to_joint_positions({joint: positions_dico[joint] for joint in left_arm.joint_names()})
        # right_arm.move_to_joint_positions({joint: positions_dico[joint] for joint in right_arm.joint_names()})
        # grip_left.close()
        # head.set_pan(0)


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
            joints = baxter_utils.IK(left_arm, position, ee_orientation, IK_SEED_POSITIONS)
        except Exception:
            try:
                joints = baxter_utils.IK(left_arm, position, ee_orientation, IK_SEED_POSITIONS)
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
# end_point_position = move_left_arm_to_init()

print('Starting up on port number {}'.format(SERVER_PORT))
context = zmq.Context()
socket = context.socket(zmq.PAIR)

socket.bind("tcp://*:{}".format(SERVER_PORT))

print("Waiting for client...")
socket.send_json({'msg': 'hello'})
print("Connected to client")

action = [0, 0, 0]
joints = None

try:
    while True:
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

        # Send arm position, button position, ...
        socket.send_json(
            {
                # XYZ position
                "position": list(end_point_position),
                "reward": 0,
                "button_pos": list(BUTTON_POS)
            },
            flags=zmq.SNDMORE
        )

        img = image_cb_wrapper.valid_img
        # to contiguous, otherwise ZMQ will complain
        img = np.ascontiguousarray(img, dtype=np.uint8)
        sendMatrix(socket, img)
except KeyboardInterrupt:
    pass

# TODO:  avoid socket pid running and 'Address already in use' error relaunching, this is not enough
print(" Exiting server - closing socket...")
socket.close()
