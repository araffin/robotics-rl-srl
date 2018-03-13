#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

import subprocess

import arm_scenario_simulator as arm_sim
import baxter_interface
import numpy as np
import rospy
import zmq
from arm_scenario_experiments import baxter_utils
from arm_scenario_experiments import utils as arm_utils
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Vector3, Vector3Stamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .constants import DELTA_POS, SERVER_PORT, IK_SEED_POSITIONS, REF_POINT, IMAGE_TOPIC, \
    ACTION_TOPIC, BUTTON_POS_TOPIC
from .utils import sendMatrix, getActions

bridge = CvBridge()


class ImageCallback(object):
    def __init__(self):
        super(ImageCallback, self).__init__()
        self.valid_img = None

    def imageCallback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = cv2_img
        except CvBridgeError as e:
            print("CvBridgeError:", e)


def move_left_arm_to_init():
    """
    Initialize robot left arm to starting position (hardcoded)
    :return: ([float])
    """
    joints = None
    position = REF_POINT
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


rospy.init_node('gym_gazebo_server', anonymous=True)

# Connect to ROS Topics
image_cb_wrapper = ImageCallback()
img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback)
action_pub = rospy.Publisher(ACTION_TOPIC, Vector3Stamped, queue_size=1)
button_pos_pub = rospy.Publisher(BUTTON_POS_TOPIC, Point, queue_size=1)

# Retrieve the different gazebo objects
left_arm = baxter_interface.Limb('left')
right_arm = baxter_interface.Limb('right')
ee_orientation = baxter_utils.get_ee_orientation(left_arm)
lever = arm_sim.Lever('lever1')
button = arm_sim.Button('button1')

# ===== Get list of allowed actions ====
possible_actions = getActions(DELTA_POS, n_actions=6)
rospy.sleep(1)

print("Initializing robot...")
# Init robot pose
subprocess.call(["rosrun", "arm_scenario_experiments", "button_init_pose"])
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
            subprocess.call(["rosrun", "arm_scenario_experiments", "button_init_pose"])
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

        # Get current position of the button
        button_pos = button.get_state().pose.position
        button_pos_absolute = arm_utils.point2array(button_pos)

        # Send arm position, button position, ...
        socket.send_json(
            {
                # XYZ position
                "position": list(end_point_position),
                "reward": int(button.is_pressed()),
                "button_pos": list(button_pos_absolute)
            },
            flags=zmq.SNDMORE
        )

        img = image_cb_wrapper.valid_img
        # to contiguous, otherwise ZMQ will complain
        img = np.ascontiguousarray(img, dtype=np.uint8)
        sendMatrix(socket, img)
except KeyboardInterrupt:
    print("Server Exiting...")
    socket.close()

# TODO:  avoid socket pid running and 'Address already in use' error relaunching, this is not enough
print("Server Exiting...")
socket.close()
