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

from .constants import IMAGE_TOPIC, ACTION_TOPIC


def resetPose():
    rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
    if rs.state().enabled:
        print("Robot already enabled")
    else:
        print("Enabling robot... ")
        rs.enable()
        # Untuck arms
        subprocess.call(['rosrun', 'baxter_tools', 'tuck_arms.py', '-u'])


rospy.init_node('real_baxter_server', anonymous=True)

# Connect to ROS Topics
# image_cb_wrapper = ImageCallback()
# img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback)
action_pub = rospy.Publisher(ACTION_TOPIC, Vector3Stamped, queue_size=1)

# Retrieve the different gazebo objects
left_arm = baxter_interface.Limb('left')
right_arm = baxter_interface.Limb('right')
ee_orientation = baxter_utils.get_ee_orientation(left_arm)

# baxter_position = arm_utils.point2array(baxter_pose.position)
# baxter_orientation = arm_utils.quat2array(baxter_pose.orientation)

rospy.sleep(1)

print("Initializing robot...")
# Init robot pose
resetPose()
print("Init Robot pose over")
try:
    while True:
        for name, arm in zip(["left_arm", "right_arm"], [left_arm, right_arm]):
            arm_pos = baxter_utils.get_ee_position(arm)
            print(name, arm_pos)
except KeyboardInterrupt:
    print("Exiting....")
