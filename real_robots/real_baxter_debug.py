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

from .constants import *

assert USING_REAL_BAXTER, "Please set USING_REAL_BAXTER to True in real_robots/constants.py"

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


rospy.init_node('real_baxter_server', anonymous=True)

# Retrieve the different gazebo objects
left_arm = baxter_interface.Limb('left')
right_arm = baxter_interface.Limb('right')
ee_orientation = baxter_utils.get_ee_orientation(left_arm)

rospy.sleep(1)

print("Initializing robot...")
# Init robot pose
resetPose()
print("Init Robot pose over")

try:
    while not should_exit[0]:
        for name, arm in zip(["left_arm", "right_arm"], [left_arm, right_arm]):
            arm_pos = baxter_utils.get_ee_position(arm)
            if name == "left_arm":
                ee_orientation = baxter_utils.get_ee_orientation(left_arm)
                # print(ee_orientation)
                # print(name, arm_pos)
                print(np.linalg.norm(BUTTON_POS - arm_pos, 2))
        rospy.sleep(0.5)
except KeyboardInterrupt:
    print("Exiting....")
