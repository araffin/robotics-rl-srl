#!/usr/bin/env python

from __future__ import division, print_function, absolute_import
import rospy

import os
import signal
import time


import numpy as np
import zmq
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, PoseStamped
from std_msgs.msg import Bool


from tf.transformations import euler_from_quaternion

from constants import *
from utils import sendMatrix

assert USING_OMNIROBOT, "Please set USING_OMNIROBOT to True in real_robots/constants.py"

bridge = CvBridge()
should_exit = [False]

# exit the script on ctrl+c
def ctrl_c(signum, frame):
    should_exit[0] = True


signal.signal(signal.SIGINT, ctrl_c)


class OmniRobot(object):
    """
    Class for controlling Robobo
    """

    def __init__(self, init_x, init_y, init_yaw):
        super(OmniRobot, self).__init__()

        # Initialize the direction
        self.init_pos = [init_x, init_y]
        self.init_yaw = init_yaw

        # OmniRobot's real position on the grid
        self.robot_pos = [0, 0]
        self.robot_yaw = 0 # in rad
        
        # OmniRobot's position command on the grid
        self.robot_pos_cmd = self.init_pos[:]
        self.robot_yaw_cmd = self.init_yaw


        # Target's real position on the grid
        self.target_pos = [0,0]
        self.target_yaw = 0


        # status of moving
        self.move_finished = False
        self.target_pos_changed = False

        # Distance for each step
        self.step_distance = 0.07

        self.visual_robot_sub = rospy.Subscriber("/visual_robot_pose", PoseStamped, self.visualRobotCallback, queue_size=10)
        self.visual_target_sub = rospy.Subscriber("/visual_target_pose", PoseStamped, self.visualTargetCallback, queue_size=10)
         
        self.pos_cmd_pub = rospy.Publisher("/position_commands", Vector3, queue_size=10)
        self.move_finished_sub = rospy.Subscriber("/finished", Bool, self.moveFinishedCallback, queue_size=10)
        self.stop_signal_pub = rospy.Publisher("/stop", Bool, queue_size=10)
        self.reset_odom_pub = rospy.Publisher("/reset_odom", Vector3, queue_size=10)
        self.reset_signal_pub = rospy.Publisher("/reset", Bool, queue_size=10)


        rospy.sleep(1) #known issues, without sleep 1 second, publishers could not been setup
                        #https://answers.ros.org/question/9665/test-for-when-a-rospy-publisher-become-available/?answer=14125#post-id-14125

    def setRobotCmdConstrained(self, x, y, yaw):
        self.robot_pos_cmd[0] = max(x, MIN_X)
        self.robot_pos_cmd[0] = min(x, MAX_X)

        self.robot_pos_cmd[1] = max(y, MIN_Y)
        self.robot_pos_cmd[1] = min(y, MAX_Y)
        self.robot_yaw_cmd = self.normalizeAngle(yaw)
    def setRobotCmd(self, x, y, yaw):
        self.robot_pos_cmd[0] = x
        self.robot_pos_cmd[1] = y
        self.robot_yaw_cmd = self.normalizeAngle(yaw)
        
    def pubPosCmd(self):
        """
        Publish the position command for the robot
        x, y, yaw are in the global frame
        """
        msg = Vector3(self.robot_pos_cmd[0], self.robot_pos_cmd[1], self.robot_yaw_cmd)
        self.pos_cmd_pub.publish(msg)
        self.move_finished = False
        print("move to x: {:.4f} y:{:.4f} yaw: {:.4f}".format(msg.x, msg.y, msg.z))

    def resetOdom(self, x, y, yaw):
        """
        The odometry of robot will be reset to x, y, yaw (in global frame)
        """
        msg = Vector3()
        msg.x = x
        msg.y = y
        msg.z = yaw
        self.reset_odom_pub.publish(msg)
    
    def reset(self):
        """
        Publish the reset signal to robot (quit the stop state)
        The odometry will not be reset automatically
        """
        msg = Bool()
        self.reset_signal_pub.publish(msg)

    def stop(self):
        """
        Publish the stop signal to robot
        """
        msg = Bool()
        msg.data = True
        self.stop_signal_pub.publish(msg)
        self.move_finished = True

    def visualRobotCallback(self, pose_stamped_msg):
        """
        Callback for ROS topic
        Get the new updated position of robot from camera
        :param pose_stamped_msg: (PoseStamped ROS message)
        """
        self.robot_pos[0] = pose_stamped_msg.pose.position.x
        self.robot_pos[1] = pose_stamped_msg.pose.position.y
        self.robot_yaw = euler_from_quaternion([pose_stamped_msg.pose.orientation.x, pose_stamped_msg.pose.orientation.y,
                                                pose_stamped_msg.pose.orientation.z, pose_stamped_msg.pose.orientation.w])[2]

    def visualTargetCallback(self, pose_stamped_msg):
        """
        Callback for ROS topic
        Get the new updated position of robot from camera
        :param pose_stamped_msg: (PoseStamped ROS message)
        """
        if self.target_pos_changed and self.move_finished:
            self.target_pos[0] = pose_stamped_msg.pose.position.x
            self.target_pos[1] = pose_stamped_msg.pose.position.y
            self.target_yaw = euler_from_quaternion([pose_stamped_msg.pose.orientation.x, pose_stamped_msg.pose.orientation.y,
                                                    pose_stamped_msg.pose.orientation.z, pose_stamped_msg.pose.orientation.w])[2]
            self.target_pos_changed = False


    
    def moveFinishedCallback(self, move_finished_msg):
        """
        Callback for ROS topic
        receive 'finished' signal when robot moves to the target
        """
        self.move_finished = move_finished_msg.data
        
        
    def forward(self):
        """
        Move one step forward (Translation)
        """ 
        self.setRobotCmd(self.robot_pos_cmd[0] + self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)
        self.pubPosCmd()

    def backward(self):
        """
        Move one step backward
        """
        self.setRobotCmd(self.robot_pos_cmd[0] - self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)
        self.pubPosCmd()

    def left(self):
        """
        Translate in left
        """
        self.setRobotCmd(self.robot_pos_cmd[0] , self.robot_pos_cmd[1] +  self.step_distance, self.robot_yaw_cmd)
        self.pubPosCmd()
    def right(self):
        self.setRobotCmd(self.robot_pos_cmd[0] , self.robot_pos_cmd[1] -  self.step_distance, self.robot_yaw_cmd)
        self.pubPosCmd()

    @staticmethod
    def normalizeAngle(angle):
        """
        :param angle: (float) (in rad)
        :return: (float) the angle in [-pi, pi] (in rad)
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


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


def saveSecondCamImage(im, episode_folder, episode_step, path="omnirobot_2nd_cam"):
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



if __name__ == '__main__':

    rospy.init_node('omni_robot_server', anonymous=True)

    # Connect to ROS Topics
    if IMAGE_TOPIC is not None:
        image_cb_wrapper = ImageCallback()
        img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, image_cb_wrapper.imageCallback, queue_size=1)

    if SECOND_CAM_TOPIC is not None:
        assert NotImplementedError
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

    omni_robot = OmniRobot(0, 0, 0) # yaw is in rad
    omni_robot.stop() # after stop, the robot need to be reset
    omni_robot.resetOdom(0, 0, 0)
    omni_robot.reset()


    omni_robot.pubPosCmd()
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        print("wait for new command")
        msg = socket.recv_json()

        print("msg: {}".format(msg))
        command = msg.get('command', '')

        if command == 'reset':
            print('Environment reset, choose random position')
            action = None
            episode_idx += 1
            episode_step = 0
            omni_robot.reset()
            if SECOND_CAM_TOPIC is not None:
                assert NotImplementedError
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
            if omni_robot.robot_pos[0] < MAX_X:
                omni_robot.forward()
            else:
                has_bumped = True
        elif action == Move.STOP:
            omni_robot.stop()
        elif action == Move.RIGHT:
            if omni_robot.robot_pos[1] < MAX_Y:
                omni_robot.right()
            else:
                has_bumped = True
        elif action == Move.LEFT:
            if omni_robot.robot_pos[1] > MIN_Y:
                omni_robot.left()
            else:
                has_bumped = True
        elif action == Move.BACKWARD:
            if omni_robot.robot_pos[0] > MIN_X:
                omni_robot.backward()
            else:
                has_bumped = True
        elif action is None:
            # Env reset
            random_init_x = np.random.random_sample() * (MAX_X -MIN_X) + MIN_X
            random_init_y = np.random.random_sample() * (MAX_Y - MIN_Y) + MIN_Y
            
            omni_robot.setRobotCmd(random_init_x, random_init_y, 0)
            omni_robot.target_pos_changed = True
            omni_robot.pubPosCmd()
            
        else:
            print("Unsupported action")

        # wait for robot to finish the action, timeout 20s
        timeout = 20 # second
        for i in range(timeout):
            readable_list, _, _ = zmq.select([socket], [], [], 0)

            if len(readable_list) > 0:
                print("New command incomes, ignore the current command")
                continue
            if omni_robot.move_finished:
                print("action done")
                break
            
            elif i == timeout -1:
                print("Error: timeout for action finished signal")  
                exit()
            time.sleep(1)
            


        if IMAGE_TOPIC is not None:
            # Retrieve last image from image topic
            original_image = np.copy(image_cb_wrapper.valid_img)

        reward = 0
        # Consider that we reached the target if we are close enough
        # we detect that computing the difference in area between TARGET_INITIAL_AREA
        # current detected area of the target
        if np.linalg.norm(np.array(omni_robot.robot_pos) - np.array(omni_robot.target_pos)) <  DIST_TO_TARGET_THRESHOLD:
            reward = 1
            print("Target reached!")

        if has_bumped:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            reward = -1
            print("Bumped into wall")
            print()
        print("reward: {}".format(reward))

        print("omni_robot position", omni_robot.robot_pos)
        print("target position", omni_robot.target_pos)
        socket.send_json(
            {
                # XYZ position
                "position": omni_robot.robot_pos,
                "reward": reward,
                "target_pos": omni_robot.target_pos
            },
            flags=zmq.SNDMORE if IMAGE_TOPIC is not None else 0
        )

        if SECOND_CAM_TOPIC is not None:
            saveSecondCamImage(image_cb_wrapper_2.valid_img, episode_folder, episode_step, DATA_FOLDER_SECOND_CAM)
            episode_step += 1

        if IMAGE_TOPIC is not None:
            # to contiguous, otherwise ZMQ will complain
            img = np.ascontiguousarray(original_image, dtype=np.uint8)
            sendMatrix(socket, img)
        r.sleep()

    print("Exiting server - closing socket...")
    socket.close()
