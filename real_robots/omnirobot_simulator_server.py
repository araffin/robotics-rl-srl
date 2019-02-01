from __future__ import division, print_function, absolute_import

import os
import signal
import time

# TODO !!!!!!!!!!!!!!!!!!!!!!!!!
# undistort origin image -> add undistort target marker -> redistort image


import numpy as np
import zmq
import cv2
import yaml
import argparse
#Konwn issue:
#- No module named 'scipy.spatial.transform'
#To resolve, try pip3 install scipy==1.2
from scipy.spatial.transform import Rotation as R

from .constants import *
from .utils import sendMatrix
from .omnirobot_simulator_utils import *


assert USING_OMNIROBOT_SIMULATOR, "Please set USING_OMNIROBOT_SIMULATOR to True in real_robots/constants.py"
should_exit = [False]

# exit the script on ctrl+c
#def ctrl_c(signum, frame):
#    should_exit[0] = True


#signal.signal(signal.SIGINT, ctrl_c)

class PosTransformer(object):
    def __init__(self, camera_mat:np.ndarray, dist_coeffs:np.ndarray,\
                 pos_camera_coord_ground:np.ndarray, rot_mat_camera_coord_ground:np.ndarray):
        """
       
        """
        super(PosTransformer, self).__init__()
        self.camera_mat = camera_mat
        
        self.dist_coeffs = dist_coeffs
        
        self.camera_2_ground_trans = np.zeros((4,4), np.float)
        self.camera_2_ground_trans[0:3,0:3] = rot_mat_camera_coord_ground
        self.camera_2_ground_trans[0:3,3] = pos_camera_coord_ground
        self.camera_2_ground_trans[3,3] = 1.0
        
        self.ground_2_camera_trans = np.linalg.inv(self.camera_2_ground_trans)


    def phyPosCam2PhyPosGround(self, pos_coord_cam):
        """
        Transform physical position in camera coordinate to physical position in ground coordinate
        """
        assert pos_coord_cam.shape == (3,1)
        homo_pos = np.ones((4,1), np.float32)
        homo_pos[0:3,:] = pos_coord_cam
        return (np.matmul(self.camera_2_ground_trans, homo_pos))[0:3,:]  
    
    def phyPosGround2PixelPos(self, pos_coord_ground, return_distort_image_pos=False):
        """
        Transform the physical position in ground coordinate to pixel position
        """
        assert pos_coord_ground.shape == (3,1) or pos_coord_ground.shape == (2,1)

        homo_pos = np.ones((4,1), np.float32)
        if pos_coord_ground.shape == (2,1):
            # by default, z =0 since it's on the ground
            homo_pos[0:2,:] = pos_coord_ground
            homo_pos[2,:] = 0.0
        else:
            homo_pos[0:3,:] = pos_coord_ground
        homo_pos = np.matmul(self.ground_2_camera_trans, homo_pos)
        pixel_points, _ =cv2.projectPoints(homo_pos[0:3,:].reshape(1,1,3), np.zeros((3,1)), np.zeros((3,1)),\
                                           self.camera_mat, self.dist_coeffs if return_distort_image_pos else None) 
        return (pixel_points.reshape((2,1)))


class OmniRobotSimulator(object):
    """
    Class for controlling Omnirobot, and interact with the simulator
    """
    def __init__(self, init_x, init_y, init_yaw, origin_size, cropped_size,\
                 back_ground_img, camera_info_path, \
                robot_marker_path, robot_marker_margin, target_marker_path, target_marker_margin,\
                robot_marker_code, target_marker_code,\
                robot_marker_length, target_marker_length):
        super(OmniRobotSimulator, self).__init__()

        # Initialize the direction
        self.init_pos = [init_x, init_y]
        self.init_yaw = init_yaw

        # OmniRobot's real position on the grid
        self.robot_pos = np.float32([0, 0])
        self.robot_yaw = 0 # in rad
        
        # OmniRobot's position command on the grid
        self.robot_pos_cmd = np.float32(self.init_pos[:])
        self.robot_yaw_cmd = self.init_yaw


        # Target's set position on the grid
        self.target_pos_cmd = np.float32([0,0])
        self.target_yaw_cmd = 0.0
        
        # Target's real position on the grid
        self.target_pos = np.float32([0,0])
        self.target_yaw = 0


        # status of moving
        self.move_finished = False
        self.target_pos_changed = False

        # Distance for each step
        self.step_distance = 0.07

        with open(camera_info_path, 'r') as stream:
            try:
                contents = yaml.load(stream)
                camera_matrix = np.array(contents['camera_matrix']['data'])
                self.origin_size = np.array([contents['image_height'], contents['image_width']])
                self.camera_matrix = np.reshape(camera_matrix,(3,3))
                self.dist_coeffs = np.array(contents["distortion_coefficients"]["data"]).reshape((1,5))
            except yaml.YAMLError as exc:
                print(exc)
        self.cropped_size = [np.min(self.origin_size), np.min(self.origin_size)] # size after being cropped
        
        # restore the image before being cropped
        self.bg_img = np.zeros([*self.origin_size,3], np.uint8)

        self.cropped_margin = (self.origin_size - self.cropped_size)/2.0
        self.cropped_range = np.array([self.cropped_margin[0],self.cropped_margin[0]+self.cropped_size[0],\
              self.cropped_margin[1],self.cropped_margin[1]+self.cropped_size[1]]).astype(np.int)
        
        if(back_ground_img.shape[0:2] != self.cropped_size):
            print("input back ground image's size: ", back_ground_img.shape)
            print("resize to ", self.cropped_size)
            self.bg_img[self.cropped_range[0]:self.cropped_range[1],self.cropped_range[2]:self.cropped_range[3],:] \
                = cv2.resize(back_ground_img, tuple(self.cropped_size)) # background image
        else:
            self.bg_img[self.cropped_range[0]:self.cropped_range[1],self.cropped_range[2]:self.cropped_range[3],:] \
                = back_ground_img # background image
                
        #self.bg_img = cv2.undistort(self.bg_img, self.camera_matrix, self.dist_coeffs)
        # Currently cannot find a solution to re-distort a image...
        
        self.target_bg_img = self.bg_img # background image with target.
        self.image = self.bg_img # image with robot and target
        
        

        # camera installation info
        camera_pos_coord_ground = [0, 0, 2.9]
        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        camera_rot_mat_coord_ground = r.as_dcm()

        self.pos_transformer = PosTransformer( self.camera_matrix, self.dist_coeffs,\
                                              camera_pos_coord_ground, camera_rot_mat_coord_ground)

        self.target_render = MarkerRender(noise_var=1.5)
        self.robot_render = MarkerRender(noise_var=1.5)
        self.robot_render.setMarkerImage(cv2.imread(robot_marker_path,cv2.IMREAD_COLOR), robot_marker_margin)
        self.target_render.setMarkerImage(cv2.imread(target_marker_path,cv2.IMREAD_COLOR), target_marker_margin)

        if robot_marker_code is not None and target_marker_code is not None:
            self.marker_finder = MakerFinder(camera_info_path)
            self.marker_finder.setMarkerCode('robot', robot_marker_code, robot_marker_length)
            self.marker_finder.setMarkerCode('target', target_marker_code, target_marker_length)        
        
    def renderTarget(self):
        """
        render the target
        """        
        #print("pixel pos: ",self.pos_transformer.phyPosGround2PixelPos(self.target_pos_cmd.reshape(2,1)))
        self.target_bg_img = self.target_render.addMarker(self.bg_img, \
                              self.pos_transformer.phyPosGround2PixelPos(self.target_pos_cmd.reshape(2,1)),\
                              self.target_yaw_cmd)

    def renderRobot(self):
        """
        render the image.
        """
        
        self.image = self.robot_render.addMarker(self.target_bg_img, \
                             self.pos_transformer.phyPosGround2PixelPos( self.robot_pos_cmd.reshape(2,1)),\
                             self.robot_yaw_cmd)
    def getCroppedImage(self):
        return self.image[self.cropped_range[0]:self.cropped_range[1],self.cropped_range[2]:self.cropped_range[3],:]
    def findMarkers(self):
        assert NotImplementedError
        # this is not tested
        tags_trans_coord_cam, tags_rot_coord_cam = self.marker_finder.getMarkerPose(self.image, ['robot','target'], True)
        if 'robot' in tags_trans_coord_cam.keys():
            self.robot_pos = self.pos_transformer.phyPosCam2PhyPosGround(tags_trans_coord_cam['robot'])
            tag_rot_coord_ground = np.matmul(self.pos_transformer.camera_2_ground_trans[0:3,0:3],tags_rot_coord_cam['robot'])[0:3,0:3]
            self.robot_yaw = R.from_dcm(tag_rot_coord_ground).as_euler('zyx', degree=False)
            print("robot_error: ". self.robot_pos - self.robot_pos_cmd)
            print("robot_yaw_error: ". self.robot_yaw - self.robot_yaw_cmd)
        if 'target' in tags_trans_coord_cam.keys():
            self.target_pos = self.pos_transformer.phyPosCam2PhyPosGround(tags_trans_coord_cam['target'])
            tag_rot_coord_ground = np.matmul(self.pos_transformer.camera_2_ground_trans[0:3,0:3],tags_rot_coord_cam['target'])[0:3,0:3]
            self.target_yaw = R.from_dcm(tag_rot_coord_ground).as_euler('zyx', degree=False)
            print("target_error: ", self.target_pos - self.target_pos_cmd)
            print("target_yaw_error: ", self.target_yaw - self.target_yaw_cmd)

       
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
        
        self.robot_pos = self.robot_pos_cmd + np.random.randn(2) * 0.03 # 0.03 m variance
        self.robot_yaw = self.robot_yaw_cmd + np.random.randn() * np.pi/180* 2 # 2 degree variance

    def setTargetCmd(self, x, y, yaw):
        self.target_pos_cmd[0] = x
        self.target_pos_cmd[1] = y
        self.target_yaw_cmd = self.normalizeAngle(yaw)

        self.target_pos = self.target_pos_cmd+ np.random.randn(2) * 0.03 # 0.03 m variance
        self.target_yaw = self.target_yaw_cmd+ np.random.randn() * np.pi/180* 2 # 2 degree variance
    def forward(self):
        """
        Move one step forward (Translation)
        """ 
        self.setRobotCmd(self.robot_pos_cmd[0] + self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)
        

    def backward(self):
        """
        Move one step backward
        """
        self.setRobotCmd(self.robot_pos_cmd[0] - self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)
        

    def left(self):
        """
        Translate in left
        """
        self.setRobotCmd(self.robot_pos_cmd[0] , self.robot_pos_cmd[1] +  self.step_distance, self.robot_yaw_cmd)
        
    def right(self):
        self.setRobotCmd(self.robot_pos_cmd[0] , self.robot_pos_cmd[1] -  self.step_distance, self.robot_yaw_cmd)
        

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Server for omnirobot simulator")
    parser.add_argument('--port', default=SERVER_PORT,help='socket port for omnirobot server',
                        type=int)
    parser.add_argument('--camera-info-path', default="real_robots/omnirobot_simulator_utils/cam_calib_info.yaml",type=str,
                        help="camera calibration file generated by ros calibrate")
    parser.add_argument('--robot-marker-path', default="real_robots/omnirobot_simulator_utils/robot_margin4_pixel.png", type=str,
                        help="robot marker's path")
    parser.add_argument('--target-marker-path', default="real_robots/omnirobot_simulator_utils/target_margin4_pixel.png", type=str,
                        help="target marker's path")
    parser.add_argument('--background-path', default="real_robots/omnirobot_simulator_utils/back_ground.jpg", type=str,
                        help="target marker's path")
    parser.add_argument('--robot-marker-margin', default=[4,4,4,4], type=int, nargs='+',
                        help="robot marker's margin in pixel")
    parser.add_argument('--target-marker-margin', default=[4,4,4,4], type=int, nargs='+',
                        help="target marker's margin in pixel")
    parser.add_argument('--output-size', default=[224, 224], type=int, nargs='+', help="output size of each frame, default [224,224]")
    args = parser.parse_args()

    assert len(args.robot_marker_margin) == 4
    assert len(args.target_marker_margin) == 4
    assert len(args.output_size) == 2
    

    print('Starting up on port number {}'.format(args.port))
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    socket.bind("tcp://*:{}".format(args.port))

    print("Waiting for client...")
    socket.send_json({'msg': 'hello'})
    print("Connected to client on port {}".format(args.port))

    action = 0
    episode_step = 0
    episode_idx = -1
    episode_folder = None
    
    back_ground_img = cv2.imread(args.background_path)
    
    omni_robot = OmniRobotSimulator(0, 0, 0, [640,480],[480,480],back_ground_img=back_ground_img, camera_info_path=args.camera_info_path,\
                                robot_marker_path=args.robot_marker_path, robot_marker_margin=args.robot_marker_margin,\
                                target_marker_path=args.target_marker_path, target_marker_margin=args.target_marker_margin,\
                                robot_marker_code=None, target_marker_code=None,\
                                robot_marker_length=0.18, target_marker_length=0.18) # yaw is in rad
    # target reset
    random_init_x = np.random.random_sample() * (MAX_X -MIN_X) + MIN_X
    random_init_y = np.random.random_sample() * (MAX_Y - MIN_Y) + MIN_Y
    print(random_init_x,random_init_y )
    omni_robot.setTargetCmd(random_init_x, random_init_y, 0)
    
    # render the target
    omni_robot.renderTarget()
    print("initial target position: ", omni_robot.target_pos)
    while not should_exit[0]:
        print("wait for new command")
        msg = socket.recv_json()

        print("msg: {}".format(msg))
        command = msg.get('command', '')

        if command == 'reset':
            print('Environment reset, choose random position')
            action = None
            episode_idx += 1
            episode_step = 0

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
            pass
        elif action == Move.RIGHT:
            if omni_robot.robot_pos[1] > MIN_Y:
                omni_robot.right()
            else:
                has_bumped = True
        elif action == Move.LEFT:
            if omni_robot.robot_pos[1] < MAX_Y:
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
            
            # target reset
            random_init_x = np.random.random_sample() * (TARGET_MAX_X -TARGET_MIN_X) + TARGET_MIN_X
            random_init_y = np.random.random_sample() * (TARGET_MAX_Y - TARGET_MIN_Y) + TARGET_MIN_Y
            omni_robot.setTargetCmd(random_init_x, random_init_y, 0)
            print("new target position: {:.4f} {:4f}".format(omni_robot.target_pos[0],omni_robot.target_pos[1]))

            # render the target
            omni_robot.renderTarget()
        else:
            print("Unsupported action")


        omni_robot.renderRobot()
        original_image = np.copy(omni_robot.getCroppedImage())
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, tuple(args.output_size))
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
                "position": omni_robot.robot_pos.tolist(),
                "reward": reward,
                "target_pos": omni_robot.target_pos.tolist()
            },
            flags=zmq.SNDMORE
        )
    
        # to contiguous, otherwise ZMQ will complain
        img = np.ascontiguousarray(original_image, dtype=np.uint8)
        sendMatrix(socket, img)

    print("Exiting server - closing socket...")
    socket.close()
