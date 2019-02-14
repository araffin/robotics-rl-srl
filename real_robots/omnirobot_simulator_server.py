from __future__ import division, print_function, absolute_import

from multiprocessing import Process, Pipe


# TODO !!!!!!!!!!!!!!!!!!!!!!!!!
# undistort origin image -> add undistort target marker -> redistort image

# Konwn issue: - No module named 'scipy.spatial.transform', To resolve, try pip3 install scipy==1.2
from scipy.spatial.transform import Rotation as R

from .constants import *
from .omnirobot_simulator_utils import *


assert USING_OMNIROBOT_SIMULATOR, "Please set USING_OMNIROBOT_SIMULATOR to True in real_robots/constants.py"
NOISE_VAR_ROBOT_POS = 0.01 #meter
NOISE_VAR_ROBOT_YAW = np.pi/180* 2.5 # 5 Deg
NOISE_VAR_TARGET_PIXEL = 2 # pixel noise on target marker
NOISE_VAR_ROBOT_PIXEL = 2
NOISE_VAR_ENVIRONMENT = 0.03 # pixel noise of final image on LAB space
NOISE_VAR_ROBOT_SIZE_PROPOTION = 0.05 # noise of robot size propotion
NOISE_VAR_TARGET_SIZE_PROPOTION = 0.05 
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
            homo_pos[2,:] = 0 #(np.random.randn() - 0.5) * 0.05 # add noise to the z-axis
        else:
            homo_pos[0:3,:] = pos_coord_ground
        homo_pos = np.matmul(self.ground_2_camera_trans, homo_pos)
        pixel_points, _ = cv2.projectPoints(homo_pos[0:3,:].reshape(1,1,3), np.zeros((3,1)), np.zeros((3,1)),
                                            self.camera_mat, self.dist_coeffs if return_distort_image_pos else None)
        return pixel_points.reshape((2,1))


class OmniRobotEnvRender():
    """
    Class for rendering Omnirobot environment
    """
    def __init__(self, init_x, init_y, init_yaw, origin_size, cropped_size,
                 back_ground_path, camera_info_path,
                 robot_marker_path, robot_marker_margin, target_marker_path, target_marker_margin,
                 robot_marker_code, target_marker_code,
                 robot_marker_length, target_marker_length, output_size,**_):
        super(OmniRobotEnvRender, self).__init__()

        self.output_size = output_size

        # store the size of robot marker
        self.robot_marker_size_proprotion = 1.0

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
        self.cropped_range = np.array([self.cropped_margin[0],self.cropped_margin[0]+self.cropped_size[0],
                                       self.cropped_margin[1],
                                       self.cropped_margin[1]+self.cropped_size[1]]).astype(np.int)
        
        back_ground_img = cv2.imread(back_ground_path)
        if(back_ground_img.shape[0:2] != self.cropped_size):
            print("input back ground image's size: ", back_ground_img.shape)
            print("resize to ", self.cropped_size)
            self.bg_img[self.cropped_range[0]:self.cropped_range[1],self.cropped_range[2]:self.cropped_range[3], :] \
                = cv2.resize(back_ground_img, tuple(self.cropped_size)) # background image
        else:
            self.bg_img[self.cropped_range[0]:self.cropped_range[1],self.cropped_range[2]:self.cropped_range[3], :] \
                = back_ground_img # background image
                
        self.bg_img = cv2.undistort(self.bg_img, self.camera_matrix, self.dist_coeffs)
        #Currently cannot find a solution to re-distort a image...
        
        self.target_bg_img = self.bg_img  # background image with target.
        self.image = self.bg_img  # image with robot and target
        
        

        # camera installation info
        camera_pos_coord_ground = [0, 0, 2.9]
        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        camera_rot_mat_coord_ground = r.as_dcm()

        self.pos_transformer = PosTransformer( self.camera_matrix, self.dist_coeffs,
                                               camera_pos_coord_ground, camera_rot_mat_coord_ground)

        self.target_render = MarkerRender(noise_var=NOISE_VAR_TARGET_PIXEL)
        self.robot_render = MarkerRender(noise_var=NOISE_VAR_ROBOT_PIXEL)
        self.robot_render.setMarkerImage(cv2.imread(robot_marker_path,cv2.IMREAD_COLOR), robot_marker_margin)
        self.target_render.setMarkerImage(cv2.imread(target_marker_path,cv2.IMREAD_COLOR), target_marker_margin)

        if robot_marker_code is not None and target_marker_code is not None:
            self.marker_finder = MakerFinder(camera_info_path)
            self.marker_finder.setMarkerCode('robot', robot_marker_code, robot_marker_length)
            self.marker_finder.setMarkerCode('target', target_marker_code, target_marker_length)        



    def renderEnvLuminosityNoise(self, origin_image, noise_var=0.1, in_RGB=False, out_RGB=False):
        """
        render the different environment luminosity
        """
        # variate luminosity and color
        origin_image_LAB = cv2.cvtColor(origin_image, cv2.COLOR_RGB2LAB if in_RGB else cv2.COLOR_BGR2LAB,cv2.CV_32F)
        origin_image_LAB[:,:,0] = origin_image_LAB[:,:,0] * (np.random.randn() * noise_var + 1.0)
        origin_image_LAB[:,:,1] = origin_image_LAB[:,:,1] * (np.random.randn() * noise_var + 1.0)
        origin_image_LAB[:,:,2] = origin_image_LAB[:,:,2] * (np.random.randn() * noise_var + 1.0)
        out_image = cv2.cvtColor(origin_image_LAB, cv2.COLOR_LAB2RGB if out_RGB else cv2.COLOR_LAB2BGR, cv2.CV_8UC3)
        return out_image

    def renderTarget(self):
        """
        render the target
        """        
        self.target_bg_img = self.target_render.addMarker(self.bg_img,
                                                          self.pos_transformer.phyPosGround2PixelPos(
                                                              self.target_pos.reshape(2,1)),
                                                          self.target_yaw, np.random.randn() * NOISE_VAR_TARGET_SIZE_PROPOTION + 1.0)

    def renderRobot(self):
        """
        render the image.
        """
        self.image = self.robot_render.addMarker(self.target_bg_img,
                                                 self.pos_transformer.phyPosGround2PixelPos(
                                                     self.robot_pos.reshape(2,1)),
                                                 self.robot_yaw, self.robot_marker_size_proprotion)
        
    def getCroppedImage(self):
        return self.image[self.cropped_range[0]:self.cropped_range[1], self.cropped_range[2]:self.cropped_range[3], :]

    def findMarkers(self):
        assert NotImplementedError
        # this is not tested
        tags_trans_coord_cam, tags_rot_coord_cam = self.marker_finder.getMarkerPose(self.image, ['robot','target'], True)
        if 'robot' in tags_trans_coord_cam.keys():
            self.robot_pos = self.pos_transformer.phyPosCam2PhyPosGround(tags_trans_coord_cam['robot'])
            tag_rot_coord_ground = np.matmul(
                self.pos_transformer.camera_2_ground_trans[0:3,0:3],tags_rot_coord_cam['robot'])[0:3, 0:3]
            self.robot_yaw = R.from_dcm(tag_rot_coord_ground).as_euler('zyx', degree=False)
            print("robot_error: ". self.robot_pos - self.robot_pos_cmd)
            print("robot_yaw_error: ". self.robot_yaw - self.robot_yaw_cmd)

        if 'target' in tags_trans_coord_cam.keys():
            self.target_pos = self.pos_transformer.phyPosCam2PhyPosGround(tags_trans_coord_cam['target'])
            tag_rot_coord_ground = np.matmul(self.pos_transformer.camera_2_ground_trans[0:3, 0:3],
                                             tags_rot_coord_cam['target'])[0:3, 0:3]
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
        
        self.robot_pos = self.robot_pos_cmd  + np.random.randn(2) * NOISE_VAR_ROBOT_POS  # add noise
        self.robot_yaw = self.normalizeAngle(self.robot_yaw_cmd + np.random.randn() * NOISE_VAR_ROBOT_YAW )  #add noise

    def setTargetCmd(self, x, y, yaw):
        self.target_pos_cmd[0] = x
        self.target_pos_cmd[1] = y
        self.target_yaw_cmd = self.normalizeAngle(yaw)

        self.target_pos = self.target_pos_cmd
        self.target_yaw = self.normalizeAngle(self.target_yaw_cmd)

    def forward(self, action=None):
        """
        Move one step forward (Translation)
        """
        self.setRobotCmd(self.robot_pos_cmd[0] + self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)

    def backward(self, action=None):
        """
        Move one step backward
        """
        self.setRobotCmd(self.robot_pos_cmd[0] - self.step_distance, self.robot_pos_cmd[1], self.robot_yaw_cmd)

    def left(self, action=None):
        """
        Translate to the left
        """
        self.setRobotCmd(self.robot_pos_cmd[0] , self.robot_pos_cmd[1] + self.step_distance, self.robot_yaw_cmd)
        
    def right(self, action=None):
        """
        Translate to the right
        """
        self.setRobotCmd(self.robot_pos_cmd[0] , self.robot_pos_cmd[1] - self.step_distance, self.robot_yaw_cmd)
    
    def moveContinous(self, action):
        """
        Perform a continuous displacement of dx, dy
        """
        self.setRobotCmd(self.robot_pos_cmd[0] + action[0], self.robot_pos_cmd[1] + action[1], self.robot_yaw_cmd)

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


class OmniRobotSimulatorSocket():
    def __init__(self, **args):
        '''
        Simulate the zmq socket like real omnirobot server 
        :param **args  arguments 
        
        '''
        defalt_args = {
            "back_ground_path":"real_robots/omnirobot_simulator_utils/back_ground.jpg", 
            "camera_info_path":"real_robots/omnirobot_simulator_utils/cam_calib_info.yaml",
            "robot_marker_path":"real_robots/omnirobot_simulator_utils/robot_margin3_pixel_only_tag.png",
            "robot_marker_margin":[3,3,3,3],
            "target_marker_path":"real_robots/omnirobot_simulator_utils/red_target_margin4_pixel_480x480.png", #for black target, use target_margin4_pixel.png", 
            "target_marker_margin":[4,4,4,4],
            "robot_marker_code":None, 
            "target_marker_code":None,
            "robot_marker_length":0.18, 
            "target_marker_length":0.18,
            "output_size" : [224,224],
            "init_x" : 0, 
            "init_y" : 0,
            "init_yaw" : 0, 
            "origin_size" : [640,480], 
            "cropped_size" : [480,480]
        }
        self.new_args = {**defalt_args, **args } #overwrite the args if it exists
        
        assert len(self.new_args['robot_marker_margin']) == 4
        assert len(self.new_args['target_marker_margin']) == 4
        assert len(self.new_args['output_size']) == 2

        self.render = OmniRobotEnvRender(**self.new_args)
        self.episode_idx = 0

    def send_json(self, msg):
        # env send msg to render
        command = msg.get('command', '')
        if command == 'reset':
            action = None
            self.episode_idx += 1

            if SECOND_CAM_TOPIC is not None:
                assert NotImplementedError

        elif command == 'action':
            if msg.get('is_discrete', False):
                action = Move(msg['action'])
            else:
                action = 'Continuous'

        elif command == "exit":
            return
        else:
            raise ValueError("Unknown command: {}".format(msg))

        has_bumped = False
        # We are always facing North
        if action == Move.FORWARD:
            if self.render.robot_pos[0] < MAX_X:
                self.render.forward()
            else:
                has_bumped = True
        elif action == Move.STOP:
            pass
        elif action == Move.RIGHT:
            if self.render.robot_pos[1] > MIN_Y:
                self.render.right()
            else:
                has_bumped = True
        elif action == Move.LEFT:
            if self.render.robot_pos[1] < MAX_Y:
                self.render.left()
            else:
                has_bumped = True
        elif action == Move.BACKWARD:
            if self.render.robot_pos[0] > MIN_X:
                self.render.backward()
            else:
                has_bumped = True

        elif action is None:
            # Env reset
            random_init_x = np.random.random_sample() * (INIT_MAX_X -INIT_MIN_X) + INIT_MIN_X
            random_init_y = np.random.random_sample() * (INIT_MAX_Y - INIT_MIN_Y) + INIT_MIN_Y
            
            self.render.setRobotCmd(random_init_x, random_init_y, 0)
            self.robot_marker_size_proprotion = np.random.randn() * NOISE_VAR_ROBOT_SIZE_PROPOTION + 1.0
            # target reset
            random_init_x = np.random.random_sample() * (TARGET_MAX_X -TARGET_MIN_X) + TARGET_MIN_X
            random_init_y = np.random.random_sample() * (TARGET_MAX_Y - TARGET_MIN_Y) + TARGET_MIN_Y
            self.render.setTargetCmd(random_init_x, random_init_y, 2 * np.pi * np.random.rand() - np.pi)

            # render the target
            self.render.renderTarget()

        elif action == 'Continuous':
            if  MIN_X < self.render.robot_pos[0] + msg['action'][0] < MAX_X and \
                    MIN_Y < self.render.robot_pos[1] + msg['action'][1] < MAX_Y:
                self.render.moveContinous(msg['action'])
            else:
                has_bumped = True
        else:
            print("Unsupported action")


        self.render.renderRobot()
        
        self.img = self.render.getCroppedImage()
        self.img = self.render.renderEnvLuminosityNoise(self.img, noise_var=NOISE_VAR_ENVIRONMENT, in_RGB=False, out_RGB=True)
        self.img = cv2.resize(self.img, tuple(self.render.output_size))
        reward = REWARD_NOTHING
        # Consider that we reached the target if we are close enough
        # we detect that computing the difference in area between TARGET_INITIAL_AREA
        # current detected area of the target
        if np.linalg.norm(np.array(self.render.robot_pos) - np.array(self.render.target_pos)) <  DIST_TO_TARGET_THRESHOLD:
            reward = REWARD_TARGET_REACH

        if has_bumped:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            reward = REWARD_BUMP_WALL

        self.reward = reward

    def recv_json(self):
        msg = {
            # XYZ position
            "position": self.render.robot_pos.tolist(),
            "reward": self.reward,
            "target_pos": self.render.target_pos.tolist()
        }
        return msg

    def recv_image(self):
        return self.img









