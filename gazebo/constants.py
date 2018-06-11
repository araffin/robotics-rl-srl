from __future__ import print_function, absolute_import, division

from enum import Enum

# ==== CONSTANTS FOR BAXTER ROBOT ====
# Socket port
SERVER_PORT = 7777
HOSTNAME = 'localhost'
USING_REAL_BAXTER = False
USING_ROBOBO = True
# For compatibility with teleop_client
Move = None
DELTA_POS = 0

# Calibrated values for Real Baxter
if USING_REAL_BAXTER:
    # Initial position of the arm
    LEFT_ARM_INIT_POS = [0.69850099, 0.14505832, 0.08032852]
    # Initial orientation
    LEFT_ARM_ORIENTATION = [0.99893116, -0.04207143, -0.00574656, -0.01826233]
    # Button position (target)
    BUTTON_POS = [0.7090276, 0.13833109, -0.11170768]
    # Distance below which the target is considered to be reached
    DIST_TO_TARGET_THRESHOLD = 0.035
    # Max distance between end effector and the button (for negative reward)
    MAX_DISTANCE = 0.18
    # Used by the inverse kinematics
    IK_SEED_POSITIONS = None
    # Constant distance delta for actions
    DELTA_POS = 0.02
    Z_TABLE = - 0.10
    # Max number of steps per episode
    MAX_STEPS = 100
    # ROS Topics
    IMAGE_TOPIC = "/kinect2/qhd/image_color"
    # Set the second cam topic to None if there is only one camera
    SECOND_CAM_TOPIC = "/camera/rgb/image_raw"
    DATA_FOLDER_SECOND_CAM = "real_baxter_second_cam"
elif USING_ROBOBO:
    # ROS Topics
    IMAGE_TOPIC = "/camera/rgb/image_raw"
    # SECOND_CAM_TOPIC = "/camera/image_repub"
    SECOND_CAM_TOPIC = None
    # Max number of steps per episode
    MAX_STEPS = 20
    # Initial area in the image of the target
    # It must be calibrated after changing the target position
    TARGET_INITIAL_AREA = 3700
    # Boundaries
    MIN_X, MAX_X = -3, 3
    MIN_Y, MAX_Y = -4, 3


    # Define the possible Moves
    class Move(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4

# Gazebo
else:
    LEFT_ARM_INIT_POS = [0.6, 0.30, 0.20]
    # ['left_e0', 'left_e1', 'left_s0', 'left_s1', 'left_w0', 'left_w1', 'left_w2']
    IK_SEED_POSITIONS = [-1.535, 1.491, -0.038, 0.194, 1.546, 1.497, -0.520]
    DELTA_POS = 0.05
    Z_TABLE = -0.14
    IMAGE_TOPIC = "/cameras/head_camera_2/image"
    MAX_STEPS = 100

# Arrow keys for teleoperation
UP_KEY = 82  # the arrow key "up"
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
D_KEY = 100  # the letter "d"
U_KEY = 117  # The letter "u"
R_KEY = 114  # the letter "r"
