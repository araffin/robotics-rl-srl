from .mobile_robot_env import *

REWARD_DIST_THRESHOLD = 0.1
ROBOT_OFFSET = 0.2  # Take into account the robot length for computing distance to target


class MobileRobotLineTargetGymEnv(MobileRobotGymEnv):
    """
    Gym wrapper for Mobile Robot with a line target environment
    WARNING: to be compatible with kuka scripts, additional keyword arguments are discarded
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) Whether to use discrete or continuous actions
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param use_srl: (bool) Set to true, use srl_models
    :param srl_model_path: (str) Path to the srl model
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param use_ground_truth: (bool) Set to true, the observation will be the ground truth (arm position)
    :param random_target: (bool) Set the target to a random position
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    :param env_rank: (int) the number ID of the environment
    :param pipe: (Queue, [Queue]) contains the input and output of the SRL model
    :param fpv: (bool) enable first person view camera
    :param srl_model: (str) The SRL_model used
    """
    def __init__(self, name="mobile_robot_line_target", **kwargs):
        super(MobileRobotLineTargetGymEnv, self).__init__(name=name, **kwargs)

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        # Return only the x-coordinate plus an offset to account for the robot length
        return self.target_pos[:1] - ROBOT_OFFSET

    def reset(self):
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, 0])
        p.setGravity(0, 0, -10)

        # Init the robot randomly
        x_start = self._max_x / 2 + self.np_random.uniform(- self._max_x / 3, self._max_x / 3)
        y_start = self._max_y / 2 + self.np_random.uniform(- self._max_y / 3, self._max_y / 3)
        self.robot_pos = np.array([x_start, y_start, 0])

        # Initialize target position
        x_pos = 0.9 * self._max_x
        y_pos = self._max_x
        if self._random_target:
            margin = 0.1 * self._max_x
            x_pos = self.np_random.uniform(self._min_x + margin, self._max_x - margin)

        self.target_uid = p.loadURDF("/urdf/wall_target.urdf", [x_pos, y_pos / 2, -0.045],
                              p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
        self.target_pos = np.array([x_pos, y_pos, 0])
        p.changeVisualShape(self.target_uid, -1, rgbaColor=[1, 1, 0, 1]) # yellow

        # Add walls
        # Path to the urdf file
        wall_urdf = "/urdf/wall.urdf"
        # Rgba colors
        red, green, blue = [0.8, 0, 0, 1], [0, 0.8, 0, 1], [0, 0, 0.8, 1]

        wall_left = p.loadURDF(wall_urdf, [self._max_x / 2, 0, 0], useFixedBase=True)
        # Change color
        p.changeVisualShape(wall_left, -1, rgbaColor=red)

        # getQuaternionFromEuler -> define orientation
        wall_bottom = p.loadURDF(wall_urdf, [self._max_x, self._max_y / 2, 0],
                                 p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)

        wall_right = p.loadURDF(wall_urdf, [self._max_x / 2, self._max_y, 0], useFixedBase=True)
        p.changeVisualShape(wall_right, -1, rgbaColor=green)

        wall_top = p.loadURDF(wall_urdf, [self._min_x, self._max_y / 2, 0],
                              p.getQuaternionFromEuler([0, 0, np.pi / 2]), useFixedBase=True)
        p.changeVisualShape(wall_top, -1, rgbaColor=blue)

        self.walls = [wall_left, wall_bottom, wall_right, wall_top]

        # Add mobile robot
        self.robot_uid = p.loadURDF(os.path.join(self._urdf_root, "racecar/racecar.urdf"), self.robot_pos,
                                    useFixedBase=True)

        self._env_step_counter = 0
        for _ in range(50):
            p.stepSimulation()

        self._observation = self.getObservation()

        if self.saver is not None:
            self.saver.reset(self._observation, self.getTargetPos(), self.getGroundTruth())

        if self.srl_model != 'raw_pixels':
            return self.getSRLState(self._observation)

        return np.array(self._observation)

    def _reward(self):
        """
        :return: (float)
        """
        # Distance to target
        distance = np.abs(self.getTargetPos()[0] - self.robot_pos[0])
        reward = 0

        if distance <= REWARD_DIST_THRESHOLD:
            reward = 1

        # Negative reward when it bumps into a wall
        if self.has_bumped:
            reward = -1

        if self._shape_reward:
            return -distance
        return reward
