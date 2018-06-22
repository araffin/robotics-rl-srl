from .mobile_robot_env import *

MAX_STEPS = 1500

class MobileRobot2TargetGymEnv(MobileRobotGymEnv):
    """
    Gym wrapper for Mobile Robot environment with 2 targets
    WARNING: to be compatible with kuka scripts, additional keyword arguments are discarded
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) Whether to use discrete or continuous actions
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param srl_model: (bool) Set to true, use srl_models
    :param srl_model_path: (str) Path to the srl model
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param use_ground_truth: (bool) Set to true, the observation will be the ground truth (arm position)
    :param random_target: (bool) Set the target to a random position
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    :param env_rank: (int) the number ID of the environment
    :param pipe: (tuple) contains the input and output of the SRL model
    """
    def __init__(self, name="mobile_robot_2target", **kwargs):
        super(MobileRobot2TargetGymEnv, self).__init__(name=name, **kwargs)

        self.current_target = 0

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.current_target = 0
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
        self.button_uid = []
        self.button_pos = []

        x_pos = 0.9 * self._max_x
        y_pos = self._max_y * 3 / 4
        if self._random_target:
            margin = 0.1 * self._max_x
            x_pos = self.np_random.uniform(self._min_x + margin, self._max_x - margin)
            y_pos = self.np_random.uniform(self._min_y + margin, self._max_y - margin)
        self.button_uid.append(p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, 0], useFixedBase=True))
        self.button_pos.append(np.array([x_pos, y_pos, 0]))

        x_pos = 0.1 * self._max_x
        y_pos = self._max_y * 3 / 4
        if self._random_target:
            margin = 0.1 * self._max_x
            x_pos = self.np_random.uniform(self._min_x + margin, self._max_x - margin)
            y_pos = self.np_random.uniform(self._min_y + margin, self._max_y - margin)
        self.button_uid.append(p.loadURDF("/urdf/simple_button_2.urdf", [x_pos, y_pos, 0], useFixedBase=True))
        self.button_pos.append(np.array([x_pos, y_pos, 0]))

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

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation)

        return np.array(self._observation)

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        # Return only the [x, y] coordinates
        return self.button_pos[self.current_target][:2]

    def step(self, action):
        """
        :param action: (int)
        """
        # True if it has bumped against a wall
        self.has_bumped = False
        if self._is_discrete:
            dv = DELTA_POS
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            dx = [-dv, dv, 0, 0][action]
            dy = [0, 0, -dv, dv][action]
            real_action = np.array([dx, dy])
        else:
            raise ValueError("Only discrete actions is supported")

        if self.verbose:
            print(np.array2string(np.array(real_action), precision=2))

        previous_pos = self.robot_pos.copy()
        self.robot_pos[:2] += real_action
        # Handle collisions
        for i, (limit, robot_dim) in enumerate(zip([self._max_x, self._max_y], [ROBOT_LENGTH, ROBOT_WIDTH])):
            margin = self.collision_margin + robot_dim / 2
            # If it has bumped against a wall, stay at the previous position
            if self.robot_pos[i] < margin or self.robot_pos[i] > limit - margin:
                self.has_bumped = True
                self.robot_pos = previous_pos
                break
        # Update mobile robot position
        p.resetBasePositionAndOrientation(self.robot_uid, self.robot_pos, [0, 0, 0, 1])

        p.stepSimulation()
        self._env_step_counter += 1

        self._observation = self.getObservation()

        reward = self._reward()
        done = self._termination()
        if self.saver is not None:
            self.saver.step(self._observation, action, reward, done, self.getGroundTruth())

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation), reward, done, {}

        return np.array(self._observation), reward, done, {}

    def _reward(self):
        """
        :return: (float)
        """
        # Distance to target
        distance = np.linalg.norm(self.getTargetPos() - self.robot_pos[:2], 2)
        reward = 0

        if distance <= REWARD_DIST_THRESHOLD:
            reward = 1
            if self.current_target < len(self.button_pos) - 1:
                self.current_target += 1
            # self.terminated = True

        # Negative reward when it bumps into a wall
        if self.has_bumped:
            reward = -1

        if self._shape_reward:
            return -distance
        return reward
