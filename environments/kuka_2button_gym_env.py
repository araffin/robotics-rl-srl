from .kuka_button_gym_env import *

MAX_STEPS = 1500

class Kuka2ButtonGymEnv(KukaButtonGymEnv):
    """
    Gym wrapper for Kuka environment with 2 push buttons
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool) Whether to use discrete or continuous actions
    :param multi_view :(bool) if TRUE -> returns stacked images of the scene on 6 channels (two cameras)
    :param name: (str) name of the folder where recorded data will be stored
    :param max_distance: (float) Max distance between end effector and the button (for negative reward)
    :param action_repeat: (int) Number of timesteps an action is repeated (here it is equivalent to frameskip)
    :param shape_reward: (bool) Set to true, reward = -distance_to_goal
    :param action_joints: (bool) Set actions to apply to the joint space
    :param use_srl: (bool) Set to true, use srl_models
    :param srl_model_path: (str) Path to the srl model
    :param record_data: (bool) Set to true, record frames with the rewards.
    :param use_ground_truth: (bool) Set to true, the observation will be the ground truth (arm position)
    :param use_joints: (bool) Set input to include the joint angles (only if not using SRL model)
    :param button_random: (bool) Set the button position to a random position on the table
    :param force_down: (bool) Set Down as the only vertical action allowed
    :param state_dim: (int) When learning states
    :param learn_states: (bool)
    :param verbose: (bool) Whether to print some debug info
    :param save_path: (str) location where the saved data should go
    """

    def __init__(self, name="kuka_2button_gym", max_distance=2, force_down=False, **kwargs):
        super(Kuka2ButtonGymEnv, self).__init__(name=name, max_distance=max_distance, force_down=force_down, **kwargs)

        self.max_steps = MAX_STEPS
        self.n_contacts = [0,0]


    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.terminated = False
        self.n_contacts = [0,0]
        self.button_all_pos = []
        self.button_uid = []
        self.goal_id = 0 # here, goal_id is used to know which button is the next one to press
        self.n_steps_outside = 0
        self.button_pressed = [False]
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])

        self.table_uid = p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                    0.000000, 0.000000, 0.0, 1.0)

        # Initialize button position
        x_pos = 0.5
        y_pos = 0.125
        if self._button_random:
            x_pos += 0.15 * self.np_random.uniform(-1, 1)
            y_pos += 0.175 * self.np_random.uniform(0, 1)

        x_pos = 0.5 + 0.0 * self.np_random.uniform(-1, 1)
        y_pos = 0.125 + 0.0 * self.np_random.uniform(-1, 1)
        self.button_uid.append(p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE]))
        self.button_all_pos.append(np.array([x_pos, y_pos, Z_TABLE + BUTTON_DISTANCE_HEIGHT]))

        x_pos = 0.5
        y_pos = -0.125
        if self._button_random:
            x_pos += 0.15 * self.np_random.uniform(-1, 1)
            y_pos += 0.175 * self.np_random.uniform(-1, 0)

        self.button_uid.append(p.loadURDF("/urdf/simple_button_2.urdf", [x_pos, y_pos, Z_TABLE]))
        self.button_all_pos.append(np.array([x_pos, y_pos, Z_TABLE + BUTTON_DISTANCE_HEIGHT]))

        # need to define this for the ground_truth model
        self.button_pos = self.button_all_pos[0]

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdf_root_path=self._urdf_root, timestep=self._timestep,
                               use_inverse_kinematics=(not self.action_joints), small_constraints=False)
        self._kuka.use_null_space = True

        self._env_step_counter = 0
        # Close the gripper and wait for the arm to be in rest position
        for _ in range(500):
            if self.action_joints:
                self._kuka.applyAction(list(np.array(self._kuka.joint_positions)[:7]) + [0, 0])
            else:
                self._kuka.applyAction([0, 0, 0, 0, 0])
            p.stepSimulation()

        # Randomize init arm pos: take 5 random actions
        for _ in range(N_RANDOM_ACTIONS_AT_INIT):
            if self._is_discrete:
                action = [0, 0, 0, 0, 0]
                sign = 1 if self.np_random.rand() > 0.5 else -1
                action_idx = self.np_random.randint(3)  # dx, dy or dz
                action[action_idx] += sign * DELTA_V
            else:
                if self.action_joints:
                    joints = np.array(self._kuka.joint_positions)[:7]
                    joints += DELTA_THETA * self.np_random.normal(joints.shape)
                    action = list(joints) + [0, 0]
                else:
                    action = np.zeros(5)
                    rand_direction = self.np_random.normal((3,))
                    # L2 normalize, so that the random direction is not too high or too low
                    rand_direction /= np.linalg.norm(rand_direction, 2)
                    action[:3] += DELTA_V_CONTINUOUS * rand_direction

            self._kuka.applyAction(list(action))
            p.stepSimulation()

        self._observation = self.getExtendedObservation()

        if self.saver is not None:
            self.saver.reset(self._observation, self.button_all_pos[self.goal_id], self.getArmPos())

        if self.use_srl:
            return self.srl_model.getState(self._observation)

        return np.array(self._observation)


    def step2(self, action):
        """
        :param action:([float])
        """
        # Apply force to the button
        for uid in self.button_uid:
            p.setJointMotorControl2(uid, BUTTON_GLIDER_IDX, controlMode=p.POSITION_CONTROL, targetPosition=0.1)

        for i in range(self._action_repeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._env_step_counter += 1

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timestep)

        reward = self._reward()
        done = self._termination()
        if self.saver is not None:
            self.saver.step(self._observation, self.action, reward, done, self.getArmPos())

        if self.use_srl:
            return self.srl_model.getState(self._observation), reward, done, {}

        return np.array(self._observation), reward, done, {}

    def _reward(self):
        gripper_pos = self.getArmPos()
        distance = np.linalg.norm(self.button_all_pos[self.goal_id] - gripper_pos, 2)
        reward = 0

        contact_points = p.getContactPoints(self.button_uid[self.goal_id], self._kuka.kuka_uid)
        self.n_contacts[self.goal_id] += int(len(contact_points) > 0)

        # for the sparse reward
        if self.goal_id == len(self.button_uid)-1:
            reward = int(len(contact_points) > 0)

        # next button
        if self.n_contacts[self.goal_id] >= N_CONTACTS_BEFORE_TERMINATION and not self.button_pressed[self.goal_id]:
            self.button_pressed[self.goal_id] = True
            self.button_pressed.append(False)
            if len(self.button_all_pos) > self.goal_id + 1:
                self.button_pos = self.button_all_pos[self.goal_id + 1]
                self.goal_id += 1

        contact_with_table = len(p.getContactPoints(self.table_uid, self._kuka.kuka_uid)) > 0

        if distance > self._max_distance or contact_with_table:
            reward = -1
            self.n_steps_outside += 1
        else:
            self.n_steps_outside = 0

        if contact_with_table or (self.n_contacts[-1] >= N_CONTACTS_BEFORE_TERMINATION) \
                or self.n_steps_outside >= N_STEPS_OUTSIDE_SAFETY_SPHERE - 1:
            self.terminated = True

        if self._shape_reward:
            # both Buttons pushed
            if self.terminated and reward > 0:
                return 50
            # button pushed
            elif (self.n_contacts[self.goal_id] < N_CONTACTS_BEFORE_TERMINATION) and (len(contact_points) > 0):
                return 25
            # table
            elif contact_with_table:
                return -250
            # out of bounds
            elif distance > self._max_distance:
                return -20
            # anything else
            else:
                return -distance

        return reward
