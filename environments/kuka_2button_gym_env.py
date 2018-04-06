from . import kuka_button_gym_env as kuka_env

kuka_env.FORCE_RENDER = False
kuka_env.MAX_STEPS = 1500
kuka_env.MAX_DISTANCE = 2

from .kuka_button_gym_env import *

class Kuka2ButtonGymEnv(KukaButtonGymEnv):
    """
    Gym wrapper for Kuka environment with a push button
    :param urdf_root: (str) Path to pybullet urdf files
    :param renders: (bool) Whether to display the GUI or not
    :param is_discrete: (bool)
    :param name: (str) name of the folder where recorded data will be stored
    """

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 renders=False,
                 is_discrete=True,
                 name="kuka_2button_gym"):
        super(Kuka2ButtonGymEnv, self).__init__(urdf_root, renders, is_discrete, name)
        self.n_contacts2 = 0

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        self.terminated = False
        self.n_contacts = 0
        self.n_contacts2 = 0
        self.n_steps_outside = 0
        self.button_pressed = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])

        self.table_uid = p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                    0.000000, 0.000000, 0.0, 1.0)

        # Initialize button position
        x_pos = 0.5 + 0.0 * self.np_random.uniform(-1, 1)
        y_pos = 0.125 + 0.0 * self.np_random.uniform(-1, 1)
        self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos1 = np.array([x_pos, y_pos, Z_TABLE])

        x_pos = 0.5 + 0.0 * self.np_random.uniform(-1, 1)
        y_pos = -0.125 + 0.0 * self.np_random.uniform(-1, 1)
        self.button_uid2 = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos2 = np.array([x_pos, y_pos, Z_TABLE])

        self.button_pos = self.button_pos1
        self.button_pos[2] = 0.1

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

        # self.button_pos = np.array(p.getLinkState(self.button_uid, BUTTON_LINK_IDX)[0])
        # self.button_pos2 = np.array(p.getLinkState(self.button_uid2, BUTTON_LINK_IDX)[0])
        if self.saver is not None:
            self.saver.reset(self._observation, self.button_pos, self.getArmPos())

        if self.use_srl:
            # if len(self.saver.srl_model_path) > 0:
            # self.srl_model.load(self.saver.srl_model_path))
            return self.srl_model.getState(self._observation)

        return np.array(self._observation)

    def step(self, action):
        """
        :param action: (int)
        """
        # if you choose to do nothing
        if action is None:
            if self.action_joints:
                return self.step2(list(np.array(self._kuka.joint_positions)[:7]) + [0, 0])
            else:
                return self.step2([0, 0, 0, 0, 0])

        self.action = action  # For saver
        if self._is_discrete:
            dv = DELTA_V  # velocity per physics step.
            # Add noise to action
            dv += self.np_random.normal(0.0, scale=NOISE_STD)
            dx = [-dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, -dv, dv][action] 
            # da = [0, 0, 0, 0, 0, -0.1, 0.1][action]  # end effector angle
            finger_angle = 0.0  # Close the gripper
            # real_action = [dx, dy, -0.002, da, finger_angle]
            real_action = [dx, dy, dz, 0, finger_angle]
        else:
            if self.action_joints:
                arm_joints = np.array(self._kuka.joint_positions)[:7]
                d_theta = DELTA_THETA
                # Add noise to action
                d_theta += self.np_random.normal(0.0, scale=NOISE_STD_JOINTS)
                # append [0,0] for finger angles
                real_action = list(action * d_theta + arm_joints) + [0, 0]  # TODO remove up action
            else:
                dv = DELTA_V_CONTINUOUS
                # Add noise to action
                dv += self.np_random.normal(0.0, scale=NOISE_STD_CONTINUOUS)
                dx = action[0] * dv
                dy = action[1] * dv
                dz = action[2] * dv 
                finger_angle = 0.0  # Close the gripper
                real_action = [dx, dy, dz, 0, finger_angle]

        if VERBOSE:
            print(np.array2string(np.array(real_action), precision=2))
        print(action)
        print(real_action)
        print(DELTA_V_CONTINUOUS)

        return self.step2(real_action)

    def step2(self, action):
        """
        :param action:([float])
        """
        # Apply force to the button
        p.setJointMotorControl2(self.button_uid, BUTTON_GLIDER_IDX, controlMode=p.POSITION_CONTROL, targetPosition=0.1)
        p.setJointMotorControl2(self.button_uid2, BUTTON_GLIDER_IDX, controlMode=p.POSITION_CONTROL, targetPosition=0.1)

        for i in range(self._action_repeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._env_step_counter += 1

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timestep)

        done = self._termination()
        reward = self._reward()
        print(reward*1000)
        if self.saver is not None:
            self.saver.step(self._observation, self.action, reward, done, self.getArmPos())

        if self.use_srl:
            return self.srl_model.getState(self._observation), reward, done, {}

        return np.array(self._observation), reward, done, {}

    def _reward(self):
        gripper_pos = self.getArmPos()
        distance = np.linalg.norm(self.button_pos - gripper_pos, 2)
        # print(distance)
        reward = 0

        contact_points = p.getContactPoints(self.button_uid, self._kuka.kuka_uid)
        contact_points2 = p.getContactPoints(self.button_uid2, self._kuka.kuka_uid)
        self.n_contacts += int(len(contact_points) > 0)
        self.n_contacts2 += int(len(contact_points2) > 0)

        if self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION - 1:
            self.button_pos = self.button_pos2
            self.button_pos[2] = 0.1
            self.button_pressed = True
            reward = int(len(contact_points2) > 0)

        contact_with_table = len(p.getContactPoints(self.table_uid, self._kuka.kuka_uid)) > 0

        if distance > MAX_DISTANCE or contact_with_table:
            reward = -1
            self.n_steps_outside += 1
        else:
            self.n_steps_outside = 0

        if contact_with_table or ((self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION - 1) and (self.n_contacts2 >= N_CONTACTS_BEFORE_TERMINATION - 1)) \
                or self.n_steps_outside >= N_STEPS_OUTSIDE_SAFETY_SPHERE - 1:
            if contact_with_table:
                print("TABLE END")
            elif self.n_steps_outside >= N_STEPS_OUTSIDE_SAFETY_SPHERE - 1:
                print("OUT OF BOUND")
            else:
                print("GOOD END")
            self.terminated = True

        if SHAPE_REWARD:
            if IS_DISCRETE and False:
                return -distance
            else:
                # both Buttons pushed
                if self.terminated and reward > 0:
                    return 0.5
                # button 1 pushed
                elif (self.n_contacts < N_CONTACTS_BEFORE_TERMINATION - 1) and (len(contact_points) > 0):
                    print("BUTTON 1 PRESSED")
                    return 0.5
                # table
                elif contact_with_table:
                    return -(distance + 0.5)/1000
                # out of bounds
                elif distance > MAX_DISTANCE:
                    return -0.02
                # anything else
                else:
                    return -distance/1000

        return reward
