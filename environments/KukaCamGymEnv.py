from __future__ import division, absolute_import, print_function

import os
import math
import time
import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data

from . import kuka

MAX_STEPS = 500
N_CONTACTS_TO_SOLVE = 5
RENDER_HEIGHT = 84 # 720 // 5
RENDER_WIDTH = 84 # 960 // 5
Z_TABLE = -0.2
MAX_DISTANCE = 0.6

# TODO: improve the physics of the button

class KukaCamGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=True):
        self._timestep = 1. / 240.
        self._urdf_root = urdfRoot
        self._action_repeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self._cam_dist = 1.1
        self._cam_yaw = 145
        self._cam_pitch = -36
        self._cam_roll = 0
        self.base_pos = (0.316, -0.2, -0.1)
        self._isDiscrete = isDiscrete
        self.terminated = 0
        self.renderer = p.ER_TINY_RENDERER
        self.debug = False
        self.n_contacts = 0

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

            self.renderer = p.ER_BULLET_HARDWARE_OPENGL
            self.debug = True
            self.x_slider = p.addUserDebugParameter("x_slider", -10, 10, self.base_pos[0])
            self.y_slider = p.addUserDebugParameter("y_slider", -10, 10, self.base_pos[1])
            self.z_slider = p.addUserDebugParameter("z_slider", -10, 10, self.base_pos[2])
            self.dist_slider = p.addUserDebugParameter("cam_dist", 0, 10, self._cam_dist)
            self.yaw_slider = p.addUserDebugParameter("cam_yaw", -180, 180, self._cam_yaw)
            self.pitch_slider = p.addUserDebugParameter("cam_pitch", -180, 180, self._cam_pitch)

        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self._seed()
        self.reset()
        observation_dim = len(self.getExtendedObservation())

        observation_high = np.array([np.finfo(np.float32).max] * observation_dim)
        if self._isDiscrete:
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 3
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3))
        self.viewer = None

    def reset(self):
        return self._reset()

    def _reset(self):
        self.terminated = 0
        self.n_contacts = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.loadURDF(os.path.join(self._urdf_root, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"), 0.5000000, 0.00000, -.820000, 0.000000, 0.000000,
                   0.0, 1.0)

        x_pos = 0.5 + 0.0 * np.random.uniform(-1, 1)
        y_pos = 0 + 0.0 * np.random.uniform(-1, 1)
        self.button_uid = p.loadURDF("/urdf/simple_button.urdf", [x_pos, y_pos, Z_TABLE])
        self.button_pos = np.array([x_pos, y_pos, Z_TABLE])
        self.glider_idx = 1

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdf_root, timeStep=self._timestep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = self._render("rgb_array")
        return self._observation

    def _step(self, action):
        if self._isDiscrete:
            # WARNING: dv not the same for the z axis
            dv = 0.01  # velocity per physics step.
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            dz = [0, 0, 0, 0, 0, -dv, dv][action]
            da = [0, 0, 0, 0, 0, -0.1, 0.1][action] # end effector angle
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
            # realAction = [dx, dy, dz, 0, f]
        else:
            dv = 0.01
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.1
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]

        return self.step2(realAction)

    def step2(self, action):

        p.setJointMotorControl2(self.button_uid, self.glider_idx, controlMode=p.POSITION_CONTROL, targetPosition=0.1)

        for i in range(self._action_repeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1

        self._observation = self.getExtendedObservation()
        if self._renders:
            time.sleep(self._timestep)

        done = self._termination()
        reward = self._reward()
        # print("len=%r" % len(self._observation))

        return np.array(self._observation), reward, done, {}

    def _render(self, mode='human', close=False):
        # Apparently this code is not used
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.base_pos

        if self.debug:
            self._cam_dist = p.readUserDebugParameter(self.dist_slider)
            self._cam_yaw = p.readUserDebugParameter(self.yaw_slider)
            self._cam_pitch = p.readUserDebugParameter(self.pitch_slider)
            x = p.readUserDebugParameter(self.x_slider)
            y = p.readUserDebugParameter(self.y_slider)
            z = p.readUserDebugParameter(self.z_slider)
            base_pos = (x, y, z)
            # self._cam_roll = p.readUserDebugParameter(self.roll_slider)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=self._cam_roll,
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self.renderer)
        # return px
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        if self.terminated or self._envStepCounter > MAX_STEPS:
            self._observation = self.getExtendedObservation()
            return True
        return False

    def _reward(self):
        button_link_idx = 1
        gripper_pos = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)[0]
        distance = np.linalg.norm(self.button_pos - gripper_pos, 2)
        # print(distance)

        contact_points = p.getContactPoints(self.button_uid, self._kuka.kukaUid, button_link_idx)
        reward = int(len(contact_points) > 0)
        self.n_contacts += reward

        if distance > MAX_DISTANCE:
            reward = -1

        if self.n_contacts > N_CONTACTS_TO_SOLVE:
            self.terminated = True

        return reward
