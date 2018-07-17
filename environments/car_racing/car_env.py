import math

import numpy as np
import cv2
from gym.envs.box2d.car_racing import PLAYFIELD, FPS, STATE_H, STATE_W, VIDEO_H, VIDEO_W, WINDOW_H, WINDOW_W, SCALE, \
    ZOOM
from gym.envs.box2d.car_racing import CarRacing as GymCarRacing
from gym.envs.classic_control import rendering
from gym import spaces
import pyglet
from pyglet import gl

from environments.srl_env import SRLGymEnv
from state_representation.episode_saver import EpisodeSaver

MAX_STEPS = 10000
RENDER_HEIGHT = 224
RENDER_WIDTH = 224
N_DISCRETE_ACTIONS = 4

RELATIVE_POS = True  # Use relative position for ground truth


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class CarRacingEnv(GymCarRacing, SRLGymEnv):
    def __init__(self, name="car_racing", renders=False, record_data=False, is_discrete=True, state_dim=-1,
                 learn_states=False, save_path='srl_zoo/data/', srl_model="raw_pixels", env_rank=0, srl_pipe=None,
                 lookahead=5, **_):
        SRLGymEnv.__init__(self, srl_model=srl_model, relative_pos=False, env_rank=env_rank, srl_pipe=srl_pipe)
        GymCarRacing.__init__(self)
        self._renders = renders
        self._width = RENDER_WIDTH
        self._height = RENDER_HEIGHT
        self._is_discrete = is_discrete
        self.lookahead = lookahead
        self.relative_pos = RELATIVE_POS
        self._env_step_counter = 0
        self._observation = None
        self.saver = None

        if record_data:
            self.saver = EpisodeSaver(name, None, state_dim, globals_=getGlobals(), relative_pos=RELATIVE_POS,
                                      learn_states=learn_states, path=save_path)

        if self._is_discrete:
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        if self.srl_model == "ground_truth":
            self.state_dim = self.getGroundTruthDim()

        if self.srl_model == "raw_pixels":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def getTargetPos(self):
        nearest_idx = np.argmin(list(map(lambda a: np.sqrt(np.sum((a[2:4] - self.getGroundTruth()) ** 2)), self.track)))
        return np.array(self.track[(nearest_idx + self.lookahead) % len(self.track)][2:4])

    @staticmethod
    def getGroundTruthDim():
        return 2

    def getGroundTruth(self):
        return np.array(list(self.car.__dict__["hull"].position))

    def getObservation(self):
        """
        :return: (numpy array)
        """
        self._observation = self.render("rgb_array")
        self._observation = cv2.resize(self._observation, (self._width, self._height))
        return self._observation

    def reset(self):
        super().reset()
        self._env_step_counter = 0
        self.getObservation()

        if self.saver is not None:
            self.saver.reset(self._observation, self.getTargetPos(), self.getGroundTruth())

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation)

        return self._observation

    def step(self, action):
        if action is not None:
            if self._is_discrete:
                self.car.steer([-1, 1, 0, 0][action])
                self.car.gas([0, 0, 1, 0][action])
                self.car.brake([0, 0, 0, 1][action])
            else:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.getObservation()

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            self._env_step_counter += 1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self._env_step_counter >= MAX_STEPS:
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        if self.saver is not None:
            self.saver.step(self._observation, action, step_reward, done, self.getGroundTruth())

        if self.srl_model != "raw_pixels":
            return self.getSRLState(self._observation), step_reward, done, {}

        return np.array(self._observation), step_reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.viewer.window.set_visible(self._renders)
            self.score_label = pyglet.text.Label('0000', font_size=36, x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left',
                                                 anchor_y='center', color=(255, 255, 255, 255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)  # Animate zoom first second
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2 - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4 - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)))
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode == "rgb_array" or mode == "state_pixels":
            win.clear()
            t = self.transform
            if mode == 'rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        # agent can call or not call env.render() itself when recording video.
        if mode == "rgb_array" and not self.human_render:
            win.flip()

        if mode == 'human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []

        return arr
