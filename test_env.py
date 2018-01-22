from __future__ import division, absolute_import, print_function

import time

import cv2
import pybullet as p
from envs.KukaGymEnv import KukaGymEnv
from envs.KukaCamGymEnv import KukaCamGymEnv

# import gym
# print(gym.envs.registry.all())
# exit()
# env = gym.make('CartPole-v0')
# env = KukaGymEnv(renders=False)
env = KukaCamGymEnv(renders=False)

i = 0
start_time = time.time()
for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        i += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("{:.2f} FPS".format(i / (time.time() - start_time)))
# cv2.imshow('test', observation)
# cv2.waitKey(0)
