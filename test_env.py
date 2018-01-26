from __future__ import division, absolute_import, print_function

import time

import pybullet as p

from envs.KukaGymEnv import KukaGymEnv
from envs.KukaCamGymEnv import KukaCamGymEnv


env = KukaCamGymEnv(renders=True, isDiscrete=True)
env.seed(0)
i = 0
start_time = time.time()
for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        i += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("{:.2f} FPS".format(i / (time.time() - start_time)))
