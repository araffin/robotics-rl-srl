from __future__ import division, absolute_import, print_function

import time

from .kuka_button_gym_env import KukaButtonGymEnv


env = KukaButtonGymEnv(renders=True, is_discrete=True, name="kuka_test")
# env.num_envs = 1
env.seed(0)
i = 0
start_time = time.time()
for i_episode in range(50):
    observation = env.reset()
    for t in range(501):
        env.render()
        # print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        i += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("{:.2f} FPS".format(i / (time.time() - start_time)))
