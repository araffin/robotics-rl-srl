from __future__ import division, absolute_import, print_function

import time

import environments.kuka_button_gym_env as kuka_env


kuka_env.MAX_DISTANCE = 0.65
kuka_env.RECORD_DATA = True

env = kuka_env.KukaButtonGymEnv(renders=False, is_discrete=True, multi_view=False, name="kuka_test_dual_cam")
# env.num_envs = 1

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
