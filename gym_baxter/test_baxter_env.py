from __future__ import division, absolute_import, print_function
import time
from gym_baxter.envs.baxter_env import BaxterEnv

env = BaxterEnv(renders=True, is_discrete=True)
timesteps = 2# 200
episodes = 1# 30
env.seed(0)
i = 0

print('Starting episodes...')
start_time = time.time()
for _ in range(episodes):
    for t in range(timesteps):
        try:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            i += 1
        except KeyboardInterrupt:
            pass
env.closeServerConnection()
print("Avg. frame rate: {:.2f} FPS".format(i / (time.time() - start_time)))
