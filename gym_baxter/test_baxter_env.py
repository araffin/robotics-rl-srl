from __future__ import division, absolute_import, print_function
import time
from gym_baxter.envs.baxter_env import BaxterEnv

"""
This is a Python module and therefore, to run it as such:
python -m gym_baxter.test_baxter_env

Note that there are 2 equivalent ways to text Baxter environment below:
1) Instantiating the environment
2) Using gym.make()

"""

env = BaxterEnv(renders=True, is_discrete=True)
timesteps =  200
episodes = 30
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
                print("Episode finished after {} timesteps".format(t+1))
                break
            i += 1
        except KeyboardInterrupt:
            env.closeServerConnection() # TODO Solve: when client fails, and therefore, for it to run again,
env.closeServerConnection()
print("Avg. frame rate: {:.2f} FPS".format(i / (time.time() - start_time)))
