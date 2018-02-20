from __future__ import division, absolute_import, print_function

import time

from environments.baxter_env import BaxterEnv
from environments.kuka_button_gym_env import KukaButtonGymEnv
"""
This is a Python module and therefore, to run it as such:
python -m envirnoments.test_baxter_env

Notice that there are 2 equivalent ways to text Baxter environment below:
1) Instantiating the environment
2) Using gym.make()

"""


env = BaxterEnv(renders=True, is_discrete=True)
# env.num_envs = 1
env.seed(0)
i = 0
start_time = time.time()
for i_episode in range(20):
    observation = env.reset()
    for t in range(500):
        env.render()
        # print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        i += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("{:.2f} FPS".format(i / (time.time() - start_time)))




# Second test (will work only after env is registered in OpenAI Gym)
# The following way requires your env to be registered, so will happen when robust enough: gym.error.UnregisteredEnv: No registered env with id: BaxterButtonGymEnv-v0

#env = gym.make("BaxterButtonGymEnv-v0")
#env.reset()
#
# for i in range(100):
#     env.render("human")
#     action = env.action_space.sample()
#     obs, reward, done, misc = env.step(action)
#     print(action, obs.shape, reward, done)
#
#     time.sleep(.1)
