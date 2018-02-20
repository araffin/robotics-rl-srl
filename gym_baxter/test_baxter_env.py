from __future__ import division, absolute_import, print_function
import time
from gym_baxter.envs.baxter_env import BaxterEnv
#from environments.kuka_button_gym_env import KukaButtonGymEnv

"""
This is a Python module and therefore, to run it as such:
python -m gym_baxter.test_baxter_env

Note that there are 2 equivalent ways to text Baxter environment below:
1) Instantiating the environment
2) Using gym.make()

"""

env = BaxterEnv(renders=True, is_discrete=True)
timesteps =  500
episodes = 20
env.seed(0)
i = 0
start_time = time.time()
try:
    for i_episode in range(episodes):
        observation = env.reset()
        for t in range(timesteps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            i += 1
except KeyboardInterrupt as e:
    pass  #     sys.exit(e) ?

print("{:.2f} FPS".format(i / (time.time() - start_time)))
env.closeServerConnection()


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
