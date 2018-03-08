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
timesteps =  5 #200
episodes = 1 #30
env.seed(0)
i = 0
# start logging statistics and recording videos
# how to do in new release? env.monitor.start("/tmp/gym-results", algorithm_id="random")

print('starting episodes...')
start_time = time.time()
try:
    for _ in range(episodes):
        #observation = env.reset() # not needed, will hang the program, done in initialization of the env
        for t in range(timesteps):
            #env.render() should not be called before sampling an action, or will never return
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            i += 1
except KeyboardInterrupt as e:
    env.closeServerConnection()
    pass  #     sys.exit(e) ?
env.closeServerConnection()
print("{:.2f} FPS".format(i / (time.time() - start_time)))

# TODO: upload stats to the website
#gym.upload("/tmp/gym-results", api_key="JMXoHnlRtm86Fif6FUw4Qop1DwDkYHy0") # TODO how to obtain this key?


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
