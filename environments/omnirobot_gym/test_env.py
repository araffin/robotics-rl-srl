import time

import environments.omnirobot_gym.omnirobot_env as omnirobot_env
timestr = time.strftime("%Y%m%d_%H%M%S")
log_folder = "omnirobot_real_" + timestr
env = omnirobot_env.OmniRobotEnv(renders=False, is_discrete=True, log_folder=log_folder, record_data=True)
timesteps = 500  # must be greater than MAX_STEPS
episodes = 20
env.seed(1)
i = 0

print('Starting episodes...')
start_time = time.time()
try:
    for _ in range(episodes):
        observation = env.reset()
        for t in range(timesteps):
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                env.render()  # render() requires first the observation to be obtained
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
                i += 1
except KeyboardInterrupt:
    pass
env.closeServerConnection()
print("Avg. frame rate: {:.2f} FPS".format(i / (time.time() - start_time)))
