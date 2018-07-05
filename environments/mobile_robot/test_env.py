import time

import environments.mobile_robot.mobile_robot_env as mobile_env

env = mobile_env.MobileRobotGymEnv(renders=True, is_discrete=True, log_folder="mobile_robot", record_data=False, random_target=False)

timesteps = 1000  # must be greater than MAX_STEPS
episodes = 100
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
