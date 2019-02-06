import time
import numpy as np
import environments.omnirobot_gym.omnirobot_env as omnirobot_env





timestr = time.strftime("%Y%m%d_%H%M%S")
save_path = "srl_zoo/data/"
name =  "omnirobot_"+ timestr
env = omnirobot_env.OmniRobotEnv(name=name, renders=False, is_discrete=True, save_path=save_path, record_data=True)
timesteps = 500  # must be greater than MAX_STEPS
episodes = 50
env.seed(1)
i = 0

print('Starting episodes..., "30%_action use torward target policy, 70%_action use random policy ')
start_time = time.time()
try:
    for _ in range(episodes):
        observation = env.reset()
        for t in range(timesteps):
                if np.random.rand() < 0.7:
                    action = env.action_space.sample()
                else:
                    action = omnirobot_env.actionPolicyTorwardTarget(env.robot_pos, env.target_pos)     
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
