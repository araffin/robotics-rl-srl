import time
import numpy as np
import environments.omnirobot_gym.omnirobot_env as omnirobot_env


def actionPolicyTorwardTarget(robot_position, target_position):
    if abs(robot_position[0] - target_position[0]) > abs(robot_position[1] - target_position[1]):
        return 0 if robot_position[0] < target_position[0] else 1
               #forward                                        # backward
    else:
        # left                                          # right
        return 2 if robot_position[1] < target_position[1] else 3



timestr = time.strftime("%Y%m%d_%H%M%S")
log_folder = "omnirobot_" + timestr
env = omnirobot_env.OmniRobotEnv(renders=False, is_discrete=True, log_folder=log_folder, record_data=True)
timesteps = 500  # must be greater than MAX_STEPS
episodes = 500
env.seed(1)
i = 0

print('Starting episodes..., 20%_action use torward target policy, 80%_action use random policy ')
start_time = time.time()
try:
    for _ in range(episodes):
        observation = env.reset()
        for t in range(timesteps):
                if np.random.rand() < 0.8:
                    action = env.action_space.sample()
                else:
                    action = actionPolicyTorwardTarget(env.robot_pos, env.target_pos)            
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
