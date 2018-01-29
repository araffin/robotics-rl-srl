from __future__ import division, absolute_import, print_function

import time

import pybullet as p
from environments.KukaCamGymEnv import KukaCamGymEnv


env = KukaCamGymEnv(renders=True, isDiscrete=True)
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

#
# from baselines import deepq
#
# def callback(lcl, glb):
#     return False
#     # stop training if reward exceeds 199
#     total = sum(lcl['episode_rewards'][-101:-1]) / 100
#     totalt = lcl['t']
#     is_solved = totalt > 2000 and total >= -50
#     return is_solved
#
#
# def main():
#
#     env = KukaCamGymEnv(renders=True, isDiscrete=True)
#     # env.num_envs = 1
#     env.seed(0)
#
#     model = deepq.models.mlp([64])
#     act = deepq.learn(
#         env,
#         q_func=model,
#         lr=1e-3,
#         max_timesteps=10000,
#         buffer_size=50000,
#         exploration_fraction=0.1,
#         exploration_final_eps=0.02,
#         print_freq=10,
#         callback=callback
#     )
#     print("Saving model to kuka_model.pkl")
#     act.save("kuka_model.pkl")
#
#
# if __name__ == '__main__':
#     main()
