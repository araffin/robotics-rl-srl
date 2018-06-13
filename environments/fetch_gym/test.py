import time

import gym
import gym.spaces
import cv2 as cv

env = gym.make('FetchReach-v1')
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
                img = env.render(mode='rgb_array')  # render() requires first the observation to be obtained 
                #cv.imshow('image'+str(t),img)
                cv.imwrite('/home/tianxiangdu/Desktop/img'+str(t)+'.jpg',img)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
                i += 1
except KeyboardInterrupt:
    pass
env.closeServerConnection()
print("Avg. frame rate: {:.2f} FPS".format(i / (time.time() - start_time)))
