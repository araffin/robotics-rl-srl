.. _quickstart:

Getting Started
===============

Here is a quick example of how to train a PPO2 agent on Mobile Robot environment for 10 000 steps using 4 parallel processes:

::

  python -m rl_baselines.train --algo ppo2 --no-vis --num-cpu 4 --num-timesteps 10000 --env MobileRobotGymEnv-v0


The complete command (logs will be saved in `logs/` folder):

::

  python -m rl_baselines.train --algo rl_algo --env env1 --log-dir logs/ --srl-model raw_pixels --num-timesteps 10000 --no-vis


To use the robot's position as input instead of pixels, just pass `--srl-model ground_truth` instead of `--srl-model raw_pixels`
