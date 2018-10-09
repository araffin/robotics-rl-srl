.. S-RL Toolbox documentation master file, created by
   sphinx-quickstart on Sun Oct  7 21:18:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to S-RL Toolbox's documentation!
========================================

S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) Toolbox for Robotics

Github repository: https://github.com/araffin/robotics-rl-srl

This repository was made to evaluate State Representation Learning
methods using Reinforcement Learning. It integrates (automatic logging,
plotting, saving, loading of trained agent) various RL algorithms
(PPO, A2C, ARS, ACKTR, DDPG, DQN, ACER, CMA-ES, SAC, TRPO) along with different SRL methods
(see `SRL Repo <https://github.com/araffin/srl-zoo>`__) in an efficient
way (1 Million steps in 1 Hour with 8-core cpu and 1 Titan X GPU).

We also release customizable Gym environments for working with
simulation (Kuka arm, Mobile Robot in PyBullet, running at 250 FPS on a
8-core machine) and real robots (Baxter Robot, Robobo with ROS).

Related paper:

-  "S-RL Toolbox: Environments, Datasets and Evaluation Metrics for
   State Representation Learning" (Raffin et al., 2018)
   `https://arxiv.org/abs/1809.09369 <https://arxiv.org/abs/1809.09369>`__


.. toctree::
   :maxdepth: 2
   :caption: Guide

   guide/install
   guide/rl
   guide/hyperparams
   guide/envs
   guide/srl
   guide/plotting
   guide/real_robots
   guide/tests
   changelog




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
