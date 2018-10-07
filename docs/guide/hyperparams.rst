.. _hyperparams:

Hyperparameter Search
~~~~~~~~~~~~~~~~~~~~~

This repository also allows hyperparameter search, using
`hyperband <https://arxiv.org/abs/1603.06560>`__ or
`hyperopt <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
for the implemented RL algorithms

for example, here is the command for a hyperband search on PPO2, ground
truth on the mobile robot environment:

.. code:: bash

   python -m rl_baselines.hyperparam_search --optimizer hyperband --algo ppo2 --env MobileRobotGymEnv-v0 --srl-model ground_truth
