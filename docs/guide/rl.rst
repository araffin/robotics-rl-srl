.. _rl:

Reinforcement Learning
----------------------

.. note::

  All CNN policies normalize input, dividing it by 255. By default,
  observations are not stacked. For SRL, states are normalized using a
  running mean/std average.

About frame-stacking, action repeat (frameskipping) please read this
blog post: `Frame Skipping and Pre-Processing for DQN on
Atari <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>`__

Before you start a RL experiment, you have to make sure that a visdom
server is running, unless you deactivate visualization.

Launch visdom server:

::

   python -m visdom.server

.. _rl-algorithms:-openai-baselines-and-more:

RL Algorithms: OpenAI Baselines and More
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several algorithms from `Stable
Baselines <https://github.com/hill-a/stable-baselines>`__ have been
integrated along with some evolution strategies and SAC:

-  A2C: A synchronous, deterministic variant of Asynchronous Advantage
   Actor Critic (A3C).
-  ACER: Sample Efficient Actor-Critic with Experience Replay
-  ACKTR: Actor Critic using Kronecker-Factored Trust Region
-  ARS: Augmented Random Search
   (`https://arxiv.org/abs/1803.07055 <https://arxiv.org/abs/1803.07055>`__)
-  CMA-ES: Covariance Matrix Adaptation Evolution Strategy
-  DDPG: Deep Deterministic Policy Gradients
-  DeepQ: DQN and variants (Double, Dueling, prioritized experience replay)
-  PPO1: Proximal Policy Optimization (MPI Implementation)
-  PPO2: Proximal Policy Optimization (GPU Implementation)
-  SAC: Soft Actor Critic
-  TRPO: Trust Region Policy Optimization (MPI Implementation)

Train an Agent with Discrete Actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train an agent (without visualization with visdom):

::

   python -m rl_baselines.train --algo ppo2 --log-dir logs/ --no-vis

You can train an agent on the latest learned model (knowing it's type)
located at ``log_folder: srl_zoo/logs/DatasetName/`` (defined for each
environment in ``config/srl_models.yaml``) :

::

   python -m rl_baselines.train --algo ppo2 --log-dir logs/ --latest --srl-model srl_combination --env MobileRobotGymEnv-v0

Train an Agent with Continuous Actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Continuous actions have been implemented for DDPG, PPO2, ARS, CMA-ES,
SAC and random agent. To use continuous actions in the position space:

::

   python -m rl_baselines.train --algo ppo2 --log-dir logs/ -c

To use continuous actions in the joint space:

::

   python -m rl_baselines.train --algo ppo2 --log-dir logs/ -c -joints

.. _train-an-agent-multiple-times-on-multiple-environments,-using-different-methods:

Train an agent multiple times on multiple environments, using different methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run multiple enviroments with multiple SRL models for a given
algorithm (you can use the same arguments as for training should you
need to specify anything to the training script):

::

   python  -m rl_baselines.pipeline --algo ppo2 --log-dir logs/ --env env1 env2 [...] --srl-model model1 model2 [...]

For example, run a total of 30 experiments of ppo2 with 4 cpus and
randomly initialized target position, in the default environment using
VAE, and using ground truth (15 experiments each):

::

   python  -m rl_baselines.pipeline --algo ppo2 --log-dir logs/ --srl-model vae ground_truth --random-target --num-cpu 4 --num-iteration 15

Load a Trained Agent
^^^^^^^^^^^^^^^^^^^^

To load a trained agent and see the result:

::

   python -m replay.enjoy_baselines --log-dir path/to/trained/agent/ --render


Add your own RL algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a class that inherits
   ``rl_baselines.base_classes.BaseRLObject`` which implements your
   algorithm. You will need to define specifically:

   -  ``save(save_path, _locals=None)``: to save your model during or
      after training.
   -  ``load(load_path, args=None)``: to load and return a saved
      instance of your class (static function).
   -  ``customArguments(parser)``: ``@classmethod`` to define specifics
      command line arguments from ``train.py`` or ``pipeline.py`` calls,
      then returns the parser object.
   -  ``getAction(observation, dones=None)``: to get the action from a
      given observation.
   -  ``makeEnv(self, args, env_kwargs=None, load_path_normalise=None)``:
      override if you need to change the environment wrappers (static
      function).
   -  ``train(args, callback, env_kwargs=None, hyperparam=None)``: to
      create the environment, and train your algorithm on said
      environment.
   -  (OPTIONAL) ``getActionProba(observation, dones=None)``: to get the
      action probabilities from a given observation. This is used for
      the action probability plotting in ``replay.enjoy_baselines``.
   -  (OPTIONAL) ``getOptParam()``: ``@classmethod`` to return the
      hyperparameters that can be optimised through the callable
      argument. Along with the type and range of said parameters.

2. Add your class to the ``registered_rl`` dictionary in
   ``rl_baselines/registry.py``, using this format
   ``NAME: (CLASS, ALGO_TYPE, [ACTION_TYPE])``, where:

   -  ``NAME``: is your algorithm's name.
   -  ``CLASS``: is your class that inherits ``BaseRLObject``.
   -  ``ALGO_TYPE``: is the type of algorithm, defined by the enumerator
      ``AlgoType`` in ``rl_baselines/__init__.py``, can be
      ``REINFORCEMENT_LEARNING``, ``EVOLUTION_STRATEGIES`` or ``OTHER``
      (``OTHER`` is used to define algorithms that can't be run in
      ``enjoy_baselines.py`` (ex: Random_agent)).
   -  ``[ACTION_TYPE]``: is the list of compatible type of actions,
      defined by the enumerator ``ActionType`` in
      ``rl_baselines/__init__.py``, can be ``CONTINUOUS`` and/or
      ``DISCRETE``.

3. Now you can call your algorithm using ``--algo NAME`` with
   ``train.py`` or ``pipeline.py``.
