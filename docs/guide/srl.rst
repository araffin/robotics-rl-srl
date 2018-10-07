.. _srl:

State Representation Learning Models
------------------------------------

Please look the `SRL Repo <https://github.com/araffin/srl-zoo>`__ to
learn how to train a state representation model. Then you must edit
``config/srl_models.yaml`` and set the right path to use the learned
state representations.

To train the Reinforcement learning baselines on a specific SRL model:

::

   python -m rl_baselines.train --algo ppo2 --log-dir logs/ --srl-model model_name

The available state representation models are:

-  ground_truth: the arm's x,y,z position
-  robotic_priors: Robotic Priors model
-  supervised: a supervised model from the raw pixels to the arm's x,y,z
   position
-  pca: pca applied to the raw pixels
-  autoencoder: an autoencoder from the raw pixels
-  vae: a variational autoencoder from the raw pixels
-  inverse: an inverse dynamics model
-  forward: a forward dynamics model
-  srl_combination: a model combining several losses (e.g. vae + forward
   + inverse...) for SRL
-  multi_view_srl: a SRL model using views from multiple cameras as
   input, with any of the above losses (e.g triplet and others)
-  joints: the arm's joints angles
-  joints_position: the arm's x,y,z position and joints angles

Note: for debugging, we integrated logging of states (we save the states
that the RL agent encountered during training) with SAC algorithm. To
log the states during RL training you have to pass the ``--log-states``
argument:

::

   python -m rl_baselines.train --srl-model ground_truth --env MobileRobotLineTargetGymEnv-v0 --log-dir logs/ --algo sac --reward-scale 10 --log-states

The states will be saved in a ``log_srl/`` folder as numpy archives,
inside the log folder of the rl experiment.
