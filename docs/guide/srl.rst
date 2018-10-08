.. _srl:

State Representation Learning Models
====================================

A State Representation Learning (SRL) model aims to compress from a high
dimensional observation, a compact representation. This learned
representation can be used instead of learning a policy directly from
pixels, in a deep reinforcement learning algorithm.

A more detailed overview:
`https://arxiv.org/pdf/1802.04181.pdf <https://arxiv.org/pdf/1802.04181.pdf>`__

Please look the `SRL Repo <https://github.com/araffin/srl-zoo>`__ to
learn how to train a state representation model. Then you must edit
``config/srl_models.yaml`` and set the right path to use the learned
state representations.

To train a Reinforcement learning agent on a specific SRL model:

::

   python -m rl_baselines.train --algo ppo2 --log-dir logs/ --srl-model model_name

Available SRL models
--------------------

The available state representation models are:

-  ground_truth: the arm's x,y,z position
-  robotic_priors: robotic priors model (`Learning State Representations
   with Robotic
   Priors <http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Jonschkowski-15-AURO.pdf>`__)
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
-  joints: the arm's joints angles (kuka environments only)
-  joints_position: the arm's x,y,z position and joints angles (kuka environments only)

Note: for debugging, we integrated logging of states (we save the states
that the RL agent encountered during training) with SAC algorithm. To
log the states during RL training you have to pass the ``--log-states``
argument:

::

   python -m rl_baselines.train --srl-model ground_truth --env MobileRobotLineTargetGymEnv-v0 --log-dir logs/ --algo sac --reward-scale 10 --log-states

The states will be saved in a ``log_srl/`` folder as numpy archives,
inside the log folder of the rl experiment.


Add a custom SRL model
----------------------


If your SRL model is a charateristic of the environment (position,
angles, ...):

1. Add the name of the model to the ``registered_srl`` dictionary in
   ``state_representation/registry.py``, using this format
   ``NAME: (SRLType.ENVIRONMENT, [LIMITED_TO_ENV])``, where:

   -  ``NAME``: is your model's name.
   -  ``[LIMITED_TO_ENV]``: is the list of environments where this model
      works (will check for subclass), set to ``None`` if this model
      applies to every environment.

2. Modifiy the ``def getSRLState(self, observation)`` in the
   environments to return the data you want for this model.
3. Now you can call your SRL model using ``--srl-model NAME`` with
   ``train.py`` or ``pipeline.py``.

Otherwise, for the SRL model that are external to the environment
(Supervised, autoencoder, ...):

1. Add your SRL model that inherits ``SRLBaseClass``, to the function
   ``state_representation.models.loadSRLModel``.
2. Add the name of the model to the ``registered_srl`` dictionary in
   ``state_representation/registry.py``, using this format
   ``NAME: (SRLType.SRL, [LIMITED_TO_ENV])``, where:

   -  ``NAME``: is your model's name.
   -  ``[LIMITED_TO_ENV]``: is the list of environments where this model
      works (will check for subclass), set to ``None`` if this model
      applies to every environment.

3. Add the name of the model to ``config/srl_models.yaml``, with the
   location of the saved model for each environment (can point to a
   dummy location, but must be defined).
4. Now you can call your SRL model using ``--srl-model NAME`` with
   ``train.py`` or ``pipeline.py``.
