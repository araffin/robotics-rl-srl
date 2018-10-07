.. _envs:

Environments
------------

All the environments we propose follow the OpenAI Gym interface. We also
extended this interface (adding extra methods) to work with SRL methods
(see `State Representation Learning
Models <#state-representation-learning-models>`__).

Available Environments
~~~~~~~~~~~~~~~~~~~~~~

You can find a recap table in the README.

If you want to add your own environment, please read
``enviroments/README.md``.

the available environments are:

-  Kuka arm: Here we have a Kuka arm which must reach a target, here a
   button.

   -  KukaButtonGymEnv-v0: Kuka arm with a single button in front.
   -  KukaRandButtonGymEnv-v0: Kuka arm with a single button in front,
      and some randomly positioned objects
   -  Kuka2ButtonGymEnv-v0: Kuka arm with 2 buttons next to each others,
      they must be pressed in the correct order (lighter button, then
      darker button).
   -  KukaMovingButtonGymEnv-v0: Kuka arm with a single button in front,
      slowly moving left to right.

-  Mobile robot: Here we have a mobile robot which reach a target
   position

   -  MobileRobotGymEnv-v0: A mobile robot on a 2d terrain where it
      needs to reach a target position.
   -  MobileRobot2TargetGymEnv-v0: A mobile robot on a 2d terrain where
      it needs to reach two target positions, in the correct order
      (lighter target, then darker target).
   -  MobileRobot1DGymEnv-v0: A mobile robot on a 1d slider where it can
      only go up and down, it must reach a target position.
   -  MobileRobotLineTargetGymEnv-v0: A mobile robot on a 2d terrain
      where it needs to reach a colored band going across the terrain.

-  Racing car: Here we have the interface for the Gym racing car
   environment. It must complete a racing course in the least time
   possible (only available in a terminal with X running)

   -  CarRacingGymEnv-v0: A racing car on a racing course, it must
      complete the racing course in the least time possible.

-  Baxter: A baxter robot that must reach a target, with its arms. (see
   `Working With Real Robots: Baxter and
   Robobo <#working-with-real-robots-baxter-and-robobo>`__)

   -  Baxter-v0: A bridge to use a baxter robot with ROS (in simulation,
      it uses Gazebo)

-  Robobo: A Robobo robot that must reach a target position.

   -  RoboboGymEnv-v0: A bridge to use a Robobo robot with ROS.

Generating Data
~~~~~~~~~~~~~~~

To test the environment with random actions:

::

   python -m environments.dataset_generator --no-record-data --display

Can be as well used to render views (or dataset) with two cameras if
``multi_view=True``.

To **record data** (i.e. generate a dataset) from the environment for
**training a SRL model**, using random actions:

.. code:: bash

   python -m environments.dataset_generator --num-cpu 4 --name folder_name
