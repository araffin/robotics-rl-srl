# Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics

This repository was made to evaluate State Representation Learning methods using Reinforcement Learning. It integrates (automatic logging, plotting, saving, loading of trained agent) various RL algorithms (PPO, A2C, ARS, DDPG, DQN, ACER, CMA-ES, SAC) along with different SRL methods (see [SRL Repo](https://github.com/araffin/srl-zoo)) in an efficient way (1 Million steps in 1 Hour with 8-core cpu and 1 Titan X GPU).

We also release customizable Gym environments for working with simulation (Kuka arm, Mobile Robot in PyBullet, running at 250 FPS on a 8-core machine) and real robots (Baxter Robot, Robobo with ROS).

Table of Contents
=================
  * [Installation](#installation)
  * [Reinforcement Learning](#reinforcement-learning)
    * [RL Algorithms: OpenAI Baselines and More](#rl-algorithms-openai-baselines-and-more)
  * [Environments](#environments)
    * [Available Environments](#available-environments)
    * [Generating Data](#generating-data)
  * [State Representation Learning Models](#state-representation-learning-models)
    * [Plot Learning Curve](#plot-learning-curve)
  * [Working With Real Robots: Baxter and Robobo](#working-with-real-robots-baxter-and-robobo)
  * [Troubleshooting](#troubleshooting)
  * [Known issues](#known-issues)


## Installation

- Python 3 is required (python 2 is not supported because of OpenAI baselines)
- [OpenAI Baselines](https://github.com/openai/baselines) (latest version, install from source (at least commit 3cc7df0))
- Install the dependencies using `environment.yml` file (for conda users)

Note: The save method of ACER of baselines is currently buggy, you need to manually add an import (see [pull request #312](https://github.com/openai/baselines/pull/312))

[PyBullet Documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)

How to download the project (note the `--recursive` argument because we are using git submodules):
```
git clone git@github.com:araffin/robotics-rl-srl.git --recursive
```

## Reinforcement Learning

Note: All CNN policies normalize input, dividing it by 255.
By default, observations are not stacked.
For SRL, states are normalized using a running mean/std average.

About frame-stacking, action repeat (frameskipping) please read this blog post: [Frame Skipping and Pre-Processing for DQN on Atari](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)

Before you start a RL experiment, you have to make sure that a visdom server is running, unless you deactivate visualization.

Launch visdom server:
```
python -m visdom.server
```

### RL Algorithms: OpenAI Baselines and More

Several algorithms from [Open AI baselines](https://github.com/openai/baselines) have been integrated along with some evolution strategies and SAC:

- DQN and variants (Double, Dueling, prioritized experience replay)
- ACER: Sample Efficient Actor-Critic with Experience Replay
- A2C: A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which gives equal performance.
- PPO2: Proximal Policy Optimization (GPU Implementation)
- DDPG: Deep Deterministic Policy Gradients
- ARS: Augmented Random Search
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- SAC: Soft Actor Critic

To train an agent (without visualization with visdom):
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ --no-vis
```

To load a trained agent and see the result:
```
python -m replay.enjoy_baselines --log-dir path/to/trained/agent/ --render
```

Continuous actions have been implemented for DDPG, PPO2, ARS, CMA-ES, SAC and random agent.
To use continuous actions in the position space:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ -c
```

To use continuous actions in the joint space:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ -c -joints
```

To run multiple enviroments with multiple SRL models for a given algorithm (you can use the same arguments as for training should you need to specify anything to the training script):
```
python  -m rl_baselines.pipeline --algo ppo2 --log-dir logs/ --env env1 env2 [...] --srl-model model1 model2 [...]
```

For example, ppo2 with 4 cpus and randomly initialized target position, in the default environment using VAE and ground_truth:
```
python  -m rl_baselines.pipeline --algo ppo2 --log-dir logs/ --srl-model vae ground_truth --random-target --num-cpu 4
```

If you want to integrate your own RL algorithm, please read `rl_baselines/README.md`.

## Environments

All the environments we propose follow the OpenAI Gym interface. We also extended this interface (adding extra methods) to work with SRL methods (see [State Representation Learning Models](#state-representation-learning-models)).

### Available Environments

If you want to add your own environment, please read `enviroments/README.md`.

the available environments are:
- Kuka arm: Here we have a Kuka arm which must reach a target, here a button.
    - KukaButtonGymEnv-v0: Kuka arm with a single button in front.
    - KukaRandButtonGymEnv-v0: Kuka arm with a single button in front, and some randomly positioned objects
    - Kuka2ButtonGymEnv-v0: Kuka arm with 2 buttons next to each others, they must be pressed in the correct order (lighter button, then darker button).
    - KukaMovingButtonGymEnv-v0: Kuka arm with a single button in front, slowly moving left to right.
- Mobile robot: Here we have a mobile robot which reach a target position
    - MobileRobotGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach a target position.
    - MobileRobot2TargetGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach two target positions, in the correct order (lighter target, then darker target).
    - MobileRobot1DGymEnv-v0: A mobile robot on a 1d slider where it can only go up and down, it must reach a target position.
    - MobileRobotLineTargetGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach a colored band going across the terrain.
- Baxter: A baxter robot that must reach a target, with its arms. (see [Working With Real Robots: Baxter and Robobo](#working-with-real-robots-baxter-and-robobo))
    - Baxter-v0: A bridge to use a baxter robot with ROS (in simulation, it uses Gazebo)
- Robobo: A Robobo robot that must reach a target position.
    - RoboboGymEnv-v0: A bridge to use a Robobo robot with ROS.


### Generating Data

To test the environment with random actions:
```
python -m environments.dataset_generator --no-record-data --display
```
Can be as well used to render views (or dataset) with two cameras if `multi_view=True`.

To **record data** (i.e. generate a dataset) from the environment for **training a SRL model**, using random actions:
```bash
python -m environments.dataset_generator --num-cpu 4 --name folder_name
```


## State Representation Learning Models

Please look the [SRL Repo](https://github.com/araffin/srl-zoo) to learn how to train a state representation model.
Then you must edit `config/srl_models.yaml` and set the right path to use the learned state representations.

To train the Reinforcement learning baselines on a specific SRL model:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ --srl-model model_name
```

The available state representation models are:
- autoencoder: an autoencoder from the raw pixels
- ground_truth: the arm's x,y,z position
- robotic_priors: Robotic Priors model
- supervised: a supervised model from the raw pixels to the arm's x,y,z position
- pca: pca applied to the raw pixels
- vae: a variational autoencoder from the raw pixels
- joints: the arm's joints angles
- joints_position: the arm's x,y,z position and joints angles

If you have trained a SRL model with views of two cameras, you need to set the global variable N_CHANNELS to 6 in `srl_zoo/preprocessing/preprocess.py` to perform the inference.

### Plot Learning Curve

To plot a learning curve from logs in visdom, you have to pass path to the experiment log folder:
```
python -m replay.plots --log-dir /logs/raw_pixels/ppo2/18-03-14_11h04_16/
```

To aggregate data from different experiments (different seeds) and plot them (mean + standard error).
You have to pass path to rl algorithm log folder (parent of the experiments log folders):
```
python -m replay.aggregate_plots --log-dir /logs/raw_pixels/ppo2/ --shape-reward --timesteps --min-x 1000 -o logs/path/to/output_file
```
Here it plots experiments with reward shaping and that have a minimum of 1000 data points (using timesteps on the x-axis), the plot data will be saved in the file `output_file.npz`.

To create a comparison plots from saved plots (.npz files), you need to pass a path to folder containing .npz files:
```
python -m replay.compare_plots -i logs/path/to/folder/ --shape-reward --timesteps
```

## Working With Real Robots: Baxter and Robobo

The instructions for working with a real robot are availables here : `real_robots/README.md`.

## Troubleshooting
If a submodule is not downloaded:
```
git submodule update --init
```
If you have troubles installing mpi4py, make sure you the following installed:
```
sudo apt-get install libopenmpi-dev openmpi-bin openmpi-doc
```

## Known issues

The inverse kinematics function has trouble finding a solution when the arm is fully straight and the arm must bend to reach the requested point.
