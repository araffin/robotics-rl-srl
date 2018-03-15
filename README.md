# Reinforcement Learning (RL) and State Representation Learning (SRL) with robotic arms (Baxter and Kuka)

## Requirements:

- Python 3 (python 2 not supported because of OpenAI baselines)
- [OpenAI Baselines](https://github.com/openai/baselines) (latest version, install from source (at least commit 3cc7df0))
- [OpenAI Gym](https://github.com/openai/gym/) (version >= 0.10.3)
- Install the dependencies using `environment.yml` file (for conda users)

Note: The save method of ACER of baselines is currently buggy, you need to manually add an import (see [pull request #312](https://github.com/openai/baselines/pull/312))

[PyBullet Documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)

```
git clone git@github.com:araffin/robotics-rl-srl.git --recursive
```

## Kuka Arm \w PyBullet

Before you start a RL experiment, you have to make sure that a visdom server is running, unless you deactivate visualization.

Launch visdom server:
```
python -m visdom.server
```

## Reinforcement Learning

Note: All CNN policies normalize input, dividing it by 255.
By default, 4 observations are stacked.

About frame-stacking, action repeat (frameskipping) please read this blog post: [Frame Skipping and Pre-Processing for DQN on Atari](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)

### OpenAI Baselines

Several algorithms from [Open AI baselines](https://github.com/openai/baselines) have been integrated along with a random agent and random search:

- DQN and variants (Double, Dueling, prioritized experience replay)
- ACER: Sample Efficient Actor-Critic with Experience Replay
- A2C: A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which gives equal performance.
- PPO2: Proximal Policy Optimization (GPU Implementation)

To train an agent:
```
python -m rl_baselines.train --algo acer --log-dir logs/
```

To load a trained agent and see the result:
```
python -m replay.enjoy_baselines --log-dir path/to/trained/agent/
```

### Pytorch Agents

This concerns the `train_pytorch.py` script.

We are using Pytorch Implementation of A2C, PPO and [ACKTR](https://blog.openai.com/baselines-acktr-a2c/) from [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) (see `pytorch_agents` folder):

- A2C: See above
- ACKTR: (pronounced “actor”) Actor Critic using Kronecker-factored Trust Region
- PPO:Proximal Policy Optimization

To load a trained agent and see the result:
```
python -m replay.enjoy_pytorch --log-dir path/to/trained/agent/
```

## State Representation Learning Models

Please look the [SRL Repo](https://github.com/araffin/srl-robotic-priors-pytorch) to learn how to train a state representation model.
Then you must edit `config/srl_models.yaml` and set the right path to use the learned state representations.

## Baxter Robot \w Gazebo and ROS
Gym Wrapper for baxter environment + RL algorithms

```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

python -m gazebo.gazebo_server
python -m gazebo.teleop_client
```

```
sudo netstat -lpn | grep :7777
```

## Troubleshooting
If a submodule is not downloaded:
```
git submodule update --init
```
