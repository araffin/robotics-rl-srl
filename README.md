# Gym wrappers for state representation learning with robotic arms (Baxter + Kuka)


To create the new environment for Python 3.6, in your Terminal run:
```
conda create -n py36 python=3.6 anaconda
```

In the conda virtual environment (`source activate env_name`), install pybullet:
```
pip/pip3 install pybullet
```

Clone the repository, which contains subrepositories. First time:
```
git clone git@github.com:araffin/robotics-srl-rl.git --recursive
```

or afterwards, instead of git pull, so that subrepositories update, do:
```
git submodule update --init
```

## Kuka Arm \w PyBullet

```
python main.py --num-processes 4 --num-stack 1 --env-name KukaButtonGymEnv-v0 --algo a2c
```

## Reinforcement Learning

We are using Pytorch Implementation of A2C, PPO and [ACKTR](https://blog.openai.com/baselines-acktr-a2c/) from [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) (see `pytorch_agents` folder):

- A2C - A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which gives equal performance.
- ACKTR (pronounced “actor”) Actor Critic using Kronecker-factored Trust Region ("Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation") is a more sample-efficient reinforcement learning algorithm than TRPO and A2C,
- PPO - Proximal Policy Optimization


## Baxter Robot \w Gazebo and ROS
Gym Wrapper for baxter environment + RL algorithms

```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

python -m gazebo.gazebo_server
python -m gazebo.teleop_client
```

When your server dies or the program does not end properly, you may need to run:
```
sudo netstat -lpn | grep :7777
```
before
```
kill -9 processNr
```
