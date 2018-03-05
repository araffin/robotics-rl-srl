# Gym wrappers for state representation learning with robotic arms (Baxter + Kuka)


```
git clone git@github.com:araffin/robotics-rl-srl.git --recursive
```

or
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
- PPO- Proximal Policy Optimization


## Baxter Robot \w Gazebo and ROS
Gym Wrapper for baxter environment + RL algorithms

```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

python -m gazebo.gazebo_server
python -m gazebo.teleop_client
```
Note, the first 3 commands need to be run in Python 2, while the teleop_client runs on
Anaconda py35 env.


```
sudo netstat -lpn | grep :7777
```

## Troubleshooting
If a submodule is not downloaded:
```
git submodule update --init
```
