# Gym wrappers for state representation learning with robotic arms (Baxter + Kuka)


Clone the repository, which contains subrepositories. First time:
```
git clone git@github.com:araffin/robotics-srl-rl.git --recursive
```

or afterwards, instead of git pull, so that subrepositories update, do:
```
git submodule update --init
```

Create your conda environment using provided file environment.yml


## Kuka Arm \w PyBullet

```
python main.py --num-processes 4 --num-stack 1 --env-name KukaButtonGymEnv-v0 --algo a2c
```

## Reinforcement Learning

We are using PyTorch Implementation of A2C, PPO and [ACKTR](https://blog.openai.com/baselines-acktr-a2c/) from [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) (see `pytorch_agents` folder):

- A2C - A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which gives equal performance.
- ACKTR (pronounced “actor”) Actor Critic using Kronecker-factored Trust Region ("Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation") is a more sample-efficient reinforcement learning algorithm than TRPO and A2C,
- PPO - Proximal Policy Optimization


## Baxter Robot \w Gazebo and ROS
Gym Wrapper for baxter environment + RL algorithms

```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

python -m gazebo.gazebo_server
```
and run Option 1) Baxter teleoperated by user:
```
python -m gazebo.teleop_client
```
or run Option 2) Baxter gym environment module with random/other agents
```
python -m gym_baxter.test_baxter_env
```
Note that there are 2 equivalent ways to run Baxter environment:
1) Instantiating BaxterEnv environment
2) Using gym.make()


When your server dies or the program does not end properly, to free the port used by the server script,
you may need to run:
```
sudo netstat -lpn | grep :7777
```
and then use the process nr to do:
```
kill -9 processNr
```
