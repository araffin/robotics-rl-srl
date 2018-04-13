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


To test the environment with random actions:
```
python -m environments.test_env
```

## Reinforcement Learning

Note: All CNN policies normalize input, dividing it by 255.
By default, observations are not stacked.
For SRL, states are normalized using a running mean/std average.

About frame-stacking, action repeat (frameskipping) please read this blog post: [Frame Skipping and Pre-Processing for DQN on Atari](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)

### OpenAI Baselines

Several algorithms from [Open AI baselines](https://github.com/openai/baselines) have been integrated along with a random agent and random search:

- DQN and variants (Double, Dueling, prioritized experience replay)
- ACER: Sample Efficient Actor-Critic with Experience Replay
- A2C: A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which gives equal performance.
- PPO2: Proximal Policy Optimization (GPU Implementation)
- DDPG: Deep Deterministic Policy Gradients

To train an agent:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/
```

To load a trained agent and see the result:
```
python -m replay.enjoy_baselines --log-dir path/to/trained/agent/
```

Contiuous actions have been implemented for DDPG, PPO2 and random agent.
To use continuous actions in the position space:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ -c
```

To use continuous actions in the joint space:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ -c -joints
```

To run all the enviroments with all the SRL models for a given algorithm (you can use the same arguments as for training):
```
python  -m rl_baselines.pipeline --algo ppo2 --log-dir logs/
```

### Plot Learning Curve

To plot a learning curve from logs in visdom, you have to pass path to the experiment log folder:
```
python -m replay.plot --log-dir /logs/raw_pixels/ppo2/18-03-14_11h04_16/
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


## State Representation Learning Models

Please look the [SRL Repo](https://github.com/araffin/srl-robotic-priors-pytorch) to learn how to train a state representation model.
Then you must edit `config/srl_models.yaml` and set the right path to use the learned state representations.

To train the Reinforcement learning baselines on a specific SRL model:
```
python -m rl_baselines.train --algo ppo2 --log-dir logs/ --srl-model model_name
```

the available state representation model are:
- autoencoder: an autoencoder from the raw pixels
- ground_truth: the arm's x,y,z position
- srl_priors: SRL priors model
- supervised: a supervised model from the raw pixels to the arm's x,y,z position
- pca: pca applied to the raw pixels
- vae: a variational autoencoder from the raw pixels
- joints: the arm's joints angles
- joints_position: the arm's x,y,z position and joints angles

## Baxter Robot with Gazebo and ROS
Gym Wrapper for baxter environment, more details in the dedicated README (environments/gym_baxter/README.md).

1. Start ros nodes (Python 2):
```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

python -m gazebo.gazebo_server
```

Then, you can either try to teleoperate the robot (python 3):
```
python -m gazebo.teleop_client
```
or test the environment with random actions (using the gym wrapper):

```
python -m environments.gym_baxter.test_baxter_env
```

If the port is already used, you can see the program pid using the following command:
```
sudo netstat -lpn | grep :7777
```
and then kill it (with `kill -9 program_pid`)

or in one line:
```
kill -9 `sudo lsof -t -i:7777`
```

## Working With a Real Baxter Robot

WARNING: Please read COMPLETELY the following instructions before running and experiment on a real baxter.

### Recording Data With a Random Agent for SRL

1. Change you environment to match baxter ROS settings (usually using the `baxter.sh` script from RethinkRobotics)
or in your .bashrc:
```
# NB: This is only an example
export ROS_HOSTNAME=192.168.0.211  # Your IP
export ROS_MASTER_URI=http://baxter.local:11311 # Baxter IP
```

2. Calibrate the different values using `gazebo/real_baxter_debug.py`:
- Position of the target (BUTTON_POS) in `gazebo/real_baxter_server.py`
- Init position and orientation (LEFT_ARM_INIT_POS, LEFT_ARM_ORIENTATION) in `gazebo/real_baxter_server.py`
- Position of the table (minimum z): Z_TABLE in `gazebo/constants.py`
- Distance below which the target is considered to be reached: DIST_TO_TARGET_THRESHOLD in `gazebo/real_baxter_server.py`
- Distance above which the agent will get a negative reward in `environments/gym_baxter/test_baxter_env.py`
- Maximum number of steps per episode: MAX_STEPS in `environments/gym_baxter/baxter_env.py`

3. Configure images topics in `gazebo/constants.py`:
- IMAGE_TOPIC: main camera
- SECOND_CAM_TOPIC: second camera (set it to None if you don't want to record any data)
- DATA_FOLDER_SECOND_CAM: folder where the images of the second camera will be saved

4. Launch ROS bridge server (python 2):
```
python -m gazebo.real_baxter_server
```

5. Deactivate ROS from your environment and switch to python 3 environment (for using this repo)

6. Set the number of episodes you want to record, name of the experiment and random seed in `environments/gym_baxter/test_baxter_env.py`

7. Record data using a random agent:
```
python -m environments.gym_baxter.test_baxter_env
```
8. Wait until the end... Note: the real robot runs at approximately 0.6 FPS.

NB: If you want to save the image without resizing, you need to comment the line in the method `getObservation()` in `environments/gym_baxter/baxter_env.py`

### RL on a Real Robot

1. Update the settings in `rl_baselines/train.py`, so it saves and log the training more often (LOG_INTERVAL, SAVE_INTERVAL, ...)

2. Uncomment the line in `rl_baselines/ppo2.py` (or in the agent you want to use):
```python
# HACK: uncomment to use real baxter
import environments.gym_baxter.baxter_env as kuka_env
```

3. Launch ROS bridge server (python 2):
```
python -m gazebo.real_baxter_server
```

4. Start visdom for visualizing the training
```
python -m visdom.server
```

4. Train the agent (python 3)
```
python -m rl_baselines.train --srl-model ground_truth --log-dir logs_real/ --num-stack 1 --shape-reward --algo ppo2 --env Baxter-v0
```

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
