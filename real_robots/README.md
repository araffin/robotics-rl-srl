Table of Contents
=================
- [Table of Contents](#table-of-contents)
  - [Baxter Robot with Gazebo and ROS](#baxter-robot-with-gazebo-and-ros)
  - [Working With a Real Baxter Robot](#working-with-a-real-baxter-robot)
    - [Recording Data With a Random Agent for SRL](#recording-data-with-a-random-agent-for-srl)
    - [RL on a Real Baxter Robot](#rl-on-a-real-baxter-robot)
  - [Working With a Real Robobo](#working-with-a-real-robobo)
    - [Recording Data With a Random Agent for SRL](#recording-data-with-a-random-agent-for-srl-1)
    - [RL on a Real Robobo](#rl-on-a-real-robobo)
  - [Working with a Omnirobot](#working-with-a-omnirobot)
    - [Architecture of Omnirobot](#architecture-of-omnirobot)
      - [Architecture of Real Omnirobot](#architecture-of-real-omnirobot)
      - [Architecture of Omnirobot Simulator](#architecture-of-omnirobot-simulator)
    - [Switch between real robot and simulator](#switch-between-real-robot-and-simulator)
    - [Real Omnirobot](#real-omnirobot)
      - [Launch RL on real omnirobot](#launch-rl-on-real-omnirobot)
      - [Recording Data of real omnirobot](#recording-data-of-real-omnirobot)
    - [Omnirobot Simulator](#omnirobot-simulator)
      - [Noise of Omnirobot Simulator](#noise-of-omnirobot-simulator)
      - [Train RL on Omnirobot Simulator](#train-rl-on-omnirobot-simulator)
      - [Generate Data from Omnirobot Simulator](#generate-data-from-omnirobot-simulator)
    - [Known issues of Omnirobot](#known-issues-of-omnirobot)
## Baxter Robot with Gazebo and ROS
Gym Wrapper for baxter environment, more details in the dedicated README (environments/gym_baxter/README.md).

**Important Note**: ROS (and Gazebo + Baxter) only works with python2, whereas this repo (except the ROS scripts) works with python3.
For Ros/Baxter installation, please look at the [Official Tutorial](http://sdk.rethinkrobotics.com/wiki/Workstation_Setup).
Also, ROS comes with its own version of OpenCV, so when running the python3 scripts, you need to deactivate ROS. In the same vein, if you use Anaconda, you need to disable it when you want to run ROS scripts (denoted as python 2 in the following instructions).

1. Start ros nodes (Python 2):
```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

python -m real_robots.gazebo_server
```

Then, you can either try to teleoperate the robot (python 3):
```
python -m real_robots.teleop_client
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

2. Calibrate the different values in `real_robots/constants.py` using `real_robots/real_baxter_debug.py`:
- Set USING_REAL_BAXTER to True
- Position of the target: BUTTON_POS
- Init position and orientation: LEFT_ARM_INIT_POS, LEFT_ARM_ORIENTATION
- Position of the table (minimum z): Z_TABLE
- Distance below which the target is considered to be reached: DIST_TO_TARGET_THRESHOLD
- Distance above which the agent will get a negative reward: MAX_DISTANCE
- Maximum number of steps per episode: MAX_STEPS

3. Configure images topics in `real_robots/constants.py`:
- IMAGE_TOPIC: main camera
- SECOND_CAM_TOPIC: second camera (set it to None if you don't want to use a second camera)
- DATA_FOLDER_SECOND_CAM: folder where the images of the second camera will be saved

4. Launch ROS bridge server (python 2):
```
python -m real_robots.real_baxter_server
```

5. Deactivate ROS from your environment and switch to python 3 environment (for using this repo)

6. Set the number of episodes you want to record, name of the experiment and random seed in `environments/gym_baxter/test_baxter_env.py`

7. Record data using a random agent:
```
python -m environments.gym_baxter.test_baxter_env
```
8. Wait until the end... Note: the real robot runs at approximately 0.6 FPS.

NB: If you want to save the image without resizing, you need to comment the line in the method `getObservation()` in `environments/gym_baxter/baxter_env.py`

### RL on a Real Baxter Robot

1. Update the settings in `rl_baselines/train.py`, so it saves and log the training more often (LOG_INTERVAL, SAVE_INTERVAL, ...)

2. Make sure that USING_REAL_BAXTER is set to True in `real_robots/constants.py`.

3. Launch ROS bridge server (python 2):
```
python -m real_robots.real_baxter_server
```

4. Start visdom for visualizing the training
```
python -m visdom.server
```

4. Train the agent (python 3)
```
python -m rl_baselines.train --srl-model ground_truth --log-dir logs_real/ --num-stack 1 --shape-reward --algo ppo2 --env Baxter-v0
```

## Working With a Real Robobo

[Robobo Documentation](https://bitbucket.org/mytechia/robobo-programming/wiki/Home)

Note: the Robobo is controlled using time (the feedback frequency is too low to do closed-loop control)
The robot was calibrated for a constant speed of 10.

### Recording Data With a Random Agent for SRL

1. Change you environment to match Robobo ROS settings or in your .bashrc:
NOTE: Robobo is using ROS Java, if you encounter any problem with the cameras (e.g. with a xtion), you should create the master node on your computer and change the settings in the robobo dev app.
```
# NB: This is only an example
export ROS_HOSTNAME=192.168.0.211  # Your IP
export ROS_MASTER_URI=http://robobo.local:11311 # Robobo IP
```

2. Calibrate the different values in `real_robots/constants.py` using `real_robots/real_robobo_server.py` and `real_robots/teleop_client.py` (Client for teleoperation):
- Set USING_ROBOBO to True
- Area of the target: TARGET_INITIAL_AREA
- Boundaries of the enviroment: (MIN_X, MAX_X, MIN_Y, MAX_Y)
- Maximum number of steps per episode: MAX_STEPS
IMPORTANT NOTE: if you use color detection to detect the target, you need to calibrate the HSV thresholds `LOWER_RED` and `UPPER_RED` in `real_robots/constants.py` (for instance, using [this script](https://github.com/sergionr2/RacingRobot/blob/v0.3/opencv/dev/threshold.py)). Be careful, you may have to change the color conversion (`cv2.COLOR_BGR2HSV` instead of `cv2.COLOR_RGB2HSV`)

3. Configure images topics in `real_robots/constants.py`:
- IMAGE_TOPIC: main camera
- SECOND_CAM_TOPIC: second camera (set it to None if you don't want to use a second camera)
- DATA_FOLDER_SECOND_CAM: folder where the images of the second camera will be saved

NOTE: If you want to use robobo's camera (phone camera), you need to republish the image to the raw format:
```
rosrun image_transport republish compressed in:=/camera/image raw out:=/camera/image_repub
```

4. Launch ROS bridge server (python 2):
```
python -m real_robots.real_robobo_server
```

5. Deactivate ROS from your environment and switch to python 3 environment (for using this repo)

6. Set the number of episodes you want to record, name of the experiment and random seed in `environments/robobo_gym/test_robobo_env.py`

7. Record data using a random agent:
```
python -m environments.robobo_gym.test_robobo_env
```

8. Wait until the end... Note: the real robobo runs at approximately 0.1 FPS.

NB: If you want to save the image without resizing, you need to comment the line in the method `getObservation()` in `environments/robobo_gym/robobo_env.py`

### RL on a Real Robobo

1. Update the settings in `rl_baselines/train.py`, so it saves and logs the training more often (LOG_INTERVAL, SAVE_INTERVAL, ...)

2. Make sure that USING_ROBOBO is set to True in `real_robots/constants.py`.

3. Launch ROS bridge server (python 2):
```
python -m real_robots.real_robobo_server
```

4. Start visdom for visualizing the training
```
python -m visdom.server
```

4. Train the agent (python 3)
```
python -m rl_baselines.train --srl-model ground_truth --log-dir logs_real/ --num-stack 1 --algo ppo2 --env RoboboGymEnv-v0
```

## Working with a Omnirobot
By default, Omnirobot uses the same reward and terminal policy with the MobileRobot environment. Thus each episodes will have exactly 251 steps, and when the robot touches the target, it will get `reward=1`, when it touches the border, it will get `reward=-1`, otherwise, `reward=0`.

All the important parameters are writed in constants.py, thus you can simply modified the reward or terminal policy of this environment.

### Architecture of Omnirobot
#### Architecture of Real Omnirobot
The omnirobot's environment contains two principle components (two threads). 
- `real_robots/omnirobot_server.py` (python2, using ROS to communicate with robot) 
- `environments/omnirobot_gym/omnirobot_env.py` (python3, wrapped baseline environment)
These two components uses zmq socket to communicate. The socket port can be changed, and by defualt it's 7777. 
These two components should be launched manually, because they use different environment (ROS and anaconda).

#### Architecture of Omnirobot Simulator
The simulator has only one thread, omnirobot_env. The simulator is a object of this running thread, it uses exactly the same api as `zmq`, thus `omnirobot_server` can be easily switched to `omnirobot_simulator_server` without changing code of `omnirobot_env`.

### Switch between real robot and simulator
- Switch from real robot to simulator
  modify `real_robots/constants.py`, set `USING_OMNIROBOT = False` and `USING_OMNIROBOT_SIMULATOR = True`
- Switch from simulator to real robot:
  modify `real_robots/constants.py`, set `USING_OMNIROBOT = True` and `USING_OMNIROBOT_SIMULATOR = False`

### Real Omnirobot
Omnirobot offers the clean environment for RL, for each step of RL, the real robot does a close-loop positional control to reach the supposed position.

when the robot is moving, `omnirobot_server` will be blocked until it receives a msg from the topic `finished`, which is sent by the robot. This blocking has a time out (by default 30s), thus if anything unexpected happens, the `omnirobot_server` will fail and close.

#### Launch RL on real omnirobot
To launch the rl  experience of omnirobot, do these step-by-step:
- switch to real robot (modify constans.py, ensure  `USING_OMNIROBOT = True`)
- setup ROS environment and comment `anaconda` in `~/.bashrc`, launch a new terminal, run `python -m real_robots.omnirobot_server`
- comment ROS environment and uncomment `anaconda` in `~/.bashrc`, launch a new terminal.
  - If you want to train RL on real robot, run `python -m rl_baselines.train --env OmnirobotEnv-v0` with other options customizable. 
  - If you want to replay the RL policy on real robot, which can be trained on the simulator, run `python -m replay.enjoy_baselines --log-dir path/to/RL/logs -render`

#### Recording Data of real omnirobot
To launch a acquisition of real robot's dataset, do these step-by-step:
- switch to real robot (modify constans.py, ensure  `USING_OMNIROBOT = True`)
- setup ROS environment and comment `anaconda` in `~/.bashrc`, launch a new terminal, run `python -m real_robots.omnirobot_server`
- Change `episodes` to the number of you want in `environments/omnirobot_gym/test_env.py`
- comment ROS environment and uncomment `anaconda` in `~/.bashrc`, launch a new terminal, run `python -m environments.omnirobot_gym.test_env`. Note that you should move the target manually between the different episodes. **Attention**, you can try to use Random Agent or a agent always do the toward target policy (this can increase the positive reward proportion in the dataset), or combine them by setting a proportion (`TORWARD_TARGET_PROPORTION`). 


### Omnirobot Simulator
This simulator uses photoshop tricks to make realistic image of environment. It need several image as input.
- back ground image (480x480, undistorted)
- robot's tag/code, cropped from a real environment image(480x480, undistorted), with a margin 3 or 4 pixels.
- target's tag/code, cropped from a real environment image (480x480, undistorted), with a margin 3 or 4 pixels.

It also needs some important information:
- margin of markerts
- camera info file's path, which generated by ROS' camera_calibration package. The camera matrix should be corresponding with original image size (eg. 640x480 for our case) 

The detail of the inputs above can be find from OmniRobotEnvRender's comments.
#### Noise of Omnirobot Simulator
To make the simulator more general, and make RL/SRL more stable, several types of noise are added to it. The parameters of these noises can be modified from the top of omnirobot_simulator_server.py

- noise of robot position, yaw.
  Gaussian noise, controlled by `NOISE_VAR_ROBOT_POS` and `NOISE_VAR_ROBOT_YAW`.
- noise of markers in pixel-wise.
  Gaussian noise to simulate camera's noise, apply pixel-wise noise on the markers' images, controlled by `NOISE_VAR_TARGET_PIXEL` and `NOISE_VAR_ROBOT_PIXEL`.
- noise of environment's luminosity.
  Apply Gaussian noise on LAB space of output image, to simulate the environment's luminosity change, controlled by `NOISE_VAR_ENVIRONMENT`.
- noise of marker's size.
  Change size of robot's and target's marker proportionally, to simulate the position variance on the vertical axis. This kind of  noise is controlled by `NOISE_VAR_ROBOT_SIZE_PROPOTION` and `NOISE_VAR_TARGET_SIZE_PROPOTION`.
#### Train RL on Omnirobot Simulator
- switch to simulator (modify constans.py, ensure  `USING_OMNIROBOT_SIMULATOR = True`)
- run directly `python -m rl_baselines.train --env OmnirobotEnv-v0` with other options customizable. 

#### Generate Data from Omnirobot Simulator
- run directly `python -m environments.dataset_generator --env OmnirobotEnv-v0` with other options customizable. 

### Known issues of Omnirobot
- error: `No module named 'scipy.spatial.transform'`, use `pip3 install scipy==1.2` to solve it
- `omnirobot_server.py` in robotics-rl-srl cannot be simply quitted by ctrl-c.
    - This is because the zmq in python2 uses blocking behavior, even SIGINT cannot be detected when it is blocking.
    - To quit the program, you should send SIGKILL to it. This can be done by `kill -9` or `htop`.
- `ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type`
    - You probably run a program expected to run in `conda` environment, sometimes even `~/.bashrc` is changed, and correctly applies `source ~/.bashrc`, the environment still stays with `ros`.
    - In this situation, simply re-check the contents in `~/.bashrc`, and open another new terminal to launch the programme.
- stuck at `wait for client to connect` or `waiting to connect server`, there are several possible reasons.
    - Port for client and server are not same. Try to use the same one
    - Port is occupied by another client/server, you should kill it. If you cannot find the process which occupies this port, use `fuser 7777\tcp -k` to kill it directly. (7777 can be changed to any number of port).

### TODO
  - ~~extract the same code from omnirobot and its simulator~~
  - ~~add minimum constraint for continous action, using action space~~
  - joint states
  - second camera