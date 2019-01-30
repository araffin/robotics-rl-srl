Table of Contents
=================
  * [Baxter Robot with Gazebo and ROS](#baxter-robot-with-gazebo-and-ros)
  * [Working With a Real Baxter Robot](#working-with-a-real-baxter-robot)
    * [Recording Data With a Random Agent for SRL](#recording-data-with-a-random-agent-for-srl)
    * [RL on a Real Baxter Robot](#rl-on-a-real-baxter-robot)
  * [Working With a Real Robobo](#working-with-a-real-robobo)
    * [Recording Data With a Random Agent for SRL](#recording-data-with-a-random-agent-for-srl-1)
    * [RL on a Real Robobo](#rl-on-a-real-robobo)


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


### Working on Omnirobot Simulator
This simulator uses photoshop tricks to make realistic image of environment.
#### Known issues
- error: `No module named 'scipy.spatial.transform'`, use `pip3 install scipy==1.2` to solve it

#### Version 1
- Launch simulator server
  - Change camera info file's path in omnirobot_simulator_server.py (The camera info file is the file generated from ROS package 'camera_calibration')
  - cd to `real_robots` and then run `python ./omnirobot_simulator_server.py` 
- Recording data
  - run `python -m environments.omnirobot_gym.test_env`