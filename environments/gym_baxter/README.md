# BaxterButtonGymEnv Description
The Baxter environment is a single agent domain featuring discrete state and action spaces. baxter_env is an OpenAI Gym environment that sends requests to the server (whose role is to communicate with Gazebo). Gazebo_server.py is the bridge between zmq (sockets) and gazebo.

Goal: Pushing a button

Associated ROS packages:

- [arm scenario experiments](https://github.com/NataliaDiaz/arm_scenario_experiments/tree/rl), branch "rl"
- [arm scenario simulator](https://github.com/araffin/arm_scenario_simulator) branch kinetic-devel


# Requirements

1) Install S-RL Toolbox requirements, cf [documentation](https://s-rl-toolbox.readthedocs.io/en/latest/)

2) If you are using ROS, for this environment to work, remember to comment from your ~/.bashrc file the line including your ROS distribution path:
#source /opt/ros/{indigo/kinetic}/setup.bash
and just run the above command before running the gazebo server and clients:
python -m gazebo.gazebo_server
python -m gazebo.teleop_client


# Running The Environment
To run Baxter Gym Environment:
1) Start ROS + Gazebo modules (outside conda env):
```
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example
```

2) Then start the server that communicates with gazebo (Python 2):
```
python -m gazebo.gazebo_server
```

3) Test this module program in the main repo directory (within your conda env py35):
```
python -m environments.gym_baxter.test_baxter_env
```


# Troubleshooting

Q: Running tests:  Error while finding module specification for 'environments.test_env' (ImportError: No module named 'environments')

A: Call it as a python module always at the root of the repo:
python -m environments.test_env


Q: Using opencv:cv2.imshow("Image", np.zeros((10, 10, 3), dtype=np.uint8))
cv2.error: /feedstock_root/build_artefacts/opencv_1489509237254/work/opencv-3.1.0/modules/highgui/src/window.cpp:545: error: (-2) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function cvShowImage

A:https://stackoverflow.com/questions/40207011/opencv-not-working-properly-with-python-on-linux-with-anaconda-getting-error-th/43526627
1. You need to have OpenCV installed and that was compiled with GUI support, so if you installed it with anaconda, you should use menpo channel version, instead of conda-forge channel. I.e., inside your conda Py3 env (for OpenCV 3.1) do:
conda install -c menpo opencv3
(mempo repo will fix the graphical interface dependencies of openCV)

2.: As alternative to cv2.imshow, use matplotlib (note that matplotlib except RGB image whereas openCV uses BGR images):
import cv2, matplotlib.pyplot as plt
img = cv2.imread('img.jpg',0)
plt.imshow(img, cmap='gray')
plt.show()

3. Build OpenCV from source with gui support.

Q: GLib-GIO-Message: Using the 'memory' GSettings backend.  Your settings will not be saved or shared with other applications.

A: Caused when calling cv2.imshow. https://github.com/conda-forge/glib-feedstock/issues/19  -> Add to your ~/.bashrc:
GIO_EXTRA_MODULES=/usr/lib/x86_64-linux-gnu/gio/modules/


### ROS dependencies troubleshooting (only for Baxter simulator)

```
# Comment when not using ROS
source /opt/ros/kinetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$(rospack find arm_scenario_simulator)/models:$GAZEBO_MODEL_PATH

# added by Anaconda2 installer. Comment when using ROS
# export PATH="/home/your_login/anaconda3/bin:$PATH"
```

Q:  python -m gazebo.gazebo_server
```
(...) "checkrc.pxd", line 21, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:6058)
zmq.error.ZMQError: Address already in use. See http://zguide.zeromq.org/page:all#toc17
```
A1: sudo netstat -ltnp, See the process owning the port (because we use 7777, do
```
   sudo netstat -lpn | grep :7777
```
 and Kill it with kill -9 <pid> or do all at once within
 ```
 kill -9 `sudo lsof -t -i:7777`
 ```
