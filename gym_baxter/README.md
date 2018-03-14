gym-baxter

# BaxterButtonGymEnv description
The Baxter environment is a single agent domain featuring discrete state and action spaces. baxter_env is an OpenAI Gym environment that sends requests to the server (whose role is to communicate with Gazebo). Gazebo_server.py is the bridge between zmq (sockets) and gazebo.

Goals: Pushing a button


# REQUIREMENTS
* Tested with Python 3.5, but does not really matter, and the same for the CUDA version. To
assure you have the same requirements:

1) Clone OpenAI baselines (https://github.com/openai/baselines) and install according to its README


2) Install requirements by using environment.yml in the repo. This will take care of mujoco, PyTorch, OpenAI Gym, etc.
a) If you are using Anaconda (recommended), create the environment from the environment.yml file:   
```
conda env create -f environment.yml   # adopts the name of given environment
```

b) If you are not using Anaconda:
```
pip install -r requirements.txt
```

3) If you are using Python 3, for this environment to work, remember to comment from your ~/.bashrc file the line including your ROS distribution path:
#source /opt/ros/{indigo/kinetic}/setup.bash
and just run the above command before running the gazebo server and clients:
python -m gazebo.gazebo_server
python -m gazebo.teleop_client


4) Install OpenAI Baselines algorithms from https://github.com/openai/baselines


# RUNNING THE ENVIRONMENT
To run Baxter Gym Environment:
1) Start ROS + Gazebo modules (outside conda env):
roslaunch arm_scenario_simulator baxter_world.launch
rosrun arm_scenario_simulator spawn_objects_example

2) Then start the server:
python -m gazebo.gazebo_server

3) Test this module program in the main repo directory (within your conda env py35):
python -m gym_baxter.test_baxter_env



# TROUBLESHOOTING

Q: Installing OpenAI baselines:
pip install -e .
Command "/home/your_login/anaconda2/envs/py35/bin/python -u -c "import setuptools, tokenize;__file__='/tmp/pip-build-cobxuqz0/mujoco-py/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /tmp/pip-6crcjes7-record/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /tmp/pip-build-cobxuqz0/mujoco-py/

A: Until this issue is fixed, remove mujoco from this list:
https://github.com/openai/baselines/blob/master/setup.py#L13

Q: Running tests:  Error while finding module specification for 'environments.test_env' (ImportError: No module named 'environments')

A: Call it as a python module always at the root of the repo:
python -m environments.test_env


Q: Using opencv:cv2.imshow("Image", np.zeros((10, 10, 3), dtype=np.uint8))
cv2.error: /feedstock_root/build_artefacts/opencv_1489509237254/work/opencv-3.1.0/modules/highgui/src/window.cpp:545: error: (-2) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function cvShowImage

A:https://stackoverflow.com/questions/40207011/opencv-not-working-properly-with-python-on-linux-with-anaconda-getting-error-th/43526627
1. You need to have OpenCV installed and that was compiled with GUI support, so if you installed it with anaconda, you should use menpo channel version, instead of conda-forge channel. I.e., inside your conda Py3 env (for OpenCV 3.1) do:
conda install -c menpo opencv3
(mempo repo will fix the graphical interface dependencies of openCV)

A.1: As alternative to cv2.imshow, use matplotlib:
import cv2, matplotlib.pyplot as plt
img = cv2.imread('img.jpg',0)
plt.imshow(img, cmap='gray')
plt.show()
3.Or try to build library by your own with option WITH_GTK=ON , or smth like that.

Q: Gym's tensorflow:
self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)
TypeError: __init__() got an unexpected keyword argument 'dtype'

A: See issue: https://github.com/openai/baselines/issues/286  Update Gym (see Gym's README):  (print(gym.__version__) must return > 0.95.)
```
cd gym; pip install -e .
```




### ROS dependencies troubleshooting (only for Baxter simulator)

Q: roslaunch arm_scenario_simulator baxter_world.launch
Traceback (most recent call last):
  File "/opt/ros/indigo/bin/roslaunch", line 34, in <module>
    import roslaunch
ImportError: No module named roslaunch

A: Test first roslaunch command, and also run Python, and import rospy to check all is installed. If not: When not working with ROS, just with the environments, the following lines should be commented in the ~/.bashrc: (requires having 2 terminals, one on Py 2 for Gazebo and Ros and other on Py3 via anaconda py35 env.)
```
# Comment when not using ROS
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$(rospack find arm_scenario_simulator)/models:$GAZEBO_MODEL_PATH

# added by Anaconda2 installer. Comment when using ROS
# export PATH="/home/your_login/anaconda2/bin:$PATH"
# export PATH=~/anaconda2/bin:$PATH
```

Q:  python -m gazebo.gazebo_server
(...) "checkrc.pxd", line 21, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:6058)
zmq.error.ZMQError: Address already in use. See http://zguide.zeromq.org/page:all#toc17

A: sudo netstat -ltnp, See the process owning the port (because we us 7777, do
   sudo netstat -lpn | grep :7777
 Kill it with kill -9 <pid>
B: Close sockets when exiting the program but also call zmq_ctx_destroy(). This destroys the context. Also close the socket, calling context.term()


# TO-DO
Add to OpenAI GYM as at the end of:
https://github.com/openai/gym/tree/master/gym/envs
