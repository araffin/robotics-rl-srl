gym-baxter

# BaxterButtonGymEnv description


The Baxter environment is a single agent domain featuring discrete state and action spaces...

Goals: Pushing a button

Actions

Rewards

Observation_space

baxter_env is an OpenAI Gym environment that sends requests to the server
(whose role is to communicate with Gazebo).
Gazebo_server.py is the bridge between zmq (sockets) and gazebo.


# REQUIREMENTS
* Tested with Python 3.5, but does not really matter, and the same for the CUDA version. To
assure you have the same requirements:

1) Clone OpenAI baselines (https://github.com/openai/baselines):
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

2) Update the path to baselines before installing requirements by editing the following line in environment.yml. E.g.:
  - baselines (/home/antonin/robotics-rl-srl/baselines)==0.1.4
Or edit that line to simply be:
```
- baselines==0.1.4
```
or install baselines from source.


3) Install requirements by using environment.yml in the repo. This will take care of mujoco, PyTorch, OpenAI Gym, etc.
a) If you are using Anaconda (recommended):
```
conda create --name py35
```
If does not work:
Create the environment from the environment.yml file:   
```
conda env create -f environment.yml   # -> how to specify name this way?
or
conda create --name myenv --file spec-file.txt
```
Activate the new environment (macOS and Linux):
```
conda activate myenv
```
and to deactivate:
```
conda deactivate
```

b) Then, update the one you just created:
but before, in order to avoid:
```
torchvision-0. 100% ... Invalid requirement: 'baselines (/home/antonin/Documents/baselines)==0.1.4'
It looks like a path. Does it exist ?
```
-> Edit:

```
conda env update -f environment.yml
```

b) If you are not using Anaconda:
```
pip install -r requirements.txt
```
IMPORTANT, note that from now on, your will not use your new environment, but the installed one; To activate this environment, use:
```
` > source activate py35
```

4) If you are using Python 3, for this environment to work, remember to comment from your ~/.bashrc file the line including your ROS distribution path:
#source /opt/ros/{indigo/kinetic}/setup.bash
and just run the above command before running the gazebo server and clients:
python -m gazebo.gazebo_server
python -m gazebo.teleop_client

5) Install opencv or opencv3 -> TESTED WITH?
```
https://anaconda.org/conda-forge/opencv  
https://anaconda.org/menpo/opencv3
```
NOTE: best one is:
conda install -c menpo opencv3
These did not work:
conda install -c conda-forge opencv
conda install --channel menpo opencv
conda install -c conda-forge opencv=3.3.1


6) Install visdom for visualization:
```
conda install -c conda-forge visdom
```

7) Install pyBullet (should not be needed if you install via conda the environment requirements):  https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
```
pip3 install pybullet
```

5) Install OpenAI Baselines algorithms from https://github.com/openai/baselines

# TO-DO
Add to OpenAI GYM as at the end of:
https://github.com/openai/gym/tree/master/gym/envs




# Troubleshooting

Q: Installing OpenAI baselines:
Command "/home/seurin/anaconda2/envs/py35/bin/python -u -c "import setuptools, tokenize;__file__='/tmp/pip-build-cobxuqz0/mujoco-py/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /tmp/pip-6crcjes7-record/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /tmp/pip-build-cobxuqz0/mujoco-py/

A: Until this issue is fix, remove mujoco from this list:
https://github.com/openai/baselines/blob/master/setup.py#L13

Q: Running tests:  Error while finding module specification for 'environments.test_env' (ImportError: No module named 'environments')

A: you have to call it as a python module always at the root of the repo:
python -m environments.test_env


Q: Using opencv:cv2.imshow("Image", np.zeros((10, 10, 3), dtype=np.uint8))
cv2.error: /feedstock_root/build_artefacts/opencv_1489509237254/work/opencv-3.1.0/modules/highgui/src/window.cpp:545: error: (-2) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function cvShowImage

A:https://stackoverflow.com/questions/40207011/opencv-not-working-properly-with-python-on-linux-with-anaconda-getting-error-th/43526627
conda remove opencv
conda update conda  ->did not work. Instead, from outside the conda env:
conda update -n your_env_name --all
And then, inside your conda Py3 env (for OpenCV 3.1) do:
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

A: Make sure you have protobuf version > 3 (e.g. via  pip freeze)
You can also call it without dtype argument, but best is to update Gym:
```
cd gym
pip install -e .
```


Q: roslaunch arm_scenario_simulator baxter_world.launch
Traceback (most recent call last):
  File "/opt/ros/indigo/bin/roslaunch", line 34, in <module>
    import roslaunch
ImportError: No module named roslaunch

A: Test first roslaunch command, and also run Python, and import rospy to check all is installed. If not: When not working with ROS, just with the environments, the following lines should be commented in the ~/.bashrc:
```
# Comment when not using ROS
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$(rospack find arm_scenario_simulator)/models:$GAZEBO_MODEL_PATH

# added by Anaconda2 installer. Comment when using ROS
# export PATH="/home/seurin/anaconda2/bin:$PATH"
# export PATH=~/anaconda2/bin:$PATH
```

Q: ~/robotics-rl-srl$ python -m gazebo.gazebo_server
(...) "checkrc.pxd", line 21, in zmq.backend.cython.checkrc._check_rc (zmq/backend/cython/socket.c:6058)
zmq.error.ZMQError: Address already in use. See http://zguide.zeromq.org/page:all#toc17

A: sudo netstat -ltnp, See the process owning the port (because we us 7777, do
   sudo netstat -lpn | grep :7777
 Kill it with kill -9 <pid>
B: Close sockets when exiting the program but also call zmq_ctx_destroy(). This destroys the context. Also close the socket, calling context.term()
