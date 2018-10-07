.. _install:

Installation
------------

**Python 3 is required** (python 2 is not supported because of OpenAI
baselines)


.. note::
  we are using `Stable Baselines <https://github.com/hill-a/stable-baselines.git>`__, a fork of
  OpenAI Baselines with unified interface and other improvements (e.g. tensorboard support).

Using Anaconda
~~~~~~~~~~~~~~

0. Download the project (note the ``--recursive`` argument because we
   are using git submodules):

::

   git clone git@github.com:araffin/robotics-rl-srl.git --recursive

1. Install the swig library:

::

   sudo apt-get install swig

2. Install the dependencies using ``environment.yml`` file (for anaconda
   users) in the current environment

::

   conda env create --file environment.yml
   source activate py35

`PyBullet Documentation <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA>`__

Using Docker
~~~~~~~~~~~~

Use Built Images
^^^^^^^^^^^^^^^^

GPU image (requires
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__):

::

   docker pull araffin/rl-toolbox

CPU only:

::

   docker pull araffin/rl-toolbox-cpu

Build the Docker Images
^^^^^^^^^^^^^^^^^^^^^^^

Build GPU image (with nvidia-docker):

::

   docker build . -f docker/Dockerfile.gpu -t rl-toolbox

Build CPU image:

::

   docker build . -f docker/Dockerfile.cpu -t rl-toolbox-cpu

Note: if you are using a proxy, you need to pass extra params during
build and do some
`tweaks <https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy>`__:

::

   --network=host --build-arg HTTP_PROXY=http://your.proxy.fr:8080/ --build-arg http_proxy=http://your.proxy.fr:8080/ --build-arg HTTPS_PROXY=https://your.proxy.fr:8080/ --build-arg https_proxy=https://your.proxy.fr:8080/

Run the images
^^^^^^^^^^^^^^

Run the nvidia-docker GPU image

::

   docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/tmp/rl_toolbox,type=bind araffin/rl-toolbox bash -c 'source activate py35 && cd /tmp/rl_toolbox/ && python -m rl_baselines.train --srl-model ground_truth --env MobileRobotGymEnv-v0 --no-vis --num-timesteps 1000'

Or, with the shell file:

::

   ./run_docker_gpu.sh python -m rl_baselines.train --srl-model ground_truth --env MobileRobotGymEnv-v0 --no-vis --num-timesteps 1000

Run the docker CPU image

::

   docker run -it --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/tmp/rl_toolbox,type=bind araffin/rl-toolbox-cpu bash -c 'source activate py35 && cd /tmp/rl_toolbox/ && python -m rl_baselines.train --srl-model ground_truth --env MobileRobotGymEnv-v0 --no-vis --num-timesteps 1000'

Or, with the shell file:

::

   ./run_docker_cpu.sh python -m rl_baselines.train --srl-model ground_truth --env MobileRobotGymEnv-v0 --no-vis --num-timesteps 1000

Explanation of the docker command:

-  ``docker run -it`` create an instance of an image (=container), and
   run it interactively (so ctrl+c will work)
-  ``--rm`` option means to remove the container once it exits/stops
   (otherwise, you will have to use ``docker rm``)
-  ``--network host`` don't use network isolation, this allow to use
   visdom on host machine
-  ``--ipc=host`` Use the host system’s IPC namespace. It is needed to
   train SRL model with PyTorch. IPC (POSIX/SysV IPC) namespace provides
   separation of named shared memory segments, semaphores and message
   queues.
-  ``--name test`` give explicitely the name ``test`` to the container,
   otherwise it will be assigned a random name
-  ``--mount src=...`` give access of the local directory (``pwd``
   command) to the container (it will be map to ``/tmp/rl_toolbox``), so
   all the logs created in the container in this folder will be kept
   (for that you need to pass the ``--log-dir logs/`` option)
-  ``bash -c 'source activate py35 && ...`` Activate the conda
   enviroment inside the docker container, and launch an experiment
   (``python -m rl_baselines.train ...``)
