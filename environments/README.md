# Gym Environments

In order to use reinforcement learning methods, we must have an environment.
As such, we have created a few robotic environments, using OpenAI's Gym environments.  

(https://github.com/openai/gym)

## Available environments
- Kuka arm: Here we have a Kuka arm which must reach a target, here a button.
    - KukaButtonGymEnv-v0: Kuka arm with a single button in front.
    - KukaRandButtonGymEnv-v0: Kuka arm with a single button in front, and some randomly positioned objects
    - Kuka2ButtonGymEnv-v0: Kuka arm with 2 buttons next to each others, they must be pressed in the correct order (lighter button, then darker button).
    - KukaMovingButtonGymEnv-v0: Kuka arm with a single button in front, slowly moving left to right.
- Mobile robot: Here we have a mobile robot which reach a target position
    - MobileRobotGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach a target position (yellow cylinder).
    - MobileRobot2TargetGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach two target positions, in the correct order (yellow target, then red target).
    - MobileRobot1DGymEnv-v0: A mobile robot on a 1d slider where it can only go up and down, it must reach a target position.
    - MobileRobotLineTargetGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach a colored band going across the terrain.
- Racing car: Here we have the interface for the Gym racing car environment. It must complete a racing course in the least time possible (only available in a terminal with X running)
    - CarRacingGymEnv-v0: A racing car on a racing course, it must complete the racing course in the least time possible.
- Baxter: A baxter robot that must reach a target, with its arms.
    - Baxter-v0: A bridge to use a baxter robot with ROS (in simulation, it uses Gazebo)
- Robobo: A Robobo robot that must reach a target position.
    - RoboboGymEnv-v0: A bridge to use a Robobo robot with ROS.

## Add your own
1. Create a class that inherits ```environments.srl_env.SRLGymEnv``` which implements your environment.
You will need to define specifically:
    * ```getTargetPos()```: returns the position of the target.
    * ```getGroundTruthDim()```: returns the number of dimensions used to encode the ground truth.
    * ```getGroundTruth()```: returns the ground truth state.
    * ```step(action)```: step the environment in simulation with the given action.
    * ```reset()```: re-initialise the environment.
    * ```render(mode='human')```: returns an observation of the environment.
    * ```close()```: closes the environment, override if you need to change it.
    * Make sure ```__init__``` has the parameter ```**_kwargs``` in order to ignore useless flag parameters sent by the calling code.
2. Add this code to the same file as the class declaration
```python
def getGlobals():
    """
    :return: (dict)
    """
    return globals()
```
 it will allow the logging of constant values used by the class
 
3. Add your class to the ```registered_env``` dictionary in ```environments/registry.py```,
using this format ```NAME: (CLASS, SUPER_CLASS, PLOT_TYPE, THREAD_TYPE)```, where:
    * ```NAME```: is your environment's name, it must only contain ```[A-Z][a-z][0-9]``` and end with the version number in this format: ```-v{number}```.
    * ```CLASS```: is your class that is a subclass of ```SRLGymEnv```.
    * ```SUPER_CLASS```: is the super class of your class, this is for saving all the globals and parameters.
    * ```PLOT_TYPE```: is the type of plotting for ```replay.enjoy_baselines```,
    defined by the enumerator ```PlottingType``` in ```environments/__init__.py```,
    can be ```PLOT_2D``` or ```PLOT_3D``` (use ```PLOT_3D``` if unsure).
    * ```THREAD_TYPE```: is the type of multithreading supported by the environment,
    defined by the enumerator ```ThreadingType``` in ```environments/__init__.py```,
    can be (from most restricive to less restricive) 
    ```PROCESS```, ```THREAD``` or ```NONE``` (use ```NONE``` if unsure).
4. Add the name of the environment to ```config/srl_models.yaml```, with the location of the saved model for each SRL model (can point to a dummy location, but must be defined).
5. Now you can call your environment using ```--env NAME``` with ```train.py```, ```pipeline.py``` or ```dataset_generator.py```.
