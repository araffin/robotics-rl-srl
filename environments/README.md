# Gym Environments

In order to use reinforcement learning methods, we must have an environment.
As such, we have created a few robotic environments, using OpenAI's Gym environments.  

(https://github.com/openai/gym)

## Available environments
- Kuka arm: Here we have a Kuka arm which must reach a target, here a button.
    - KukaButtonGymEnv-v0: Kuka arm with a single button in front.
    - KukaRandButtonGymEnv-v0: Kuka arm with a single button in front, and some randomly positioned objects
    - Kuka2ButtonGymEnv-v0: Kuka arm with 2 buttons next to each others, they must be pressed in the correct order.
    - KukaMovingButtonGymEnv-v0: Kuka arm with a single button in front, slowly moving left to right.
- Mobile robot: Here we have a mobile robot which reach a target position
    - MobileRobotGymEnv-v0: A mobile robot on a 2d terrain where it needs to reach a target position.
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
2. Add your class to the ```registered_env``` dictionary in ```environments/registry.py```, 
using this format ```NAME: (CLASS, SUPER_CLASS)```, where:
    * ```NAME```: is your environment's name.
    * ```CLASS```: is your class that is a subclass of ```SRLGymEnv```.
    * ```SUPER_CLASS```: is the super class of your class, this is for saving all the globals and parameters.
3. Add the name of the environment to ```config/srl_models.yaml```, with the location of the saved model for each SRL model (can point to a dummy location, but must be defined).
4. Now you can call your environment using ```--env NAME``` with ```train.py```, ```pipeline.py``` or ```dataset_generator.py```. 
