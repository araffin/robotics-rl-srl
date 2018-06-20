# Reinforcement learning algorithms

## Available algorithms
- DeepQ: and variants (Double, Dueling, prioritized experience replay)
- ACER: Sample Efficient Actor-Critic with Experience Replay
- A2C: A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C) which gives equal performance.
- PPO2: Proximal Policy Optimization (GPU Implementation)
- DDPG: Deep Deterministic Policy Gradients
- ARS: Augmented Random Search
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy

## Add your own
1. Create a class that inherits ```rl_baselines.base_classes.BaseRLObject``` which implements your algorithm. 
You will need to define specifically: 
    * ```save(save_path, _locals=None)```: to save your model during or after training.
    * ```load(load_path, args=None)```: to load and return a saved instance of your class (static function).
    * ```customArguments(parser)```: to define specifics flags from ```train.py``` or ```pipeline.py``` calls, then returns the parser object. 
    * ```getAction(observation, dones=None)```: to get the action from a given observation.
    * ```makeEnv(self, args, env_kwargs=None, load_path_normalise=None)```: override if you need to change 
    the environment wrappers (static function).
    * ```train(args, callback, env_kwargs=None)```: to create the environment, and train your algorithm on said environment.
    * (OPTIONAL) ```getActionProba(observation, dones=None)```: to get the action probabilities from a given observation. This is used for the action probability plotting in ```replay.enjoy_baselines```.
2. Add your class to the ```registered_rl``` dictionary in ```rl_baselines/registry.py```, 
using this format ```NAME: (CLASS, ALGO_TYPE, [ACTION_TYPE])```, where:
    * ```NAME```: is your algorithm's name.
    * ```CLASS```: is your class that inherits ```BaseRLObject```.
    * ```ALGO_TYPE```: is the type of algorithm, defined by the enumerator ```AlgoType``` in ```rl_baselines/__init__.py```,
    can be ```REINFORCEMENT_LEARNING```, ```EVOLUTION_STRATEGIES``` or ```OTHER``` 
    (```OTHER``` is used to define algorithms that can't be run in ```enjoy_baselines.py``` (ex: Random_agent)).
    * ```[ACTION_TYPE]```: is the list of compatible type of actions, defined by the enumerator ```ActionType``` 
    in ```rl_baselines/__init__.py```, can be ```CONTINUOUS``` and/or ```DISCRETE```.
3. Now you can call your algorithm using ```--algo NAME``` with ```train.py``` or ```pipeline.py```. 