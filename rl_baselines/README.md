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
1. Create a class that inherits ```rl_baselines.rl_algorithm.BaseRLObject``` that implements your algorithm, specificaly: 
    * ```save(save_path, _locals=None)```: to save your model during or after training
    * ```load(load_path, args=None)```: to load the object of your class with variables
    * ```customArguments(parser)```: to parse flags from ```train.py``` call
    * ```getAction(observation, dones=None)```: to get the action from a given observation
    * ```train(args, callback, env_kwargs=None)```: to train you algorithm    
2. Add you class to the ```registered_rl``` dictionary in ```rl_baselines/__init__.py```, 
using this format ```NAME: (CLASS, ALGO_TYPE, [ACTION_TYPE])```, where:
    * ```NAME```: is your algorithm's name.
    * ```CLASS```: is your class that inherits ```BaseRLObject```.
    * ```ALGO_TYPE```: is the type of algorithm, defined by ```AlgoType``` in ```rl_baselines/__init__.py```,
    can be ```Reinforcement_learning```, ```Evolution_stategies``` or ```Other``` 
    (```Other``` is used to define algorithms that can't be run in enjoy_baselines.py (ex: Random_agent)).
    * ```[ACTION_TYPE]```: is the list of compatible action types, 
    defined by ```ActionType``` in ```rl_baselines/__init__.py```, can be ```Continuous``` and/or ```Discrete```.
3. Now you can call you algorithm using ```--algo NAME``` with ```train.py``` or ```pipeline.py```. 