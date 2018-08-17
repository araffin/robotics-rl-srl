# Reinforcement learning setting

Reinforcement Learning (RL) algorithms, are agents in an environment that are capable of learning a _policy_ in order to try and maximise an objective function (eg., reward).  
RL algorithms can me modeled as a [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process), where we have:

- Set of states
- Set of actions
- Probability of transition from state s to state s' under action a
- Reward for every transition from state s to state s' under action a
- The observation the agent can see

We define a Policy, as a function that returns an action or the probability of taking action, a when in state s.  
Here the RL algorithm, will create a policy from the observations and the rewards obtained from interacting with the environment through actions.  
In our case, the policies of the RL algorithms will be deep neural networks.


## Available algorithms
- A2C: A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C).
- ACER: Sample Efficient Actor-Critic with Experience Replay
- ACKTR: Actor Critic using Kronecker-Factored Trust Region
- ARS: Augmented Random Search (https://arxiv.org/abs/1803.07055)
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- DDPG: Deep Deterministic Policy Gradients
- DeepQ: and variants (Double, Dueling, prioritized experience replay)
- PPO1: Proximal Policy Optimization (MPI Implementation)
- PPO2: Proximal Policy Optimization (GPU Implementation)
- SAC: Soft Actor Critic
- TPRO: Trust Region Policy Optimization (MPI Implementation)

## Add your own
1. Create a class that inherits ```rl_baselines.base_classes.BaseRLObject``` which implements your algorithm.
You will need to define specifically:
    * ```save(save_path, _locals=None)```: to save your model during or after training.
    * ```load(load_path, args=None)```: to load and return a saved instance of your class (static function).
    * ```customArguments(parser)```: ```@classmethod``` to define specifics command line arguments from ```train.py``` or ```pipeline.py``` calls, then returns the parser object. 
    * ```getAction(observation, dones=None)```: to get the action from a given observation.
    * ```makeEnv(self, args, env_kwargs=None, load_path_normalise=None)```: override if you need to change
    the environment wrappers (static function).
    * ```train(args, callback, env_kwargs=None, hyperparam=None)```: to create the environment, and train your algorithm on said environment.
    * (OPTIONAL) ```getActionProba(observation, dones=None)```: to get the action probabilities from a given observation. This is used for the action probability plotting in ```replay.enjoy_baselines```.
    * (OPTIONAL) ```getOptParam()```: ```@classmethod``` to return the hyperparameters that can be optimised through the callable argument. Along with the type and range of said parameters.
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
