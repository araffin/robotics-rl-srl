from enum import Enum

from rl_baselines.rl_algorithm import BaseRLObject
from rl_baselines.a2c import A2CModel
from rl_baselines.acer import ACERModel
from rl_baselines.ars import ARSModel
from rl_baselines.cma_es import CMAESModel
from rl_baselines.ddpg import DDPGModel
from rl_baselines.deepq import DeepQModel
from rl_baselines.ppo2 import PPO2Model
from rl_baselines.random_agent import RandomAgentModel


class AlgoType(Enum):
    Reinforcement_learning = 1
    Evolution_stategies = 2
    Other = 3  # used to define other algorithms that can't be run in enjoy_baselines.py (ex: Random_agent)


class ActionType(Enum):
    Discrete = 1
    Continuous = 2


# Register, name: (algo class, algo type, list of action types)
registered_rl = {
    "a2c":          (A2CModel, AlgoType.Reinforcement_learning, [ActionType.Discrete]),
    "acer":         (ACERModel, AlgoType.Reinforcement_learning, [ActionType.Discrete]),
    "ars":          (ARSModel, AlgoType.Evolution_stategies, [ActionType.Discrete, ActionType.Continuous]),
    "cma-es":       (CMAESModel, AlgoType.Evolution_stategies, [ActionType.Discrete, ActionType.Continuous]),
    "ddpg":         (DDPGModel, AlgoType.Reinforcement_learning, [ActionType.Continuous]),
    "deepq":        (DeepQModel, AlgoType.Reinforcement_learning, [ActionType.Discrete]),
    "ppo2":         (PPO2Model, AlgoType.Reinforcement_learning, [ActionType.Discrete, ActionType.Continuous]),
    "random_agent": (RandomAgentModel, AlgoType.Other, [ActionType.Discrete, ActionType.Continuous])
}

# Checking validity of the registered RL algorithms
for _, val in registered_rl.items():
    assert issubclass(val[0], BaseRLObject), "Error: tried to load {} as a BaseRLObject".format(val[0])
