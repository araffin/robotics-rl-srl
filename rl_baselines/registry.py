from rl_baselines import AlgoType, ActionType
from rl_baselines.base_classes import BaseRLObject
from rl_baselines.rl_algorithm.a2c import A2CModel
from rl_baselines.rl_algorithm.acer import ACERModel
from rl_baselines.rl_algorithm.acktr import ACKTRModel
from rl_baselines.evolution_strategies.ars import ARSModel
from rl_baselines.evolution_strategies.cma_es import CMAESModel
from rl_baselines.rl_algorithm.ddpg import DDPGModel
from rl_baselines.rl_algorithm.deepq import DQNModel
from rl_baselines.rl_algorithm.ppo1 import PPO1Model
from rl_baselines.rl_algorithm.ppo2 import PPO2Model
from rl_baselines.random_agent import RandomAgentModel
from rl_baselines.rl_algorithm.sac import SACModel
from rl_baselines.rl_algorithm.trpo import TRPOModel

# Register, name: (algo class, algo type, list of action types)
registered_rl = {
    "a2c":          (A2CModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "acer":         (ACERModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE]),
    "acktr":        (ACKTRModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE]),
    "ars":          (ARSModel, AlgoType.EVOLUTION_STRATEGIES, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "cma-es":       (CMAESModel, AlgoType.EVOLUTION_STRATEGIES, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "ddpg":         (DDPGModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.CONTINUOUS]),
    "deepq":        (DQNModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE]),
    "ppo1":         (PPO1Model, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "ppo2":         (PPO2Model, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "random_agent": (RandomAgentModel, AlgoType.OTHER, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "sac":          (SACModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE, ActionType.CONTINUOUS]),
    "trpo":         (TRPOModel, AlgoType.REINFORCEMENT_LEARNING, [ActionType.DISCRETE, ActionType.CONTINUOUS])
}

# Checking validity of the registered RL algorithms
for _, val in registered_rl.items():
    assert issubclass(val[0], BaseRLObject), "Error: tried to load {} as a BaseRLObject".format(val[0])
