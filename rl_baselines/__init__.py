from enum import Enum


class AlgoType(Enum):
    REINFORCEMENT_LEARNING = 1
    EVOLUTION_STRATEGIES = 2
    OTHER = 3  # used to define other algorithms that can't be run in enjoy_baselines.py (ex: Random_agent)


class ActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2

