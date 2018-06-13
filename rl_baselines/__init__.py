from .a2c import A2CModel
from .acer import ACERModel
from .ars import ARSModel
from .cma_es import CMAESModel
from .ddpg import DDPGModel
from .deepq import DeepQModel
from .ppo2 import PPO2Model
from .random_agent import RandomAgentModel

registered_rl = {
    "a2c": A2CModel,
    "acer": ACERModel,
    "ars": ARSModel,
    "cma-es": CMAESModel,
    "ddpg": DDPGModel,
    "deepq": DeepQModel,
    "ppo2": PPO2Model,
    "random_agent": RandomAgentModel
}
