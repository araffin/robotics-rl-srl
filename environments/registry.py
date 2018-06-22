import gym
from gym.envs import registry

from environments.srl_env import SRLGymEnv
from environments.kuka_gym.kuka_button_gym_env import KukaButtonGymEnv
from environments.kuka_gym.kuka_rand_button_gym_env import KukaRandButtonGymEnv
from environments.kuka_gym.kuka_2button_gym_env import Kuka2ButtonGymEnv
from environments.kuka_gym.kuka_moving_button_gym_env import KukaMovingButtonGymEnv
from environments.mobile_robot.mobile_robot_env import MobileRobotGymEnv
from environments.gym_baxter.baxter_env import BaxterEnv
from environments.robobo_gym.mobile_robot_env import RoboboEnv


def register(_id, **kvargs):
    if _id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(_id, **kvargs)


registered_env = {
    "KukaButtonGymEnv-v0":       (KukaButtonGymEnv, SRLGymEnv),
    "KukaRandButtonGymEnv-v0":   (KukaRandButtonGymEnv, KukaButtonGymEnv),
    "Kuka2ButtonGymEnv-v0":      (Kuka2ButtonGymEnv, KukaButtonGymEnv),
    "KukaMovingButtonGymEnv-v0": (KukaMovingButtonGymEnv, KukaButtonGymEnv),
    "MobileRobotGymEnv-v0":      (MobileRobotGymEnv, SRLGymEnv),
    "Baxter-v0":                 (BaxterEnv, SRLGymEnv),
    "RoboboGymEnv-v0":           (RoboboEnv, SRLGymEnv)
}


for name, (env_class, _) in registered_env.items():
    register(
        _id=name,
        entry_point=env_class.__module__ + ":" + env_class.__name__,
        timestep_limit=None,  # This limit is changed in the file
        reward_threshold=None  # Threshold at which the environment is considered as solved
    )

