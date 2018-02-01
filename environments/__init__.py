import gym
from gym.envs.registration import registry


def register(_id, **kvargs):
    if _id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(_id, **kvargs)


register(
    _id='KukaButtonGymEnv-v0',
    entry_point='environments.kuka_button_gym_env:KukaButtonGymEnv',
    timestep_limit=1000,
    reward_threshold=5.0,  # Threshold at which the environment is considered as solved
)
