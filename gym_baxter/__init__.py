import gym
from gym.envs.registration import registry


def register(_id, **kvargs):
    if _id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(_id, **kvargs)


register(
    _id='Baxter-v0',
    entry_point='environments.gym_baxter.envs:baxter_env:BaxterEnv',
    timestep_limit=1000,
     # Threshold at which the environment is considered as solved
    reward_threshold=5.0,
)
