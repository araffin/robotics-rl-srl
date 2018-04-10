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
    timestep_limit=500,
    reward_threshold=5.0,  # Threshold at which the environment is considered as solved
)

register(
    _id='KukaRandButtonGymEnv-v0',
    entry_point='environments.kuka_rand_button_gym_env:KukaRandButtonGymEnv',
    timestep_limit=1000,
    reward_threshold=5.0,  # Threshold at which the environment is considered as solved
)

register(
    _id='Kuka2ButtonGymEnv-v0',
    entry_point='environments.kuka_2button_gym_env:Kuka2ButtonGymEnv',
    timestep_limit=1500,
    reward_threshold=5.0,  # Threshold at which the environment is considered as solved
)

register(
    _id='Baxter-v0',
    entry_point='environments.gym_baxter.baxter_env:BaxterEnv',
    timestep_limit=500,
    # Threshold at which the environment is considered as solved
    reward_threshold=5.0,
)
