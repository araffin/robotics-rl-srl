import gym
from gym.envs.registration import registry, make, spec

def register(id,*args,**kvargs):
	if id in  registry.env_specs:
		return
	else:
		return gym.envs.registration.register(id,*args,**kvargs)

register(
	id='KukaCamBulletEnv-v1',
	entry_point='environments.KukaCamGymEnv:KukaCamGymEnv',
	timestep_limit=1000,
	reward_threshold=5.0,  # Threshold at which the environment is considered as solved
)
