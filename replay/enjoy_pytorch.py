import numpy as np
import torch as th
from torch.autograd import Variable

from rl_baselines.random_search import initNetwork
from rl_baselines.utils import computeMeanReward
from replay.enjoy import parseArguments


supported_models = ['ppo_pytorch', 'a2c_pytorch', 'random_search']
load_args, train_args, load_path, log_dir, algo, envs = parseArguments(supported_models, pytorch=True)

load_args.cuda = not load_args.no_cuda and th.cuda.is_available()

obs_shape = envs.observation_space.shape
if len(obs_shape) > 0:
    obs_shape = (obs_shape[0] * train_args['num_stack'], *obs_shape[1:])
else:
    obs_shape = (train_args['num_stack'], *obs_shape[0])

actor_critic = initNetwork(load_args, envs, obs_shape)
current_obs = th.zeros(load_args.num_cpu, *obs_shape)
actor_critic.load_state_dict(th.load(load_path))

# Recurrent policy not supported yet for replay
states = None
masks = None


def update_current_obs(observation):
    n_channels = envs.observation_space.shape[0]
    obs_tensor = th.from_numpy(observation).float()
    if train_args['num_stack'] > 1:
        current_obs[:, :-n_channels] = current_obs[:, n_channels:]
    current_obs[:, -n_channels:] = obs_tensor


obs = envs.reset()
update_current_obs(obs)

if load_args.cuda:
    actor_critic.cuda()
    current_obs = current_obs.cuda()

n_done = 0
last_n_done = 0
for _ in range(load_args.num_timesteps):
    _, actions, _, _ = actor_critic.act(Variable(current_obs, volatile=True),
                                        states,
                                        masks,
                                        deterministic=True)

    cpu_actions = actions.data.squeeze(1).cpu().numpy()

    # Observe reward and next obs
    obs, _, dones, _ = envs.step(cpu_actions)

    # If done then clean the history of observations.
    masks = th.from_numpy(np.array(dones).astype(np.float32).reshape(-1, 1))

    if load_args.cuda:
        masks = masks.cuda()

    if current_obs.dim() == 4:
        current_obs *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_obs *= masks

    update_current_obs(obs)

    n_done += sum(dones)
    if (n_done - last_n_done) > 1:
        last_n_done = n_done
        _, mean_reward = computeMeanReward(log_dir, n_done)
        print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

_, mean_reward = computeMeanReward(log_dir, n_done)
print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))
