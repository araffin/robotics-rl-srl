import argparse
import os
import json
from datetime import datetime

import yaml
import torch as th
from torch.autograd import Variable
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from pytorch_agents.envs import make_env
import environments.kuka_button_gym_env as kuka_env
from rl_baselines.random_search import initNetwork
from rl_baselines.utils import computeMeanReward
from srl_priors.utils import printGreen, printYellow


parser = argparse.ArgumentParser(description="Load trained RL model")
parser.add_argument('--env', help='environment ID', default='KukaButtonGymEnv-v0')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
parser.add_argument('--log-dir', help='folder with the saved agent model', required=True)
parser.add_argument('--num-timesteps', type=int, default=int(10e3))
parser.add_argument('--render', action='store_true', default=False,
                    help='Render the environment (show the GUI)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
load_args = parser.parse_args()

load_args.cuda = not load_args.no_cuda and th.cuda.is_available()

with open('config/srl_models.yaml', 'rb') as f:
    models = yaml.load(f)

for algo in ['ppo', 'a2c', 'random_search', 'not_supported']:
    if algo in load_args.log_dir:
        break

if algo == "not_supported":
    raise ValueError("RL algo not supported for replay")
printGreen("\n" + algo + "\n")

load_path = "{}/{}_model.pth".format(load_args.log_dir, algo)

env_globals = json.load(open(load_args.log_dir + "kuka_env_globals.json", 'r'))
train_args = json.load(open(load_args.log_dir + "args.json", 'r'))

kuka_env.FORCE_RENDER = load_args.render
kuka_env.ACTION_REPEAT = env_globals['ACTION_REPEAT']

# Log dir for testing the agent
log_dir = "/tmp/gym/"
log_dir += "{}/{}/".format(algo, datetime.now().strftime("%m-%d-%y_%Hh%M_%S"))
os.makedirs(log_dir, exist_ok=True)


envs = SubprocVecEnv([make_env(train_args['env'], load_args.seed, i, log_dir, pytorch=True)
                      for i in range(load_args.num_cpu)])
# envs = VecFrameStack(envs, train_args['num_stack'])


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


def update_current_obs(obs):
    n_channels = envs.observation_space.shape[0]
    obs = th.from_numpy(obs).float()
    if train_args['num_stack'] > 1:
        current_obs[:, :-n_channels] = current_obs[:, n_channels:]
    current_obs[:, -n_channels:] = obs


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
    print(cpu_actions)

    # Observe reward and next obs
    obs, _, dones, _ = envs.step(cpu_actions)

    # If done then clean the history of observations.
    masks = th.FloatTensor([[0.0] if done else [1.0] for done in dones])

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
