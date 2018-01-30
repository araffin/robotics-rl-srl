import argparse
import os
import types
import sys

import numpy as np
import torch
from torch.autograd import Variable
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import environments
import environments.KukaCamGymEnv as kuka_env
kuka_env.FORCE_RENDER = True

sys.path.insert(0, os.path.abspath("pytorch_agents/"))

from arguments import get_args
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='KukaCamBulletEnv-v1',
                    help='environment to train on (default: KukaCamBulletEnv-v1)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
args = parser.parse_args()


env = make_env(args.env_name, args.seed, 0, None)
env = DummyVecEnv([env])

actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pth"))


if len(env.observation_space.shape) == 1:
    env = VecNormalize(env, ret=False)
    env.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    env._obfilt = types.MethodType(_obfilt, env)
    render_func = env.venv.envs[0].render
else:
    render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)


def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


render_func('human')
obs = env.reset()
update_current_obs(obs)

while True:
    value, action, _, states = actor_critic.act(Variable(current_obs, volatile=True),
                                                Variable(states, volatile=True),
                                                Variable(masks, volatile=True),
                                                deterministic=True)
    states = states.data
    cpu_actions = action.data.squeeze(1).cpu().numpy()
    # Obser reward and next obs
    obs, reward, done, _ = env.step(cpu_actions)

    masks.fill_(0.0 if done else 1.0)

    if current_obs.dim() == 4:
        current_obs *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_obs *= masks
    update_current_obs(obs)

    render_func('human')
