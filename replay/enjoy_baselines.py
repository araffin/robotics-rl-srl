"""
Enjoy script for ACER
"""
import argparse
import os
import json
from datetime import datetime

from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import tf_util
from baselines.common import set_global_seeds


import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env
from rl_baselines.utils import createTensorflowSession
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
load_args = parser.parse_args()

for algo in ['acer', 'ppo2', 'a2c', 'not_supported']:
    if algo in load_args.log_dir:
        break

if algo == "not_supported":
    raise ValueError("RL algo not supported for replay")
printGreen("\n" + algo + "\n")

load_path = "{}/{}_model.pkl".format(load_args.log_dir, algo)


env_globals = json.load(open(load_args.log_dir + "kuka_env_globals.json", 'r'))
train_args = json.load(open(load_args.log_dir + "args.json", 'r'))

kuka_env.FORCE_RENDER = load_args.render
kuka_env.ACTION_REPEAT = env_globals['ACTION_REPEAT']

# Log dir for testing the agent
log_dir = "/tmp/gym/"
log_dir += "{}/{}/".format(algo, datetime.now().strftime("%m-%d-%y_%Hh%M_%S"))
os.makedirs(log_dir, exist_ok=True)

envs = SubprocVecEnv([make_env(train_args['env'], load_args.seed, i, log_dir, pytorch=False)
                      for i in range(load_args.num_cpu)])

envs = VecFrameStack(envs, train_args['num_stack'])
nstack = train_args['num_stack']
ob_space = envs.observation_space
ac_space = envs.action_space


tf.reset_default_graph()
set_global_seeds(load_args.seed)
createTensorflowSession()

sess = tf_util.make_session()
printYellow("Compiling Policy function....")
if algo == "acer":
    policy = AcerCnnPolicy
    # nstack is already handled in the VecFrameStack
    model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, nstack=1, reuse=False)
elif algo == "ppo2":
    policy = CnnPolicy
    model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)

params = find_trainable_variables("model")

tf.global_variables_initializer().run(session=sess)

# Load weights
loaded_params = joblib.load(load_path)
restores = []
for p, loaded_p in zip(params, loaded_params):
    restores.append(p.assign(loaded_p))
ps = sess.run(restores)

dones = [False for _ in range(load_args.num_cpu)]
obs = envs.reset()
# print(obs.shape)

n_done = 0
last_n_done = 0
for _ in range(load_args.num_timesteps):
    if algo == "acer":
        actions, state, _ = model.step(obs, state=None, mask=dones)
    elif algo == "ppo2":
        actions, _, states, _ = model.step(obs, None, dones)
    obs, rewards, dones, _ = envs.step(actions)
    n_done += sum(dones)
    if (n_done - last_n_done) > 1:
        last_n_done = n_done
        _, mean_reward = computeMeanReward(log_dir, n_done)
        print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

_, mean_reward = computeMeanReward(log_dir, n_done)
print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))
