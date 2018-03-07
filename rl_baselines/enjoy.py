"""
Enjoy script for ACER
"""
import os
import json
from datetime import datetime

from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import tf_util
from baselines.common import set_global_seeds


import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env
from rl_baselines.utils import createTensorflowSession
from rl_baselines.utils import computeMeanReward


log_folder = "logs/raw_pixels/acer/02-03-18_14h07_37/"
algo = "acer"
load_path = log_folder + "acer_model.pkl"

# log_folder = "logs/raw_pixels/ppo2/28-02-18_18h17_01/"
# algo = "ppo2"
# load_path = log_folder + "ppo2_model.pkl"

env_globals = json.load(open(log_folder + "kuka_env_globals.json", 'r'))
args = json.load(open(log_folder + "args.json", 'r'))
# args = {'env': 'KukaButtonGymEnv-v0', 'num_stack': 4}

kuka_env.FORCE_RENDER = False
kuka_env.ACTION_REPEAT = env_globals['ACTION_REPEAT']

log_dir = "/tmp/gym/"
log_dir += "{}/{}/".format("acer", datetime.now().strftime("%d-%m-%y_%Hh%M_%S"))

os.makedirs(log_dir, exist_ok=True)

seed = 0
nenvs = 8
env = SubprocVecEnv([make_env(args['env'], seed, i, log_dir, pytorch=False) for i in range(nenvs)])

nstack = args['num_stack']
ob_space = env.observation_space
ac_space = env.action_space


tf.reset_default_graph()
set_global_seeds(seed)
createTensorflowSession()

sess = tf_util.make_session()
print("Compiling Policy function....")
if algo == "acer":
    policy = AcerCnnPolicy
    model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
elif algo in ['ppo2', 'a2c']:
    raise NotImplementedError("Not yet implemented for algo != ACER")
    # policy = CnnPolicy
    # model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

params = find_trainable_variables("model")

tf.global_variables_initializer().run(session=sess)

loaded_params = joblib.load(load_path)
restores = []
for p, loaded_p in zip(params, loaded_params):
    restores.append(p.assign(loaded_p))
ps = sess.run(restores)

nh, nw, nc = env.observation_space.shape
dones = [False for _ in range(nenvs)]
current_obs = np.zeros((nenvs, nh, nw, nc * nstack), dtype=np.uint8)


def update_obs(current_obs, obs, dones=None):
    if dones is not None:
        current_obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
    current_obs = np.roll(current_obs, shift=-nc, axis=3)
    current_obs[:, :, :, -nc:] = obs[:, :, :, :]


obs = env.reset()
update_obs(current_obs, obs)

n_done = 0
last_n_done = 0
for _ in range(10 * 500):
    actions = model.act(current_obs)
    obs, rewards, dones, _ = env.step(actions)
    n_done += sum(dones)
    update_obs(current_obs, obs, dones)
    if (n_done - last_n_done) > 1:
        last_n_done = n_done
        _, mean_reward = computeMeanReward(log_dir, n_done)
        print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

_, mean_reward = computeMeanReward(log_dir, n_done)
print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))
