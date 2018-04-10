"""
Enjoy script for OpenAI Baselines
"""
from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy
from baselines.ppo2.policies import CnnPolicy, MlpPolicy
from baselines.common import tf_util
from baselines.common import set_global_seeds
from baselines import deepq

import rl_baselines.ddpg as ddpg
import rl_baselines.ars as ars
from rl_baselines.utils import createTensorflowSession
from rl_baselines.utils import computeMeanReward
from rl_baselines.policies import MlpPolicyDicrete, AcerMlpPolicy, CNNPolicyContinuous
from srl_priors.utils import printYellow
from replay.enjoy import parseArguments


supported_models = ['acer', 'ppo2', 'a2c', 'deepq', 'ddpg', 'ars']
load_args, train_args, load_path, log_dir, algo, envs = parseArguments(supported_models)

nstack = train_args['num_stack']
ob_space = envs.observation_space
ac_space = envs.action_space


tf.reset_default_graph()
set_global_seeds(load_args.seed)
createTensorflowSession()

sess = tf_util.make_session()
printYellow("Compiling Policy function....")
if algo == "acer":
    policy = {'cnn': AcerCnnPolicy, 'mlp': AcerMlpPolicy}[train_args["policy"]]
    # nstack is already handled in the VecFrameStack
    model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, nstack=1, reuse=False)
elif algo == "a2c":
    policy = {'cnn': CnnPolicy, 'mlp': MlpPolicyDicrete}[train_args["policy"]]
    model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)
elif algo == "ppo2":
    if train_args["continuous_actions"]:
        policy = {'cnn': CNNPolicyContinuous, 'mlp': MlpPolicy}[train_args["policy"]]
    else:
        policy = {'cnn': CnnPolicy, 'mlp': MlpPolicyDicrete}[train_args["policy"]]
    model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)
elif algo == "ddpg":
    model = ddpg.load(load_path, sess)
elif algo == "ars":
    model = ars.load(load_path)


params = find_trainable_variables("model")

tf.global_variables_initializer().run(session=sess)

# Load weights
if algo in ["acer", "a2c", "ppo2"]:
    loaded_params = joblib.load(load_path)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    ps = sess.run(restores)
elif algo == "deepq":
    model = deepq.load(load_path)
elif algo == "ddpg":
    model.load(load_path)

dones = [False for _ in range(load_args.num_cpu)]
obs = envs.reset()
# print(obs.shape)

n_done = 0
last_n_done = 0
for _ in range(load_args.num_timesteps):
    if algo == "acer":
        actions, state, _ = model.step(obs, state=None, mask=dones)
    elif algo in ["a2c", "ppo2"]:
        actions, _, states, _ = model.step(obs, None, dones)
    elif algo == "deepq":
        actions = model(obs[None])[0]
    elif algo == "ddpg":
        actions = model.pi(obs, apply_noise=False, compute_Q=False)[0]
    elif algo == "ars":
        actions = model.getAction(obs.reshape(1,-1))
    obs, rewards, dones, _ = envs.step(actions)

    if algo in ["deepq", "ddpg"]:
        if dones:
            obs = envs.reset()
        dones = [dones]
    n_done += sum(dones)
    if (n_done - last_n_done) > 1:
        last_n_done = n_done
        _, mean_reward = computeMeanReward(log_dir, n_done)
        print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

_, mean_reward = computeMeanReward(log_dir, n_done)
print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))
