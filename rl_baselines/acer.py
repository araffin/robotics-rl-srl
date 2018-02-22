import argparse

from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env
import rl_baselines.common as common

common.LOG_INTERVAL = 1
common.LOG_DIR = "logs/raw_pixels/acer/"
common.PLOT_TITLE = "Raw Pixels"
common.ALGO = "ACER"


# kuka_env.ACTION_REPEAT = 4

# TODO: save the learned model
def learn(policy, env, seed, nsteps=20, nstack=4, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=5000, replay_ratio=4, replay_start=1000, c=10.0,
          trust_region=True, alpha=0.99, delta=1, callback=None):
    win, win_smooth, win_episodes = None, None, None
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, q_coef=q_coef, gamma=gamma,
                  max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon,
                  total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
                  trust_region=trust_region, alpha=alpha, delta=delta)

    runner = Runner(env=env, model=model, nsteps=nsteps, nstack=nstack)
    if replay_ratio > 0:
        _buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size=buffer_size)
    else:
        _buffer = None
    nbatch = nenvs * nsteps
    acer = Acer(runner, model, _buffer, log_interval)
    acer.tstart = time.time()

    # nbatch samples, 1 on_policy call and multiple off-policy calls
    for acer.steps in range(0, total_timesteps, nbatch):
        acer.call(on_policy=True)
        if callback is not None:
            callback(locals(), globals())

        if replay_ratio > 0 and _buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this


def train(envs, num_timesteps, seed, policy, lrschedule, callback=None):
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        raise ValueError("Policy {} not implemented".format(policy))
    learn(policy_fn, envs, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, callback=callback)


def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env', help='environment ID', default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    args = parser.parse_args()

    common.ENV_NAME = args.env
    envs = [make_env(args.env, 0, i, common.LOG_DIR, pytorch=False)
            for i in range(args.num_cpu)]

    envs = SubprocVecEnv(envs)

    train(envs, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, callback=common.callback)


if __name__ == '__main__':
    main()
