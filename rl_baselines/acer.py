import argparse

from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


import environments
import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.arguments import get_args
from pytorch_agents.envs import make_env
from pytorch_agents.visualize import visdom_plot, episode_plot
from visdom import Visdom

viz = Visdom(port=8097)
log_interval = 1
log_dir = "logs/"
PLOT_TITLE = "Raw Pixels"
log_dir += "raw_pixels/"
algo = "acer"
env_name = "KukaButtonGymEnv-v0"

# kuka_env.ACTION_REPEAT = 4

# TODO: save the learned model
def learn(policy, env, seed, nsteps=20, nstack=4, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=5000, replay_ratio=4, replay_start=1000, c=10.0,
          trust_region=True, alpha=0.99, delta=1):

    win, win_smooth, win_episodes = None, None, None
    # print("Running Acer Simple")
    # print(locals())
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
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

    n_steps = 0

    for acer.steps in range(0, total_timesteps, nbatch): #nbatch samples, 1 on_policy call and multiple off-policy calls
        acer.call(on_policy=True)
        if (n_steps + 1) % log_interval == 0:
            win = visdom_plot(viz, win, log_dir, env_name, algo, bin_size=1, smooth=0, title=PLOT_TITLE)
            win_smooth = visdom_plot(viz, win_smooth, log_dir, env_name, algo, title=PLOT_TITLE + " smoothed")
            win_episodes = episode_plot(viz, win_episodes, log_dir, env_name, algo, window=20, title=PLOT_TITLE + " [Episodes]")

        n_steps += 1

        if replay_ratio > 0 and _buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this



def train(envs, num_timesteps, seed, policy, lrschedule):
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        raise ValueError("Policy {} not implemented".format(policy))
    learn(policy_fn, envs, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)


def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    args = parser.parse_args()
    seed = 0
    num_cpu = 4
    num_timesteps = 1e6

    envs = [make_env("KukaButtonGymEnv-v0", 0, i, "logs/raw_pixels/", pytorch=False)
            for i in range(num_cpu)]

    envs = SubprocVecEnv(envs)

    train(envs, num_timesteps=num_timesteps, seed=seed,
          policy=args.policy, lrschedule=args.lrschedule)


if __name__ == '__main__':
    main()
