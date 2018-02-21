import argparse

import gym
from baselines import deepq
from baselines.common import set_global_seeds
from baselines import logger
from pytorch_agents.visualize import visdom_plot, episode_plot
from visdom import Visdom

import environments
import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env

viz = Visdom(port=8097)
log_interval = 100
log_dir = "logs/"
PLOT_TITLE = "Raw Pixels"
log_dir += "raw_pixels/deepq/"
algo = "deepq"
env_name = "KukaButtonGymEnv-v0"
n_steps = 0

win, win_smooth, win_episodes = None, None, None

def callback(_locals, _globals):
    global win, win_smooth, win_episodes, n_steps
    if (n_steps + 1) % log_interval == 0:
        win = visdom_plot(viz, win, log_dir, env_name, algo, bin_size=1, smooth=0, title=PLOT_TITLE)
        win_smooth = visdom_plot(viz, win_smooth, log_dir, env_name, algo, title=PLOT_TITLE + " smoothed")
        win_episodes = episode_plot(viz, win_episodes, log_dir, env_name, algo, window=20, title=PLOT_TITLE + " [Episodes]")
    n_steps += 1
    return False


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    args = parser.parse_args()

    logger.configure()
    set_global_seeds(args.seed)
    env = make_env(args.env, 0, 0, log_dir, pytorch=False)()
    # model = deepq.models.mlp([64])
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=args.num_timesteps,
        buffer_size=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=500,
        target_network_update_freq=500,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        print_freq=500,
        callback=callback
    )
    # act.save("deepq_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
