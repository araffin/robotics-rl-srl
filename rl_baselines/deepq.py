import argparse

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import logger

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env
import rl_baselines.common as common

common.LOG_INTERVAL = 100
common.LOG_DIR = "logs/raw_pixels/deepq/"
common.PLOT_TITLE = "Raw Pixels"
common.ALGO = "deepq"


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
    common.ENV_NAME = args.env
    env = make_env(args.env, 0, 0, common.LOG_DIR, pytorch=False)()
    # model = deepq.models.mlp([64])

    # Atari CNN
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
        callback=common.callback
    )
    act.save(common.LOG_DIR + "deepq_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
