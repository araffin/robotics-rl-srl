"""
Random agent: randomly sample actions from the action space
"""
import time
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num_cpu', help='Number of processes', type=int, default=1)
    return parser


def main(args, callback=None):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """
    envs = [make_env(args.env, 0, i, args.log_dir, pytorch=False)
            for i in range(args.num_cpu)]

    envs = SubprocVecEnv(envs)
    obs = envs.reset()
    num_updates = int(args.num_timesteps) // args.num_cpu
    start_time = time.time()
    # for j in range(num_updates):
    for step in range(args.num_timesteps):
        actions = [envs.action_space.sample() for _ in range(args.num_cpu)]
        obs, reward, done, info = envs.step(actions)
        if callback is not None:
            callback(locals(), globals())
        if (step + 1) % 500 == 0:
            total_steps = step * args.num_cpu
            print("{} steps - {:.2f} FPS".format(total_steps, total_steps / (time.time() - start_time)))
