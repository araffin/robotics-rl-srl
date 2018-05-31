"""
Random agent: randomly sample actions from the action space
"""
import time

from rl_baselines.utils import createEnvs


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    return parser


def main(args, callback, env_kwargs=None):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    :param env_kwargs: (dict) The extra arguments for the environment
    """
    env = createEnvs(args, env_kwargs=env_kwargs)

    obs = env.reset()
    num_updates = int(args.num_timesteps) // args.num_cpu
    start_time = time.time()

    for step in range(num_updates):
        actions = [env.action_space.sample() for _ in range(args.num_cpu)]
        obs, reward, done, info = env.step(actions)
        if callback is not None:
            callback(locals(), globals())
        if (step + 1) % 500 == 0:
            total_steps = step * args.num_cpu
            print("{} steps - {:.2f} FPS".format(total_steps, total_steps / (time.time() - start_time)))
