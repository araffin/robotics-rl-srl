import time
import pickle

import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from environments.utils import makeEnv
from rl_baselines.utils import CustomVecNormalize, VecFrameStack
from srl_priors.utils import printYellow


class ARS:
    """
    Augmented Random Search algorithm for gym enviroment
    https://arxiv.org/abs/1803.07055
    :param n_population: (int)
    :param observation_space: (int) vectorized length of the obs space
    :param action_space: (int) vectorized length of the action space
    :param top_population: (int) how many of the population are used in updating
    :param step_size: (float) the step size for the parameter update
    :param exploration_noise: (float) standard deviation of the exploration noise
    :param continuous_actions: (bool)
    :param max_step_amplitude: (float) the maximum amplitude factor for step_size 
    """

    def __init__(self, n_population, observation_space, action_space, top_population=2,
                 step_size=0.02, exploration_noise=0.02, continuous_actions=False, max_step_amplitude=10):
        self.n_population = n_population
        self.top_population = top_population
        self.step_size = step_size
        self.exploration_noise = exploration_noise
        self.continuous_actions = continuous_actions
        self.max_step_amplitude = max_step_amplitude

        # The linear policy, initialized to zero
        self.M = np.zeros((observation_space, action_space))

    def getAction(self, obs, delta=0):
        """
        returns the policy action
        :param obs: ([float]) vectorized observation
        :param delta: ([float]) the exploration noise added to the param (default=0)
        :return: ([float]) the chosen action
        """
        action = np.dot(obs, self.M + delta)

        if not self.continuous_actions:
            action = np.argmax(action)

        return action

    def save(self, save_path):
        """
        :param save_path: (str)
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def train(self, env, callback, num_updates=int(1e6 * 1.1)):
        """
        :param env: (gym enviroment)
        :param callback: (function)
        :param num_updates: (int) the number of updates to do (default=110000)
        """
        start_time = time.time()
        step = 0

        while step < num_updates:
            r = np.zeros((self.n_population, 2))
            delta = np.random.normal(size=(self.n_population,) + self.M.shape)
            done = np.full((self.n_population * 2,), False)
            obs = env.reset()
            while not done.all():
                actions = []
                for k in range(self.n_population):
                    for direction in range(2):
                        if not done[k * 2 + direction]:
                            current_obs = obs[k * 2 + direction].reshape(-1)
                            if direction == 0:
                                action = self.getAction(current_obs, delta=(self.exploration_noise * delta[k]))
                            else:
                                action = self.getAction(current_obs, delta=(-self.exploration_noise * delta[k]))

                            actions.append(action)
                        else:
                            actions.append(None)  # do nothing, as we are done

                obs, reward, new_done, info = env.step(actions)
                step += self.n_population

                done = np.bitwise_or(done, new_done)

                # cumulate the reward for every enviroment that is not finished
                update_idx = ~(done.reshape(self.n_population, 2))
                r[update_idx] += (reward.reshape(self.n_population, 2))[update_idx]

                if callback is not None:
                    callback(locals(), globals())
                if (step / self.n_population + 1) % 500 == 0:
                    print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))

            idx = np.argsort(np.max(r, axis=1))[::-1]

            delta_sum = 0
            for i in range(self.top_population):
                delta_sum += (r[idx[i], 0] - r[idx[i], 1]) * delta[idx[i]]
            # here, we need to be careful with the normalization of step_size, as the variance can be 0 on sparse reward
            self.M += self.step_size / max(self.top_population * np.std(r[idx[:self.top_population]]), 1 / self.max_step_amplitude) * delta_sum


def load(save_path):
    """
    :param save_path: (str)
    :return: (ARS Object)
    """
    with open(save_path, "rb") as f:
        class_dict = pickle.load(f)
    model = ARS(1, 0, 0)
    model.__dict__ = class_dict
    return model


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-population', help='Number of population (each one has 2 threads)', type=int, default=20)
    parser.add_argument('--exploration-noise', help='The standard deviation of the exploration noise', type=float,
                        default=0.02)
    parser.add_argument('--step-size', help='The step size for param update', type=float, default=0.02)
    parser.add_argument('--top-population', help='Number of top population to use in update', type=int, default=2)
    parser.add_argument('--algo-type', help='"v1" is standard ARS, "v2" is for rolling average normalization.',
                        type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument('--max-step-amplitude', type=float, default=10, 
                        help='Set the maximum update vectors amplitude (mesured in factors of step_size)')
    return parser


def main(args, callback=None, env_kwargs={}):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    :param env_kwargs: (dict) The extra arguments for the environment
    """

    assert args.top_population <= args.num_population, \
        "Cannot select top %d, from population of %d." % (args.top_population, args.num_population)
    assert args.num_population > 1, "The population cannot be less than 2."

    envs = [makeEnv(args.env, args.seed, i, args.log_dir, allow_early_resets=True, env_kwargs=env_kwargs)
            for i in range(args.num_population * 2)]
    envs = SubprocVecEnv(envs)
    envs = VecFrameStack(envs, args.num_stack)

    if args.srl_model != "":
        printYellow("Using MLP policy because working on state representation")
        args.policy = "mlp"
        if args.algo_type == "v2":
            envs = CustomVecNormalize(envs, norm_obs=True, norm_rewards=False)

    if args.continuous_actions:
        action_space = np.prod(envs.action_space.shape)
    else:
        action_space = envs.action_space.n

    model = ARS(
        args.num_population,
        np.prod(envs.observation_space.shape),
        action_space,
        top_population=args.top_population,
        step_size=args.step_size,
        exploration_noise=args.exploration_noise,
        continuous_actions=args.continuous_actions,
        max_step_amplitude=args.max_step_amplitude
    )

    model.train(envs, callback, num_updates=(int(args.num_timesteps) // args.num_population * 2))
