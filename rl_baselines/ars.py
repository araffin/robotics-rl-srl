import time
import pickle

import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env

class ARS:
    def __init__(self, n_population, observation_space, action_space, type='v1', top_population=2, rollout_length=1000, step_size=0.02, exploration_noise=0.02, continuous_actions=False):
        self.n = 0
        self.mu = 0
        self.sigma = 0
        self.new_mu = 0
        self.new_sigma = 0

        self.n_population = n_population
        self.type = type
        self.top_population = top_population
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.exploration_noise = exploration_noise
        self.continuous_actions = continuous_actions
        
        self.M = np.zeros((observation_space,action_space))


    def getAction(self, obs, delta=0):
        # v2 is a rolling normalized version of v1.
        if self.type == "v2":
            self.n += 1
            if self.n == 1:
                # init rolling average
                self.mu = obs
                self.new_mu = self.mu
                self.sigma = self.mu
                self.new_sigma = 0
            else:
                rolling_delta = obs - self.new_mu
                self.new_mu += rolling_delta / self.n
                self.new_sigma += rolling_delta*rolling_delta*(self.n-1)/self.n

            x = (obs - self.mu) / (self.sigma + 1e-8)
            
        else:
            x = obs

        action = np.dot(x, self.M+delta)

        if not self.continuous_actions:
            action = np.argmax(action, axis=1)

        return action

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def train(self, env, callback, num_updates=1e6):
        start_time = time.time()
        step = 0

        for _ in range(num_updates):
            r = np.zeros((self.n_population,2))
            delta = np.random.normal(size=(self.n_population,) + self.M.shape)
            done = np.full((self.n_population*2,), False)
            obs = env.reset()
            for _ in range(self.rollout_length):
                actions = []
                for k in range(self.n_population):
                    for dir in range(2):
                        if not done[k*2+dir]:
                            current_obs = obs[k*2+dir].reshape(-1)
                            if dir == 0:
                                action = self.getAction(current_obs, delta=(self.exploration_noise*delta[k]))
                            else:
                                action = self.getAction(current_obs, delta=(-self.exploration_noise*delta[k]))

                            actions.append(action)
                        else:
                            actions.append(np.zeros(self.M.shape[1])) # do nothing, as we are done

                obs, reward, done, info = env.step(actions)
                step += 1 

                # cumulate the reward for every enviroment that is not finished
                update_idx = ~(done.reshape(self.n_population,2))
                r[update_idx] += (reward.reshape(self.n_population,2))[update_idx]

                if callback is not None:
                    callback(locals(), globals())
                if (step + 1) % 500 == 0:
                    print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))

                # Should all enviroments end before the rollout_length, stop the loop
                if done.all():
                    break

            if type == "v2":
                self.mu = self.new_mu
                self.sigma = np.sqrt(self.new_sigma / (self.n-1))

            idx = np.argsort(np.max(r, axis=1))[::-1]

            delta_sum = 0
            for i in range(self.top_population):
                delta_sum += (r[idx[i],0] - r[idx[i],1]) * delta[idx[i]]
            self.M += self.step_size/(self.top_population*np.std(r[idx[:self.top_population]])) * delta_sum




def load(save_path):
    with open(save_path, "rb") as f:
        class_dict = pickle.load(f)
    model = ARS(1,0,0)
    model.__dict__ = class_dict
    return model

def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-population', help='Number of population (each one has 2 threads)', type=int, default=20)
    parser.add_argument('--exploration-noise', help='The standard deviation of the exploration noise', type=float, default=0.02)
    parser.add_argument('--step-size', help='The step size for param update', type=float, default=0.02)
    parser.add_argument('--top-population', help='Number of top population to use in update', type=int, default=2)
    parser.add_argument('--type', help='"v1" is standard ARS, "v2" is for rolling average normalization.', type=str, default="v1")
    parser.add_argument('--rollout-length', help='The max number of rollout for each episodes', type=int, default=1000)
    return parser


def main(args, callback=None):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """

    assert kuka_env.MAX_STEPS <= args.rollout_length, "rollout_length cannot be less than an episode of the enviroment (%d)." % kuka_env.MAX_STEPS
    assert args.top_population <= args.num_population, "Cannot select top %d, from population of %d." % (args.top_population, args.num_population)

    envs = [make_env(args.env, args.seed, i, args.log_dir, pytorch=False)
            for i in range(args.num_population*2)]

    if len(envs) == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)
    envs = VecNormalize(envs)

    if args.continuous_actions:
        action_space = np.prod(envs.action_space.shape)
    else:
        action_space = envs.action_space.n

    model = ARS(
        args.num_population, 
        np.prod(envs.observation_space.shape), 
        action_space,
        type=args.type, 
        top_population=args.top_population, 
        rollout_length=args.rollout_length,
        step_size=args.step_size, 
        exploration_noise=args.exploration_noise, 
        continuous_actions=args.continuous_actions
    )

    model.train(envs, callback, num_updates=(int(args.num_timesteps) // args.num_population*2))



        
