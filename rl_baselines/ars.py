import time

import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env

def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=10)
    return parser


def main(args, callback=None):
    N = args.num_cpu
    nu = 0.02
    b = 2
    step_size = 0.02
    type = "v1"
    rollout_length = 1000
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """

    assert kuka_env.MAX_STEPS <= rollout_length, "rollout_length cannot be less than an episode of the enviroment (%d)." % kuka_env.MAX_STEPS

    envs = [make_env(args.env, args.seed, i, args.log_dir, pytorch=False)
            for i in range(N*2)]

    if len(envs) == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    num_updates = int(args.num_timesteps) // N*2
    start_time = time.time()

    if args.continuous_actions:
        M = np.zeros((np.prod(envs.observation_space.shape),np.prod(envs.action_space.shape)))
    else:
        M = np.zeros((np.prod(envs.observation_space.shape),envs.action_space.n))
        
    n = 0
    step = 0

    for _ in range(num_updates):
        r = np.zeros((N,2))
        delta = np.random.normal(size=(N,) + M.shape)
        done = np.full((N*2,), False)
        obs = envs.reset()
        for i in range(rollout_length):
            actions = []
            for k in range(N):
                for dir in range(2):
                    if not done[k*2+dir]:
                        current_obs = obs[k*2+dir].reshape(-1)
                        if type == "v2":
                            n += 1
                            if n == 1:
                                # init rolling average
                                mu = current_obs
                                new_mu = mu
                                sigma = mu
                                new_sigma = 0
                            else:
                                rolling_delta = current_obs - new_mu
                                new_mu += rolling_delta / n
                                new_sigma += rolling_delta*rolling_delta*(n-1)/n

                            x = (current_obs - mu) / (sigma + 1e-8)
                            
                        else:
                            x = current_obs

                        if dir == 0:
                            action = np.dot(x, M+nu*delta[k])
                        else:
                            action = np.dot(x, M-nu*delta[k])

                        if not args.continuous_actions:
                            action = np.argmax(action, axis=1)

                        actions.append(action)
                    else:
                        actions.append(np.zeros(M.shape[1])) # do nothing, as we are done

            obs, reward, done, info = envs.step(actions)
            step += 1 

            # cumulate the reward for every enviroment that is not finished
            update_idx = ~(done.reshape(N,2))
            r[update_idx] += (reward.reshape(N,2))[update_idx]

            if callback is not None:
                callback(locals(), globals())
            if (step + 1) % 500 == 0:
                print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))

            # Should all enviroments end before the rollout_length, stop the loop
            if done.all():
                break

        if type == "v2":
            mu = new_mu
            sigma = np.sqrt(new_sigma / (n-1))

        idx = np.argsort(np.max(r, axis=1))[::-1]

        delta_sum = 0
        for i in range(b):
            delta_sum += (r[idx[i],0] - r[idx[i],1]) * delta[idx[i]]
        M += step_size/(b*np.std(r[idx[:b]])) * delta_sum