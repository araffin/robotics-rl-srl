from __future__ import division, absolute_import, print_function

import argparse
import glob
import multiprocessing
import os
import shutil
import time

import numpy as np
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import CnnPolicy
import tensorflow as tf
import torch as th
from torch.autograd import Variable

from environments import ThreadingType
from environments.registry import registered_env
from environments.utils import makeEnv
from real_robots.constants import *
from replay.enjoy_baselines import createEnv, loadConfigAndSetup
from rl_baselines.utils import MultiprocessSRLModel, loadRunningAverage
from srl_zoo.utils import printRed, printYellow
from srl_zoo.preprocessing.utils import deNormalize
from state_representation.models import loadSRLModel, getSRLDim

RENDER_HEIGHT = 224
RENDER_WIDTH = 224
VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae", "dae", "random"]
VALID_POLICIES = ['walker', 'random', 'ppo2', 'custom']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


def latestPath(path):
    """
    :param path: path to the log folder (defined in srl_model.yaml) (str)
    :return: path to latest learned model in the same dataset folder (str)
    """
    return max([path + d for d in os.listdir(path) if os.path.isdir(path + "/" + d)], key=os.path.getmtime) + '/'


def walkerPath():
    """

    :return:
    """
    eps = 0.01
    N_times = 14
    path = []
    left = [0 for _ in range(N_times)]
    right = [1 for _ in range(N_times)]

    for idx in range(N_times * 2):
        path += left if idx % 2 == 0 else right
        path += [3] if idx < N_times else [2]

    return path


def convertImagePath(args, path, record_id_start):
    """
    Used to convert an image path, from one location, to another
    :param args: (ArgumentParser object)
    :param path: (str)
    :param record_id_start: (int) where does the current part start counting its records
    :return:
    """
    image_name = path.split("/")[-1]
    # get record id for output, by adding the current offset with the record_id
    # of the folder
    new_record_id = record_id_start + int(path.split("/")[-2].split("_")[-1])
    return args.name + "/record_{:03d}".format(new_record_id) + "/" + image_name


def vecEnv(env_kwargs_local, env_class):
    """
    Local Env Wrapper
    :param env_kwargs_local: arguments related to the environment wrapper
    :param env_class: class of the env
    :return: env for the pretrained algo
    """
    train_env = env_class(**{**env_kwargs_local, "record_data": False, "renders": False})
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    return train_env


def env_thread(args, thread_num, partition=True):
    """
    Run a session of an environment
    :param args: (ArgumentParser object)
    :param thread_num: (int) The thread ID of the environment session
    :param partition: (bool) If the output should be in multiple parts (default=True)
    """
    env_kwargs = {
        "max_distance": args.max_distance,
        "random_target": args.random_target,
        "force_down": True,
        "is_discrete": not args.continuous_actions,
        "renders": thread_num == 0 and args.display,
        "record_data": not args.no_record_data,
        "multi_view": args.multi_view,
        "save_path": args.save_path,
        "shape_reward": args.shape_reward,
        "simple_continual_target": args.simple_continual,
        "circular_continual_move": args.circular_continual,
        "square_continual_move": args.square_continual,
        "chasing_continual_move":args.chasing_continual,
        "escape_continual_move": args.escape_continual,
        "short_episodes":  args.short_episodes
    }

    if partition:
        env_kwargs["name"] = args.name + "_part-" + str(thread_num)
    else:
        env_kwargs["name"] = args.name

    load_path, train_args, algo_name, algo_class = None, None, None, None
    model = None
    srl_model = None
    srl_state_dim = 0
    generated_obs = None

    if args.run_policy in ["walker", "custom"]:
        if args.latest:
            args.log_dir = latestPath(args.log_custom_policy)
        else:
            args.log_dir = args.log_custom_policy
        args.log_dir = args.log_custom_policy
        args.render = args.display
        args.plotting, args.action_proba = False, False

        train_args, load_path, algo_name, algo_class, _, env_kwargs_extra = loadConfigAndSetup(args)
        env_kwargs["srl_model"] = env_kwargs_extra["srl_model"]
        env_kwargs["random_target"] = env_kwargs_extra.get("random_target", False)
        env_kwargs["use_srl"] = env_kwargs_extra.get("use_srl", False)

        # TODO REFACTOR
        env_kwargs["simple_continual_target"] = env_kwargs_extra.get("simple_continual_target", False)
        env_kwargs["circular_continual_move"] = env_kwargs_extra.get("circular_continual_move", False)
        env_kwargs["square_continual_move"] = env_kwargs_extra.get("square_continual_move", False)
        env_kwargs["eight_continual_move"] = env_kwargs_extra.get("eight_continual_move", False)
        env_kwargs["chasing_continual_move"] = env_kwargs_extra.get("chasing_continual_move",False)
        env_kwargs["escape_continual_move"] = env_kwargs_extra.get("escape_continual_move", False)


        eps = 0.2
        env_kwargs["state_init_override"] = np.array([MIN_X + eps, MAX_X - eps]) \
            if args.run_policy == 'walker' else None
        if env_kwargs["use_srl"]:
            env_kwargs["srl_model_path"] = env_kwargs_extra.get("srl_model_path", None)
            env_kwargs["state_dim"] = getSRLDim(env_kwargs_extra.get("srl_model_path", None))
            srl_model = MultiprocessSRLModel(num_cpu=args.num_cpu, env_id=args.env, env_kwargs=env_kwargs)
            env_kwargs["srl_pipe"] = srl_model.pipe

    env_class = registered_env[args.env][0]
    env = env_class(**env_kwargs)

    if env_kwargs.get('srl_model', None) not in ["raw_pixels", None]:
        # TODO: Remove env duplication
        # This is a dirty trick to normalize the obs.
        # So for as we override SRL environment functions (step, reset) for on-policy generation & generative replay
        # using stable-baselines' normalisation wrappers (step & reset) breaks...
        env_norm = [makeEnv(args.env, args.seed, i, args.log_dir, allow_early_resets=False, env_kwargs=env_kwargs)
                    for i in range(args.num_cpu)]
        env_norm = DummyVecEnv(env_norm)
        env_norm = VecNormalize(env_norm, norm_obs=True, norm_reward=False)
        env_norm = loadRunningAverage(env_norm, load_path_normalise=args.log_custom_policy)
    using_real_omnibot = args.env == "OmnirobotEnv-v0" and USING_OMNIROBOT

    walker_path = None
    action_walker = None
    state_init_for_walker = None
    kwargs_reset, kwargs_step = {}, {}

    if args.run_policy in ['custom', 'ppo2', 'walker']:
        # Additional env when using a trained agent to generate data
        train_env = vecEnv(env_kwargs, env_class)

        if args.run_policy == 'ppo2':
            model = PPO2(CnnPolicy, train_env).learn(args.ppo2_timesteps)
        else:
            _, _, algo_args = createEnv(args, train_args, algo_name, algo_class, env_kwargs)
            tf.reset_default_graph()
            set_global_seeds(args.seed % 2 ^ 32)
            printYellow("Compiling Policy function....")
            model = algo_class.load(load_path, args=algo_args)
            if args.run_policy == 'walker':
                walker_path = walkerPath()

    if len(args.replay_generative_model) > 0:
        srl_model = loadSRLModel(args.log_generative_model, th.cuda.is_available())
        srl_state_dim = srl_model.state_dim
        srl_model = srl_model.model.model

    frames = 0
    start_time = time.time()

    # divide evenly, then do an extra one for only some of them in order to get the right count
    for i_episode in range(args.num_episode // args.num_cpu + 1 * (args.num_episode % args.num_cpu > thread_num)):

        # seed + position in this slice + size of slice (with reminder if uneven partitions)
        seed = args.seed + i_episode + args.num_episode // args.num_cpu * thread_num + \
               (thread_num if thread_num <= args.num_episode % args.num_cpu else args.num_episode % args.num_cpu)
        seed = seed % 2 ^ 32
        if not (args.run_policy in ['custom', 'walker']):
            env.seed(seed)
            env.action_space.seed(seed)  # this is for the sample() function from gym.space

        if len(args.replay_generative_model) > 0:

            sample = Variable(th.randn(1, srl_state_dim))
            if th.cuda.is_available():
                sample = sample.cuda()

            generated_obs = srl_model.decode(sample)
            generated_obs = generated_obs[0].detach().cpu().numpy()
            generated_obs = deNormalize(generated_obs)

            kwargs_reset['generated_observation'] = generated_obs
        obs = env.reset(**kwargs_reset)
        done = False
        action_proba = None
        t = 0
        episode_toward_target_on = False

        while not done:

            env.render()

            # Policy to run on the fly - to be trained before generation
            if args.run_policy == 'ppo2':
                action, _ = model.predict([obs])

            # Custom pre-trained Policy (SRL or End-to-End)
            elif args.run_policy in['custom', 'walker']:
                obs = env_norm._normalize_observation(obs)
                action = [model.getAction(obs, done)]
                action_proba = model.getActionProba(obs, done)
                if args.run_policy == 'walker':
                    action_walker = np.array(walker_path[t])
            # Random Policy
            else:
                # Using a target reaching policy (untrained, from camera) when collecting data from real OmniRobot
                if episode_toward_target_on and np.random.rand() < args.toward_target_timesteps_proportion and \
                        using_real_omnibot:
                    action = [env.actionPolicyTowardTarget()]
                else:
                    action = [env.action_space.sample()]

            # Generative replay +/- for on-policy action
            if len(args.replay_generative_model) > 0:

                if args.run_policy == 'custom':
                    obs = obs.reshape(1, srl_state_dim)
                    obs = th.from_numpy(obs.astype(np.float32)).cuda()
                    z = obs
                    generated_obs = srl_model.decode(z)
                else:
                    sample = Variable(th.randn(1, srl_state_dim))

                    if th.cuda.is_available():
                        sample = sample.cuda()

                    generated_obs = srl_model.decode(sample)
                generated_obs = generated_obs[0].detach().cpu().numpy()
                generated_obs = deNormalize(generated_obs)

            action_to_step = action[0]
            kwargs_step = {k: v for (k, v) in [("generated_observation", generated_obs),
                                               ("action_proba", action_proba),
                                               ("action_grid_walker", action_walker)] if v is not None}

            obs, _, done, _ = env.step(action_to_step, **kwargs_step)

            frames += 1
            t += 1
            if done:
                if np.random.rand() < args.toward_target_timesteps_proportion and using_real_omnibot:
                    episode_toward_target_on = True
                else:
                    episode_toward_target_on = False
                print("Episode finished after {} timesteps".format(t + 1))

        if thread_num == 0:
            print("{:.2f} FPS".format(frames * args.num_cpu / (time.time() - start_time)))



def main():
    parser = argparse.ArgumentParser(description='Deteministic dataset generator for SRL training ' +
                                                 '(can be used for environment testing)')
    parser.add_argument('--num-cpu', type=int, default=1, help='number of cpu to run on')
    parser.add_argument('--num-episode', type=int, default=50, help='number of episode to run')
    parser.add_argument('--save-path', type=str, default='srl_zoo/data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--name', type=str, default='kuka_button', help='Folder name for the output')
    parser.add_argument('--env', type=str, default='KukaButtonGymEnv-v0', help='The environment wanted',
                        choices=list(registered_env.keys()))
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no-record-data', action='store_true', default=False)
    parser.add_argument('--max-distance', type=float, default=0.28,
                        help='Beyond this distance from the goal, the agent gets a negative reward')
    parser.add_argument('-c', '--continuous-actions', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force the save, even if it overrides something else,' +
                             ' including partial parts if they exist')
    parser.add_argument('-r', '--random-target', action='store_true', default=False,
                        help='Set the button to a random position')
    parser.add_argument('--multi-view', action='store_true', default=False, help='Set a second camera to the scene')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('--reward-dist', action='store_true', default=False,
                        help='Prints out the reward distribution when the dataset generation is finished')
    parser.add_argument('--run-policy', type=str, default="random",
                        choices=VALID_POLICIES,
                        help='Policy to run for data collection ' +
                             '(random, localy pretrained ppo2, pretrained custom policy)')
    parser.add_argument('--log-custom-policy', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--latest', action='store_true', default=False,
                        help='load the latest learned model (location: args.log-custom-policy)')
    parser.add_argument('-rgm', '--replay-generative-model', type=str, default="", choices=['vae'],
                        help='Generative model to replay for generating a dataset (for Continual Learning purposes)')
    parser.add_argument('--log-generative-model', type=str, default='',
                        help='Logs of the custom pretained policy to run for data collection')
    parser.add_argument('--ppo2-timesteps', type=int, default=1000,
                        help='number of timesteps to run PPO2 on before generating the dataset')
    parser.add_argument('--toward-target-timesteps-proportion', type=float, default=0.0,
                        help="propotion of timesteps that use simply towards target policy, should be 0.0 to 1.0")
    parser.add_argument('-sc', '--simple-continual', action='store_true', default=False,
                        help='Simple red square target for task 1 of continual learning scenario. ' +
                             'The task is: robot should reach the target.')
    parser.add_argument('-cc', '--circular-continual', action='store_true', default=False,
                        help='Blue square target for task 2 of continual learning scenario. ' +
                             'The task is: robot should turn in circle around the target.')
    parser.add_argument('-sqc', '--square-continual', action='store_true', default=False,
                        help='Green square target for task 3 of continual learning scenario. ' +
                             'The task is: robot should turn in square around the target.')
    parser.add_argument('-chc', '--chasing-continual', action='store_true', default=False,
                        help='Two chasing robots in the  same domain of environment' +
                             'The task is: one robot should keep a certain distance towars the other.')
    parser.add_argument('-esc', '--escape-continual', action='store_true', default=False,
                        help='Two chasing robots in the  same domain of environment' +
                             'The task is: the trainable agent tries to escape from the "zombie" robot.')
    parser.add_argument('--short-episodes', action='store_true', default=False,
                        help='Generate short episodes (only 10 contacts with the target allowed).')
    parser.add_argument('--episode', type=int, default=-1,
                        help='Model saved at episode N that we want to load')

    args = parser.parse_args()

    assert (args.num_cpu > 0), "Error: number of cpu must be positive and non zero"
    assert (args.max_distance > 0), "Error: max distance must be positive and non zero"
    assert (args.num_episode > 0), "Error: number of episodes must be positive and non zero"
    assert not args.reward_dist or not args.shape_reward, \
        "Error: cannot display the reward distribution for continuous reward"
    assert not(registered_env[args.env][3] is ThreadingType.NONE and args.num_cpu != 1), \
        "Error: cannot have more than 1 CPU for the environment {}".format(args.env)
    if args.num_cpu > args.num_episode:
        args.num_cpu = args.num_episode
        printYellow("num_cpu cannot be greater than num_episode, defaulting to {} cpus.".format(args.num_cpu))

    assert sum([args.simple_continual, args.circular_continual, args.square_continual]) <= 1, \
        "For continual SRL and RL, please provide only one scenario at the time !"

    assert not (args.log_custom_policy == '' and args.run_policy in ['walker', 'custom']), \
        "If using a custom policy, please specify a valid log folder for loading it."

    assert not (args.log_generative_model == '' and args.replay_generative_model == 'custom'), \
        "If using a custom policy, please specify a valid log folder for loading it."

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # this is done so seed 0 and 1 are different and not simply offset of the same datasets.
    args.seed = np.random.RandomState(args.seed).randint(int(1e10))

    # File exists, need to deal with it
    if not args.no_record_data and os.path.exists(args.save_path + args.name):
        assert args.force, "Error: save directory '{}' already exists".format(args.save_path + args.name)

        shutil.rmtree(args.save_path + args.name)
        for part in glob.glob(args.save_path + args.name + "_part-[0-9]*"):
            shutil.rmtree(part)
    if not args.no_record_data:
        # create the output
        os.mkdir(args.save_path + args.name)

    if args.num_cpu == 1:
        env_thread(args, 0, partition=False)
    else:
        # try and divide into multiple processes, with an environment each
        try:
            jobs = []
            for i in range(args.num_cpu):
                process = multiprocessing.Process(target=env_thread, args=(args, i, True))
                jobs.append(process)

            for j in jobs:
                j.start()

            try:
                for j in jobs:
                    j.join()
            except Exception as e:
                printRed("Error: unable to join thread")
                raise e

        except Exception as e:
            printRed("Error: unable to start thread")
            raise e

    if not args.no_record_data and args.num_cpu > 1:
        # sleep 1 second, to avoid congruency issues from multiprocess (eg., files still writing)
        time.sleep(1)
        # get all the parts
        file_parts = sorted(glob.glob(args.save_path + args.name + "_part-[0-9]*"), key=lambda a: int(a.split("-")[-1]))

        # move the config files from any as they are identical
        os.rename(file_parts[0] + "/dataset_config.json", args.save_path + args.name + "/dataset_config.json")
        os.rename(file_parts[0] + "/env_globals.json", args.save_path + args.name + "/env_globals.json")

        ground_truth = None
        preprocessed_data = None

        # used to convert the part record_id to the fused record_id
        record_id = 0
        for part in file_parts:
            # sort the record names alphabetically, then numerically
            records = sorted(glob.glob(part + "/record_[0-9]*"), key=lambda a: int(a.split("_")[-1]))

            record_id_start = record_id
            for record in records:
                os.renames(record, args.save_path + args.name + "/record_{:03d}".format(record_id))
                record_id += 1

            # fuse the npz files together, in the right order
            if ground_truth is None:
                # init
                ground_truth = {}
                preprocessed_data = {}
                ground_truth_load = np.load(part + "/ground_truth.npz")
                preprocessed_data_load = np.load(part + "/preprocessed_data.npz")

                for arr in ground_truth_load.files:
                    if arr == "images_path":
                        ground_truth[arr] = np.array(
                            [convertImagePath(args, path, record_id_start) for path in ground_truth_load[arr]])
                    else:
                        ground_truth[arr] = ground_truth_load[arr]
                for arr in preprocessed_data_load.files:
                    preprocessed_data[arr] = preprocessed_data_load[arr]

            else:
                ground_truth_load = np.load(part + "/ground_truth.npz")
                preprocessed_data_load = np.load(part + "/preprocessed_data.npz")

                for arr in ground_truth_load.files:
                    if arr == "images_path":
                        sanitised_paths = np.array(
                            [convertImagePath(args, path, record_id_start) for path in ground_truth_load[arr]])
                        ground_truth[arr] = np.concatenate((ground_truth[arr], sanitised_paths))
                    else:
                        ground_truth[arr] = np.concatenate((ground_truth[arr], ground_truth_load[arr]))
                for arr in preprocessed_data_load.files:
                    preprocessed_data[arr] = np.concatenate((preprocessed_data[arr], preprocessed_data_load[arr]))

            # remove the current part folder
            shutil.rmtree(part)

        # save the fused outputs
        np.savez(args.save_path + args.name + "/ground_truth.npz", **ground_truth)
        np.savez(args.save_path + args.name + "/preprocessed_data.npz", **preprocessed_data)

    if args.reward_dist:
        rewards, counts = np.unique(np.load(args.save_path + args.name + "/preprocessed_data.npz")['rewards'],
                                    return_counts=True)
        counts = ["{:.2f}%".format(val * 100) for val in counts / np.sum(counts)]
        print("reward distribution:")
        [print(" ", reward, count) for reward, count in list(zip(rewards, counts))]


if __name__ == '__main__':
    main()
