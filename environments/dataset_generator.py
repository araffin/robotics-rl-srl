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

from environments import ThreadingType
from environments.registry import registered_env
from replay.enjoy_baselines import createEnv, loadConfigAndSetup
from rl_baselines.utils import WrapFrameStack
from srl_zoo.utils import printRed, printYellow


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


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


def env_thread(args, thread_num, partition=True, use_ppo2=False):
    """
    Run a session of an environment
    :param args: (ArgumentParser object)
    :param thread_num: (int) The thread ID of the environment session
    :param partition: (bool) If the output should be in multiple parts (default=True)
    :param use_ppo2: (bool) Use ppo2 to generate the dataset
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
        "square_continual_move": args.square_continual

    }

    if partition:
        env_kwargs["name"] = args.name + "_part-" + str(thread_num)
    else:
        env_kwargs["name"] = args.name

    if args.run_policy == "custom":
        args.log_dir = args.log_custom_policy
        args.render = args.display
        args.plotting, args.action_proba = False, False

        train_args, load_path, algo_name, algo_class, _, env_kwargs_extra = loadConfigAndSetup(args)
        env_kwargs["srl_model"] = env_kwargs_extra["srl_model"]

    env_class = registered_env[args.env][0]
    env = env_class(**env_kwargs)
    model = None
    if use_ppo2 or args.run_policy == 'custom':

        # Additional env when using a trained agent to generate data
        train_env = vecEnv(env_kwargs, env_class)

        if use_ppo2:
            model = PPO2(CnnPolicy, train_env).learn(args.ppo2_timesteps)
        else:
            _, _, algo_args = createEnv(args, train_args, algo_name, algo_class, env_kwargs)

            tf.reset_default_graph()
            set_global_seeds(args.seed)
            printYellow("Compiling Policy function....")
            model = algo_class.load(load_path, args=algo_args)

    frames = 0
    start_time = time.time()
    # divide evenly, then do an extra one for only some of them in order to get the right count
    for i_episode in range(args.num_episode // args.num_cpu + 1 * (args.num_episode % args.num_cpu > thread_num)):
        # seed + position in this slice + size of slice (with reminder if uneven partitions)
        seed = args.seed + i_episode + args.num_episode // args.num_cpu * thread_num + \
               (thread_num if thread_num <= args.num_episode % args.num_cpu else args.num_episode % args.num_cpu)

        if not args.run_policy == 'custom':
            env.seed(seed)
            env.action_space.seed(seed)  # this is for the sample() function from gym.space

        obs = env.reset()
        done = False
        t = 0
        episode_toward_target_on = False
        while not done:
            env.render()

            if use_ppo2:
                action, _ = model.predict([obs])
            elif args.run_policy == 'custom':
                action = [model.getAction(obs, done)]
            else:
                if episode_toward_target_on and np.random.rand() < args.toward_target_timesteps_proportion:
                    action = [env.actionPolicyTowardTarget()]
                else:
                    action = [env.action_space.sample()]

            action_to_step = action[0]
            new_obs, _, done, _ = env.step(action_to_step)

            if args.run_policy == 'custom':
                obs = new_obs

            frames += 1
            t += 1
            if done:

                if np.random.rand() < args.toward_target_timesteps_proportion:
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
                        choices=['random', 'ppo2', 'custom'],
                        help='Policy to run for data collection ' +
                             '(random, localy pretrained ppo2, pretrained custom policy)')
    parser.add_argument('--log-custom-policy', type=str, default='',
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

    assert len(args.log_custom_policy) >= 0 and args.run_policy == "custom", \
        "If using a custom policy, please specify a valid log folder for loading it."

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
        env_thread(args, 0, partition=False, use_ppo2=args.run_policy=="ppo2")
    else:
        # try and divide into multiple processes, with an environment each
        try:
            jobs = []
            for i in range(args.num_cpu):
                process = multiprocessing.Process(target=env_thread, args=(args, i, True, args.run_policy=="ppo2"))
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
