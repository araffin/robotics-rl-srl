from __future__ import division, absolute_import, print_function

import argparse
import glob
import multiprocessing
import os
import shutil
import time

import numpy as np

import environments.kuka_gym.kuka_2button_gym_env as kuka_env_2
import environments.kuka_gym.kuka_button_gym_env as kuka_env
import environments.kuka_gym.kuka_rand_button_gym_env as kuka_env_rand
import environments.kuka_gym.kuka_moving_button_gym_env as kuka_env_moving
import environments.mobile_robot.mobile_robot_env as mobile_robot
import environments.mobile_robot.mobile_robot_2target_env as mobile_robot_2target
import environments.mobile_robot.mobile_robot_1D_env as mobile_robot_1D
from srl_zoo.utils import printRed


def convertImagePath(args, path, record_id_start):
    """
    Used to convert an image path, from one location, to an other
    :param args: (ArgumentParser object)
    :param path: (str)
    :param record_id_start: (int) where does the current part start counting its records
    :return:
    """
    image_name = path.split("/")[-1]
    # get record id for output, by adding the current offset with the record_id
    # of the folder
    new_record_id = record_id_start + int(path.split("/")[-2].split("_")[-1])
    return args.save_name + "/record_{:03d}".format(new_record_id) + "/" + image_name


def env_thread(args, thread_num, partition=True):
    """
    Run a session of an environment
    :param args: (ArgumentParser object)
    :param thread_num: (int) The thread ID of the environment session
    :param partition: (bool) If the output should be in multiple parts (default=True)
    """
    env_kwargs = {
        "max_distance": args.max_distance,
        "random_target": args.relative,
        "force_down": True,
        "is_discrete": not args.continuous_actions,
        "renders": thread_num == 0 and not args.no_display,
        "record_data": args.record_data,
        "multi_view": args.multi_view,
        "save_path": args.save_folder,
        "shape_reward": args.shape_reward
    }

    if args.env == "Kuka2ButtonGymEnv":
        env_kwargs["force_down"] = False

    env_class = {"KukaButtonGymEnv-v0": kuka_env.KukaButtonGymEnv,
                 "Kuka2ButtonGymEnv-v0": kuka_env_2.Kuka2ButtonGymEnv,
                 "KukaRandButtonGymEnv-v0": kuka_env_rand.KukaRandButtonGymEnv,
                 "KukaMovingButtonGymEnv-v0": kuka_env_moving.KukaMovingButtonGymEnv,
                 "MobileRobotGymEnv-v0": mobile_robot.MobileRobotGymEnv,
                 "MobileRobot2TargetGymEnv-v0": mobile_robot_2target.MobileRobot2TargetGymEnv,
                 "MobileRobot1DGymEnv-v0": mobile_robot_1D.MobileRobot1DGymEnv
                 }[args.env]

    if partition:
        env_kwargs["name"] = args.save_name + "_part-" + str(thread_num)
    else:
        env_kwargs["name"] = args.save_name

    env = env_class(**env_kwargs)
    env.seed(args.seed + thread_num)

    frames = 0
    start_time = time.time()
    # divide evenly, then do an extra one for only some of them in order to get the right count
    for i_episode in range(args.num_episode // args.num_cpu + 1 * (args.num_episode % args.num_cpu > thread_num)):
        env.reset()
        done = False
        t = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            frames += 1
            t += 1
            if done:
                print("Episode finished after {} timesteps".format(t + 1))

        if thread_num == 0:
            print("{:.2f} FPS".format(frames * args.num_cpu / (time.time() - start_time)))


def main():
    parser = argparse.ArgumentParser(description='Environment tester (can be used to record datasets for SRL training)')
    parser.add_argument('--num-cpu', type=int, default=1, help='number of cpu to run on')
    parser.add_argument('--num-episode', type=int, default=50, help='number of episode to run')
    parser.add_argument('--save-folder', type=str, default='srl_zoo/data/',
                        help='Folder where the environments will save the output')
    parser.add_argument('--save-name', type=str, default='kuka_button', help='Folder name for the output')
    parser.add_argument('--env', type=str, default='KukaButtonGymEnv-v0', help='The environment wanted',
                        choices=["KukaButtonGymEnv-v0", "KukaRandButtonGymEnv-v0", "Kuka2ButtonGymEnv-v0",
                                 "KukaMovingButtonGymEnv-v0", "MobileRobotGymEnv-v0", "MobileRobot2TargetGymEnv-v0",
                                 "MobileRobot1DGymEnv-v0"])
    parser.add_argument('--no-display', action='store_true', default=False)
    parser.add_argument('--record-data', action='store_true', default=False)
    parser.add_argument('--max-distance', type=float, default=0.28,
                        help='Beyond this distance from the goal, the agent gets a negative reward')
    parser.add_argument('-c', '--continuous-actions', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='the seed')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force the save, even if it overrides something else (including partial parts if they exist)')
    parser.add_argument('-r', '--relative', action='store_true', default=False,
                        help='Set the button to a random position')
    parser.add_argument('--multi-view', action='store_true', default=False, help='Set a second camera to the scene')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    args = parser.parse_args()

    assert (args.num_cpu > 0), "Error: number of cpu must be positive and non zero"
    assert (args.max_distance > 0), "Error: max distance must be positive and non zero"
    assert (args.num_episode > 0), "Error: number of episodes must be positive and non zero"

    # File exists, need to deal with it
    if args.record_data and os.path.exists(args.save_folder + args.save_name):
        assert args.force, "Error: save directory '{}' already exists".format(args.save_folder + args.save_name)

        shutil.rmtree(args.save_folder + args.save_name)
        for part in glob.glob(args.save_folder + args.save_name + "_part-[0-9]*"):
            shutil.rmtree(part)
    elif args.record_data:
        # create the output
        os.mkdir(args.save_folder + args.save_name)

    if args.num_cpu == 1:
        env_thread(args, 0, partition=False)
    else:
        # try and divide into multiple processes, with an environment each
        try:
            jobs = []
            for i in range(args.num_cpu):
                process = multiprocessing.Process(target=env_thread, args=(args, i))
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

    if args.record_data and args.num_cpu > 1:

        # get all the parts
        file_parts = glob.glob(args.save_folder + args.save_name + "_part-[0-9]*")

        # move the config files from any as they are identical
        os.rename(file_parts[0] + "/dataset_config.json",
                  args.save_folder + args.save_name + "/dataset_config.json")
        os.rename(file_parts[0] + "/env_globals.json",
                  args.save_folder + args.save_name + "/env_globals.json")

        ground_truth = None
        preprocessed_data = None

        # used to convert the part record_id to the fused record_id
        record_id = 0
        for part in file_parts:
            # sort the record names alphabetically, then numerically
            records = sorted(glob.glob(part + "/record_[0-9]*"),
                             key=lambda a: int(a.split("_")[-1]))

            record_id_start = record_id
            for record in records:
                os.renames(record, args.save_folder + args.save_name + "/record_{:03d}".format(record_id))
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
        np.savez(args.save_folder + args.save_name + "/ground_truth.npz", **ground_truth)
        np.savez(args.save_folder + args.save_name + "/preprocessed_data.npz", **preprocessed_data)


if __name__ == '__main__':
    main()
