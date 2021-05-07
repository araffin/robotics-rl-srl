import argparse
import datetime
import glob
import os
import subprocess
import time

import json
import numpy as np

from environments.registry import registered_env
from rl_baselines.evaluation.cross_eval_utils import loadConfigAndSetup, latestPolicy
from srl_zoo.utils import printRed, printYellow, printBlue
from state_representation.registry import registered_srl

CONTINUAL_LEARNING_LABELS = ['CC', 'SC', 'EC', 'SQC', 'ESC']
CL_LABEL_KEY = "continual_learning_label"

def OnPolicyDatasetGenerator(teacher_path, output_name, task_id, episode=-1, env_name='OmnirobotEnv-v0', num_cpu=1,
                             num_eps=200, test_mode=False):
    """

    :param teacher_path:
    :param output_name:
    :param task_id:
    :param episode:
    :param env_name:
    :param num_cpu:
    :param num_eps:
    :return:
    """
    assert task_id in CONTINUAL_LEARNING_LABELS
    command_line = ['python', '-m', 'environments.dataset_generator', '--run-policy', 'custom']
    cpu_command = ['--num-cpu', str(num_cpu)]
    name_command = ['--name', output_name]
    save_path = ['--save-path', "data/"]
    env_command = ['--env', env_name]
    task_command = ['-' + task_id.lower()]
    if task_id == 'SC':
        episode_command = ['--num-episode', str(10 if test_mode else 400)]
    else:
        episode_command = ['--num-episode', str(10 if test_mode else 60)]

    policy_command = ['--log-custom-policy', teacher_path]
    if episode == -1:
        eps_policy = []
    else:
        eps_policy = ['--episode', str(episode)]

    command = command_line + cpu_command + policy_command + name_command + env_command + task_command + \
        episode_command + eps_policy + save_path + ['-f'] + ['--seed', str(2)]

    if task_id == 'SC':
        command += ['--short-episodes']

    ok = subprocess.call(command)
    assert ok == 0, "Teacher dataset for task " + task_id + " was not generated !"


def allPolicy(log_dir):
    """

    :param log_dir:
    :return:
    """
    train_args, algo_name, algo_class, srl_model_path, env_kwargs = loadConfigAndSetup(log_dir)
    files = glob.glob(os.path.join(log_dir + algo_name + '_*_model.pkl'))
    files_list = []
    for file in files:
        eps = int((file.split('_')[-2]))
        files_list.append((eps, file))

    def sortFirst(val):
        """

        :param val:
        :return:
        """
        return val[0]

    files_list.sort(key=sortFirst)
    res = np.array(files_list)
    print(res)
    return res[:, 0], res[:, 1]


def allPolicyFiles(log_dir):
    """

    :param log_dir:
    :return:
    """
    printYellow(log_dir)
    files = glob.glob(log_dir + '/model_*')

    files_list = []
    for file in files:
        eps = int((file.split('_')[-1]))
        files_list.append((eps, file+'/'))

    def sortFirst(val):
        """

        :param val:
        :return:
        """
        return val[0]

    files_list.sort(key=sortFirst)
    res = np.array(files_list)
    return res[:, 0], res[:, 1]


def newPolicy(episodes, file_path):
    """

    :param episodes:
    :param file_path:
    :return:
    """
    train_args, algo_name, algo_class, srl_model_path, env_kwargs = loadConfigAndSetup(file_path)
    episode, model_path, OK = latestPolicy(file_path, algo_name)
    if episode in episodes:
        return -1, '', False
    else:
        return episode, model_path, True


def trainStudent(teacher_data_path, task_id, yaml_file='config/srl_models.yaml', log_dir='logs/',
                 srl_model='srl_combination', env_name='OmnirobotEnv-v0', training_size=40000, epochs=20):
    """

    :param teacher_data_path:
    :param task_id: Environment ID
    :param yaml_file:
    :param log_dir:
    :param srl_model:
    :param env_name:
    :param training_size:
    :param epochs:
    :return:
    """
    command_line = ['python', '-m', 'rl_baselines.train', '--latest', '--algo', 'distillation', '--log-dir', log_dir]
    srl_command = ['--srl-model', srl_model]
    env_command = ['--env', env_name]
    policy_command = ['--teacher-data-folder', teacher_data_path]
    size_epochs = ['--distillation-training-set-size', str(training_size), '--epochs-distillation', str(epochs)]
    task_command = ['-' + task_id.lower()]
    ok = subprocess.call(command_line + srl_command
                         + env_command + policy_command + size_epochs + task_command + ['--srl-config-file', yaml_file])
    assert ok == 0


def mergeData(teacher_dataset_1, teacher_dataset_2, merge_dataset, force=False):
    """

    :param teacher_dataset_1:
    :param teacher_dataset_2:
    :param merge_dataset:
    :return:
    """
    merge_command = ['--merge', teacher_dataset_1, teacher_dataset_2, merge_dataset]
    if force:
        merge_command.append('-f')
    # -rm is to remove the original dataset after training, this would be of the same effect as the old version
    # which will remove the dataset automatically.
    ok = subprocess.call(['python', '-m', 'environments.dataset_merger', '-rm'] + merge_command)
    assert ok == 0


def main():
    # Global variables for callback
    parser = argparse.ArgumentParser(description="Evaluation script for distillation from two teacher policies")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--env', type=str, help='environment ID', default='OmnirobotEnv-v0',
                        choices=list(registered_env.keys()))
    parser.add_argument('--episode_window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--log-dir-teacher-one', default='/tmp/gym/', type=str,
                        help='directory to load an optmimal agent for task 1 (default: /tmp/gym)')
    parser.add_argument('--log-dir-teacher-two', default='/tmp/gym/', type=str,
                        help='directory to load an optmimal agent for task 2 (default: /tmp/gym)')
    parser.add_argument('--log-dir-student', default='/tmp/gym/', type=str,
                        help='directory to save the student agent logs and model (default: /tmp/gym)')
    parser.add_argument('--srl-config-file-one', type=str, default="config/srl_models_one.yaml",
                        help='Set the location of the SRL model path configuration.')
    parser.add_argument('--srl-config-file-two', type=str, default="config/srl_models_two.yaml",
                        help='Set the location of the SRL model path configuration.')
    parser.add_argument('--epochs-distillation', type=int, default=30, metavar='N',
                        help='number of epochs to train for distillation(default: 30)')
    parser.add_argument('--distillation-training-set-size', type=int, default=-1,
                        help='Limit size (number of samples) of the training set (default: -1)')
    parser.add_argument('--eval-tasks', type=str, nargs='+', default=['cc', 'sqc', 'sc'],
                        help='A cross evaluation from the latest stored model to all tasks')
    parser.add_argument('--continual-learning-labels', type=str, nargs=2, metavar=('label_1', 'label_2'),
                        default=argparse.SUPPRESS,
                        help='Labels for the continual learning RL distillation task.')
    parser.add_argument('--student-srl-model', type=str, default='raw_pixels', choices=list(registered_srl.keys()),
                        help='SRL model to use for the student RL policy')
    parser.add_argument('--epochs-teacher-datasets', type=int, default=30, metavar='N',
                        help='number of epochs for generating both RL teacher datasets (default: 30)')
    parser.add_argument('--num-iteration', type=int, default=1,
                        help='number of time each algorithm should be run the eval (N seeds).')
    parser.add_argument('--eval-episode-window', type=int, default=400, metavar='N',
                        help='Episode window for saving each policy checkpoint for future distillation(default: 100)')

    args, unknown = parser.parse_known_args()

    if 'continual_learning_labels' in args:
        assert args.continual_learning_labels[0] in CONTINUAL_LEARNING_LABELS and args.continual_learning_labels[1] \
               in CONTINUAL_LEARNING_LABELS, "Please specify a valid Continual learning label to each dataset to be " \
                                             "used for RL distillation !"
    print(args.continual_learning_labels)
    assert os.path.exists(args.srl_config_file_one), \
        "Error: cannot load \"--srl-config-file {}\", file not found!".format(args.srl_config_file_one)

    assert os.path.exists(args.srl_config_file_two), \
        "Error: cannot load \"--srl-config-file {}\", file not found!".format(args.srl_config_file_two)
    if not (args.log_dir_teacher_one == "None"):
        assert os.path.exists(args.log_dir_teacher_one), \
            "Error: cannot load \"--srl-config-file {}\", file not found!".format(args.log_dir_teacher_one)
    assert os.path.exists(args.log_dir_teacher_two), \
        "Error: cannot load \"--srl-config-file {}\", file not found!".format(args.srl_config_file_two)

    teacher_pro = args.log_dir_teacher_one
    teacher_learn = args.log_dir_teacher_two

    # The output path generate from the
    teacher_pro_data = args.continual_learning_labels[0] + '/'
    teacher_learn_data = args.continual_learning_labels[1] + '/'
    merge_path = "data/on_policy_merged"

    print(teacher_pro_data, teacher_learn_data)
    episodes, policy_path = allPolicy(teacher_learn)

    rewards_at_episode = {}

    if args.continual_learning_labels[1] == "CC":
        episodes_to_test = [e for e in episodes if (int(e) < 2000 and int(e) % 200 == 0) or
                            (int(e) > 2000 and int(e) % 1000 == 0)]
    else:
        episodes_to_test = [e for e in episodes if (int(e) <= 5000 and int(e) % 1000 == 0) or
                            (int(e) > 5000 and int(e) % 10000 == 0)]

    if not (args.log_dir_teacher_one == "None"):
        # generate data from Professional teacher
        printYellow("\nGenerating on policy for optimal teacher: " + args.continual_learning_labels[0])

        OnPolicyDatasetGenerator(teacher_pro, args.continual_learning_labels[0] + '_copy/',
                                 task_id=args.continual_learning_labels[0], num_eps=args.epochs_teacher_datasets,
                                 episode=-1, env_name=args.env)
    print("Eval on eps list: ", episodes_to_test)
    for eps in episodes_to_test:
        student_path = args.log_dir_student
        printBlue("\n\nEvaluation at episode " + str(eps))

        if not (args.log_dir_teacher_one == "None"):
            # Use a copy of the optimal teacher
            ok = subprocess.call(
                ['cp', '-r', 'data/' + args.continual_learning_labels[0] + '_copy/', 'data/' + teacher_pro_data, '-f'])
            assert ok == 0
            time.sleep(2)

        # Generate data from learning teacher
        printYellow("\nGenerating on-policy data from the optimal teacher: " + args.continual_learning_labels[1])
        OnPolicyDatasetGenerator(teacher_learn, teacher_learn_data, task_id=args.continual_learning_labels[1],
                                 num_eps=args.epochs_teacher_datasets, episode=eps, env_name=args.env)

        # If Performing policy distillation from a single (learning) teacher at multiple checkpoints
        if args.log_dir_teacher_one == "None":
            merge_path = 'data/' + teacher_learn_data
            ok = subprocess.call(
                ['cp', '-r', merge_path, 'srl_zoo/data/', '-f'])
        else:
            # merge the data
            mergeData('data/' + teacher_pro_data, 'data/' + teacher_learn_data, merge_path, force=True)

            ok = subprocess.call(
                ['cp', '-r', 'data/on_policy_merged/', 'srl_zoo/data/', '-f'])
        assert ok == 0
        time.sleep(2)

        # Train a policy with distillation on the merged teacher's datasets
        trainStudent('srl_zoo/' + merge_path, args.continual_learning_labels[1], yaml_file=args.srl_config_file_one,
                     log_dir=args.log_dir_student,
                     srl_model=args.student_srl_model, env_name=args.env,
                     training_size=args.distillation_training_set_size, epochs=args.epochs_distillation)
        student_path += args.env + '/' + args.student_srl_model + "/distillation/"
        latest_student_path = max([student_path + "/" + d for d in os.listdir(student_path)
                                   if os.path.isdir(student_path + "/" + d)], key=os.path.getmtime) + '/'
        rewards = {}
        printRed("\nSaving the student at path: " + latest_student_path)
        for task_label in ["-sc", "-cc"]:
            rewards[task_label] = []

            for seed_i in range(args.num_iteration):
                printYellow("\nEvaluating student on task: " + task_label + " for seed: " + str(seed_i))
                command_line_enjoy_student = ['python', '-m', 'replay.enjoy_baselines', '--num-timesteps', '251',
                                              '--log-dir', latest_student_path, task_label, "--seed", str(seed_i)]
                ok = subprocess.check_output(command_line_enjoy_student)
                ok = ok.decode('utf-8')
                str_before = "Mean reward: "
                str_after = "\npybullet"
                idx_before = ok.find(str_before) + len(str_before)
                idx_after = ok.find(str_after)
                seed_reward = float(ok[idx_before: idx_after])
                rewards[task_label].append(seed_reward)
        print("rewards at eps ", eps, ": ", rewards)
        rewards_at_episode[eps] = rewards
    print("All rewards: ", rewards_at_episode)
    json_dict = json.dumps(rewards_at_episode)
    json_dict_name = \
        args.log_dir_student + "/reward_at_episode_" + datetime.datetime.now().strftime("%y-%m-%d_%Hh%M_%S") + '.json'
    f = open(json_dict_name, "w")
    f.write(json_dict)
    f.close()
    printRed("\nSaving the evalation at path: " + json_dict_name)


if __name__ == '__main__':
    main()
