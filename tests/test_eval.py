from __future__ import print_function, division, absolute_import

import subprocess
import shutil
import pytest
import os
from environments import ThreadingType
from environments.registry import registered_env

DEFAULT_ALGO = "ppo2"
DEFAULT_SRL = "raw_pixels"
NUM_ITERATION = 1
NUM_TIMESTEP = 251  # this should be long enough to call a reset of the environment
SEED = 0
DEFAULT_ENV = "OmnirobotEnv-v0"
DEFAULT_LOG = "logs/test_eval/"
EPOCH_DATA = 2
EPISODE_WINS =40
DIR_STUDENT = 'logs/test_students/'
EPOCH_DISTILLATION = 5
def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)

# @pytest.mark.fast
# @pytest.mark.parametrize("task", ['-sc','-cc'])
# def testCrossEval(task):
#     #Evaluation for the policy on different tasks
#     # Long enough to save one policy model
#
#     num_timesteps = 10000
#     args = ['--algo', DEFAULT_ALGO, '--srl-model', DEFAULT_SRL,
#                         '--num-timesteps', num_timesteps, '--seed', SEED, '--no-vis',
#                         '--episode-window', EPISODE_WINS,
#                         '--env', DEFAULT_ENV, '--log-dir', DEFAULT_LOG , task,
#                         '--min-episodes-save', 0]
#
#     args = list(map(str, args))
#     #We firstly train a policy to have some checkpoint to evaluate
#     ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + args)
#     assertEq(ok, 0)
#     eval_path = DEFAULT_LOG
#     for i in range(4):  # Go into the folder that contains the policy file
#         eval_path += os.listdir(eval_path)[-1] + '/'
#
#     args= ['--log-dir', eval_path, '--num-iteration', str(NUM_ITERATION)]
#     ok = subprocess.call(['python', '-m', 'rl_baselines.cross_eval'] + args)
#     assertEq(ok, 0)
#
#     #Remove test files
#     shutil.rmtree(DEFAULT_LOG)

@pytest.mark.fast
@pytest.mark.parametrize("tasks", [['-cc','-sc']])
def testStudentEval(tasks,teacher_folder_one='logs/teacher_one/', teacher_folder_two='logs/teacher_two/' ):


    teacher_args_one = ['--algo', DEFAULT_ALGO, '--srl-model', DEFAULT_SRL,
                        '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--no-vis',
                        '--episode-window', EPISODE_WINS,
                        '--env', DEFAULT_ENV, '--log-dir', teacher_folder_one , tasks[0],
                        '--min-episodes-save', 0]

    teacher_args_one = list(map(str, teacher_args_one))

    teacher_args_two = ['--algo', DEFAULT_ALGO, '--srl-model', DEFAULT_SRL,
                        '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--no-vis',
                        '--episode-window', EPISODE_WINS,
                        '--env', DEFAULT_ENV, '--log-dir', teacher_folder_two , tasks[1]]
    teacher_args_two = list(map(str, teacher_args_two))

    ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + teacher_args_one)
    assertEq(ok, 0)
    ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + teacher_args_two)
    assertEq(ok, 0)

    folder2remove = [teacher_folder_one, teacher_folder_two]

    for i in range(4):#Go into the folder that contains the policy file
        teacher_folder_two += os.listdir(teacher_folder_two)[-1] + '/'
        teacher_folder_one += os.listdir(teacher_folder_one)[-1] + '/'


    #Distillation part
    args = ['--num-iteration', NUM_ITERATION, '--epochs-teacher-datasets', EPOCH_DATA,
            '--env', DEFAULT_ENV, '--log-dir-student', DIR_STUDENT,
            '--log-dir-teacher-one', teacher_folder_one,'--log-dir-teacher-two', teacher_folder_two,
            '--epochs-distillation', EPOCH_DISTILLATION]
    if(tasks ==['-cc','-sc']):
        args+=['--srl-config-file-one', 'config/srl_models_circular.yaml',
               '--srl-config-file-two','config/srl_models_simple.yaml',
               '--continual-learning-labels', 'CC', 'SC']

    else:
        args += ['--srl-config-file-one', 'config/srl_models_simple.yaml',
                 '--srl-config-file-two', 'config/srl_models_circular.yaml',
                 '--continual-learning-labels', 'SC', 'CC']

    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'rl_baselines.student_eval'] + args)
    assertEq(ok, 0)
    for i in range(10):
        print("OK 1 ")
    #Remove test files
    shutil.rmtree(folder2remove[0])
    shutil.rmtree(folder2remove[1])
    shutil.rmtree(DIR_STUDENT)
    for i in range(10):
        print("test finished")
#
#
# @pytest.mark.slow
# @pytest.mark.parametrize("algo", ['a2c', 'acer', 'ars', 'cma-es', 'ddpg', 'deepq', 'ppo1', 'ppo2', 'random_agent',
#                                   'sac', 'trpo'])
# @pytest.mark.parametrize("tasks", [['-cc','-sc'],['-cc', '-sc']])
# def testStudentEvalAlgo(tasks,algo):
#     teacher_folder_one = 'logs/teacher_one/'
#     teacher_folder_two = 'logs/teacher_two/'
#
#     teacher_args_one = ['--algo', algo, '--srl-model', DEFAULT_SRL,
#                         '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--no-vis',
#                         '--episode-window', EPISODE_WINS,
#                         '--env', DEFAULT_ENV, '--log-dir', teacher_folder_one , tasks[0],
#                         '--min-episodes-save', 0]
#
#     teacher_args_one = list(map(str, teacher_args_one))
#
#     teacher_args_two = ['--algo', algo, '--srl-model', DEFAULT_SRL,
#                         '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--no-vis',
#                         '--episode-window', EPISODE_WINS,
#                         '--env', DEFAULT_ENV, '--log-dir', teacher_folder_two , tasks[1]]
#     teacher_args_two = list(map(str, teacher_args_two))
#
#     ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + teacher_args_one)
#     assertEq(ok, 0)
#     ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + teacher_args_two)
#     assertEq(ok, 0)
#
#     folder2remove = [teacher_folder_one, teacher_folder_one]
#
#     for i in range(4):#Go into the folder that contains the policy file
#         teacher_folder_two += os.listdir(teacher_folder_two)[-1] + '/'
#         teacher_folder_one += os.listdir(teacher_folder_one)[-1] + '/'
#
#
#     args = ['--num-iteration', NUM_ITERATION, '--epochs-teacher-datasets', EPOCH_DATA,
#             '--env', DEFAULT_ENV, '--log-dir-student', DIR_STUDENT,
#             '--log-dir-teacher-one', teacher_folder_one,'--log-dir-teacher-two', teacher_folder_two,
#             '--epochs-distillation', 5]
#     if(tasks ==['-cc','-sc']):
#         args+=['--srl-config-file-one', 'config/srl_models_circular.yaml',
#                '--srl-config-file-two','config/srl_models_simple.yaml',
#                '--continual-learning-labels', 'CC', 'SC']
#
#     else:
#         args += ['--srl-config-file-one', 'config/srl_models_simple.yaml',
#                  '--srl-config-file-two', 'config/srl_models_circular.yaml',
#                  '--continual-learning-labels', 'SC', 'CC']
#
#     args = list(map(str, args))
#     ok = subprocess.call(['python', '-m', 'rl_baselines.student_eval'] + args)
#     assertEq(ok, 0)
#
#     #Remove test files
#     shutil.rmtree(folder2remove[0])
#     shutil.rmtree(folder2remove[1])
#     shutil.rmtree(DIR_STUDENT)
