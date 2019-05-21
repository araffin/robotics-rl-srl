import subprocess
import pytest
import os
import shutil
import time

from rl_baselines.student_eval import OnPolicyDatasetGenerator, mergeData, trainStudent


ENV_NAME = 'OmnirobotEnv-v0'
PATH_SRL = "srl_zoo/data/"
DEFAULT_SRL_TEACHERS = "ground_truth"
DEFAULT_SRL_STUDENT = "raw_pixels"
NUM_TIMESTEP = 25000
NUM_CPU = 4


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


@pytest.mark.fast
def testOnPolicyDatasetGeneration():

    # # Train Ground_truth teacher policies for each env

    # do not write distillation in path to prevent loading irrelevant algo based on folder name
    test_log_dir = "logs/test_dist/"
    test_log_dir_teacher_one = test_log_dir + 'teacher_one/'
    test_log_dir_teacher_two = test_log_dir + 'teacher_two/'

    if os.path.exists(test_log_dir):
        print("Destination log directory '{}' already exists - removing it before re-creating it".format(test_log_dir))
        shutil.rmtree(test_log_dir)

    os.mkdir(test_log_dir)
    os.mkdir(test_log_dir_teacher_one)
    os.mkdir(test_log_dir_teacher_two)

    args = ['--algo', "ppo2", '--env', ENV_NAME, '--srl-model', DEFAULT_SRL_TEACHERS,
            '--num-timesteps', NUM_TIMESTEP, '--num-cpu', NUM_CPU, '--no-vis', '--log-dir',
            test_log_dir_teacher_one, '-sc']
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + args)
    assertEq(ok, 0)

    args = ['--algo', "ppo2", '--env', ENV_NAME, '--srl-model', DEFAULT_SRL_TEACHERS,
            '--num-timesteps', NUM_TIMESTEP, '--num-cpu', NUM_CPU, '--no-vis', '--log-dir',
            test_log_dir_teacher_two, '-cc']
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + args)
    assertEq(ok, 0)

    # Generate on-policy datasets from each teacher
    test_log_dir_teacher_one += ENV_NAME + '/' + DEFAULT_SRL_TEACHERS + "/ppo2"
    teacher_one_path = \
        max([test_log_dir_teacher_one + "/" + d for d in os.listdir(test_log_dir_teacher_one)
             if os.path.isdir(test_log_dir_teacher_one + "/" + d)], key=os.path.getmtime) + '/'

    OnPolicyDatasetGenerator(teacher_path=teacher_one_path,
                             output_name='test_SC_copy/', task_id='SC', episode=-1, env_name=ENV_NAME, test_mode=True)

    test_log_dir_teacher_two += ENV_NAME + '/' + DEFAULT_SRL_TEACHERS + "/ppo2"
    teacher_two_path = \
        max([test_log_dir_teacher_two + "/" + d for d in os.listdir(test_log_dir_teacher_two)
             if os.path.isdir(test_log_dir_teacher_two + "/" + d)], key=os.path.getmtime) + '/'

    OnPolicyDatasetGenerator(teacher_path=teacher_two_path, output_name='test_CC_copy/',
                             task_id='CC', episode=-1, env_name=ENV_NAME, test_mode=True)

    # Merge those on-policy datasets
    merge_path = "data/on_policy_merged_test"
    mergeData('data/test_SC_copy/', 'data/test_CC_copy/', merge_path, force=True)

    ok = subprocess.call(['cp', '-r', merge_path, 'srl_zoo/data/', '-f'])
    assert ok == 0
    time.sleep(10)

    # Train a raw_pixels student policy via distillation
    trainStudent(merge_path, "CC", log_dir=test_log_dir, srl_model=DEFAULT_SRL_STUDENT,
                 env_name=ENV_NAME, training_size=500, epochs=3)
    print("Distillation test performed!")
