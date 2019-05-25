import subprocess
import pytest
import os
import shutil

DATA_FOLDER_NAME_1 = "kuka_test_f1"
DATA_FOLDER_NAME_2 = "kuka_test_f2"
DATA_FOLDER_NAME_3 = "kuka_test_f3"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
PATH_SRL = "srl_zoo/data/"


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


@pytest.mark.fast
def testDataGenForFusion():
    args_1 = ['--num-cpu', 4, '--num-episode', 8, '--name', DATA_FOLDER_NAME_1, '--force', '--env', DEFAULT_ENV]
    args_1 = list(map(str, args_1))

    ok = subprocess.call(['python', '-m', 'environments.dataset_generator'] + args_1)
    assertEq(ok, 0)

    args_2 = ['--num-cpu', 4, '--num-episode', 8, '--name', DATA_FOLDER_NAME_2, '--force', '--env', DEFAULT_ENV]
    args_2 = list(map(str, args_2))

    ok = subprocess.call(['python', '-m', 'environments.dataset_generator'] + args_2)
    assertEq(ok, 0)

    args_3 = ['--merge', PATH_SRL + DATA_FOLDER_NAME_1, PATH_SRL + DATA_FOLDER_NAME_2, PATH_SRL + DATA_FOLDER_NAME_3]
    args_3 = list(map(str, args_3))

    ok = subprocess.call(['python', '-m', 'environments.dataset_fusioner'] + args_3)
    assertEq(ok, 0)

    # Checking inexistance of original datasets to be merged
    assert not os.path.isdir(PATH_SRL + DATA_FOLDER_NAME_1)
    assert not os.path.isdir(PATH_SRL + DATA_FOLDER_NAME_2)
    assert os.path.isdir(PATH_SRL + DATA_FOLDER_NAME_3)

    # Removing fusionned test dataset
    shutil.rmtree(PATH_SRL + DATA_FOLDER_NAME_3)
