"""
Script to verify the states distribution
"""
import json
import os
import argparse


import numpy as np
import torch as th
from ipdb import set_trace as tt

from state_representation.models import loadSRLModel, getSRLDim
from srl_zoo.utils import loadData
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader, DataLoader


#os.chdir('/home/tete/Robotics-branches/robotics-rl-srl-two/logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')



def dataDistribution(data_foler, srl_model_path=None):
    state_dim = getSRLDim(srl_model_path)
    srl_model = loadSRLModel(srl_model_path,th.cuda.is_available(), state_dim, env_object=None)

    #load images and other data
    training_data, ground_truth, true_states, _ = loadData(data_foler, absolute_path=True)
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
    images_path = ground_truth['images_path']
    actions = training_data['actions']
    actions_proba = training_data['actions_proba']

    # we change the path to the local path at the toolbox level
    images_path_copy = ["srl_zoo/data/" + images_path[k] for k in range(images_path.shape[0])]
    images_path = np.array(images_path_copy)

    tt()




    return

def loadKwargs(log_dir):
    with open(os.path.join(args.log_dir, 'args.json')) as data:
        rl_kwargs = json.load(data)
    with open(os.path.join(args.log_dir, 'env_globals.json')) as data:
        env_kwargs = json.load(data)
    return rl_kwargs, env_kwargs

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Train script for RL algorithms")
    parser.add_argument('--log-dir', type=str, default='logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')
    parser.add_argument('--data-path', type=str, default='data/test_dataset/')
    args, unknown = parser.parse_known_args()

    #rl_kwargs, env_kwargs = loadKwargs(args.log_dir)
    #srl_model_path = env_kwargs['srl_model_path']
    srl_model_path = 'srl_zoo/logs/test_dataset/19-06-26_23h44_20_custom_cnn_ST_DIM200_inverse_autoencoder/srl_model.pth'

    print('Loading data for separation ')
    dataDistribution(data_folder=args.data_path, srl_model_path=srl_model_path)
    print("OK")
