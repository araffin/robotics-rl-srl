"""
Script to verify the states distribution
"""
import json
import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch as th
from tqdm import tqdm
from ipdb import set_trace as tt



from state_representation.models import loadSRLModel, getSRLDim
from srl_zoo.utils import loadData
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader, DataLoader

sns.set()

#os.chdir('/home/tete/Robotics-branches/robotics-rl-srl-two/logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')

BATCH_SIZE = 32
N_WORKERS = 4
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
VALIDATION_SIZE = 0.2  # 20% of training data for validation

def PCA(data, dim=2):
    # preprocess the data
    X = th.from_numpy(data).to(DEVICE)
    X_mean = th.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = th.svd(th.t(X))
    C = th.mm(X,U[:,:dim]).to('cpu').numpy()
    return C

def dataSeparator(data_folder, srl_model_path=None):
    state_dim = getSRLDim(srl_model_path)
    srl_model = loadSRLModel(srl_model_path, th.cuda.is_available(), state_dim, env_object=None)

    #load images and other data
    print('Loading data for separation ')
    training_data, ground_truth, true_states, _ = loadData(data_folder, absolute_path=True)
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
    images_path = ground_truth['images_path']
    actions = training_data['actions']
    actions_proba = training_data['actions_proba']
    ground_turht_states_dim = true_states.shape[1]



    # we change the path to the local path at the toolbox level
    images_path_copy = ["srl_zoo/data/" + images_path[k] for k in range(images_path.shape[0])]
    images_path = np.array(images_path_copy)

    num_samples = images_path.shape[0]  # number of samples

    # indices for all time steps where the episode continues
    indices = np.array([i for i in range(num_samples-1) if not episode_starts[i + 1]], dtype='int64')

    minibatchlist = [np.array(sorted(indices[start_idx:start_idx + BATCH_SIZE]))
                     for start_idx in range(0, len(indices) - BATCH_SIZE + 1, BATCH_SIZE)]

    data_loader = DataLoader(minibatchlist, images_path, n_workers=N_WORKERS, multi_view=False,
                             use_triplets=False, is_training=True, absolute_path=True)

    srl_data = []
    #we only use the srl model to deduct the states
    srl_model.model = srl_model.model.eval()
    pbar = tqdm(total=len(data_loader))
    for minibatch_num, (minibatch_idx, obs, _, _, _) in enumerate(data_loader):
        obs = obs.to(DEVICE)
        state = srl_model.model.getStates(obs).to('cpu').detach().numpy()
        srl_data.append(state)
        pbar.update(1)
    # concatenate into one numpy array
    srl_data = np.concatenate(srl_data,axis=0)
    # PCA for the v
    pca_srl_data = PCA(srl_data, dim=ground_turht_states_dim)


    training_indices = np.concatenate(minibatchlist)
    np.random.shuffle(training_indices)

    val_num = int(len(training_indices) * VALIDATION_SIZE)
    # TODO: subplot
    # plt.scatter(pca_srl_data[:val_num, 0], pca_srl_data[:val_num, 1], s=10, c='r',label='Validation')
    # plt.scatter(pca_srl_data[val_num:, 0], pca_srl_data[val_num:, 1], s=3, c='b', label='Training')
    # plt.legend()
    # plt.show()


    plt.hist2d(pca_srl_data[:val_num, 0], pca_srl_data[:val_num, 1], bins=val_num//10)
    plt.show()

    plt.hist2d(pca_srl_data[val_num:, 0], pca_srl_data[val_num:, 1], bins=len(pca_srl_data[val_num:])//10 )
    plt.show()

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


    dataSeparator(data_folder=args.data_path, srl_model_path=srl_model_path)
    print("OK")
