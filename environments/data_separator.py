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
from multiprocessing import Pool
from functools import partial
from ipdb import set_trace as tt



from state_representation.models import loadSRLModel, getSRLDim
from srl_zoo.utils import loadData
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader, DataLoader

sns.set()

#os.chdir('/home/tete/Robotics-branches/robotics-rl-srl-two/logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')

BATCH_SIZE = 256
N_WORKERS = 8
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

def dataSrlLoad(data_folder, srl_model_path=None, pca_mode=True, normalized=True, threshold=0.01):
    """

    :param data_folder: (str) the path to the dataset we want to sample
    :param srl_model_path: (str)
    :return: the dataset after the srl evaluation and a pca preprocessd,
             it self, a random sampled training set, validation set
    """

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
    if pca_mode:
        pca_srl_data = PCA(srl_data, dim=ground_turht_states_dim)
    else:
        pca_srl_data = srl_data
    if normalized: # Normilized into -0.5 to +0.5
        for k in range(pca_srl_data.shape[1]):
            pca_srl_data[:, k] = (pca_srl_data[:, k] - np.min(pca_srl_data[:, k])) / (
                        np.max(pca_srl_data[:, k]) - np.min(pca_srl_data[:, k])) - 0.5

    training_indices = np.concatenate(minibatchlist)

    val_num = int(len(training_indices) * VALIDATION_SIZE)

    #return the index that we dont need to save anymore
    index_del = dataSelection(0.01,pca_srl_data)

    index_save = [i for i in range(len(index_del)) if not index_del[i]]

    return

def plotDistribution(pca_srl_data, val_num):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[24, 8])
    x_min, x_max = pca_srl_data[:, 0].min(), pca_srl_data[:, 0].max()
    y_min, y_max = pca_srl_data[:, 1].min(), pca_srl_data[:, 1].max()
    ax[0].scatter(pca_srl_data[val_num:, 0], pca_srl_data[val_num:, 1], s=5, c='b', label='Training')
    ax[0].scatter(pca_srl_data[:val_num, 0], pca_srl_data[:val_num, 1], s=5, c='r', label='Validation')
    ax[0].legend()
    ax[0].title.set_text('Sample')
    # plt.show()
    ax[1].hist2d(pca_srl_data[:val_num, 0], pca_srl_data[:val_num, 1],
                 bins=100, range=[[x_min, x_max], [y_min, y_max]])
    ax[1].title.set_text('Validation distribution')
    ax[2].hist2d(pca_srl_data[val_num:, 0], pca_srl_data[val_num:, 1],
                 bins=100, range=[[x_min, x_max], [y_min, y_max]])
    ax[2].title.set_text('Training distribution')
    plt.show()


def _del_val(p_val, train_set, threshold):
    """
    if the points are too close to each other, we will delete it from the dataset.
    :param p_val: (np.array) the data points of validation set
    :param train_set: (np.array) the training set
    :param threshold:  (float)
    :return:
    """
    for p_train in train_set:
        if (np.linalg.norm(p_val - p_train) < threshold):
            # we will delete the data point
            return True
        else:
            return False

def dataSelection(threshold, train_set, val_set=None):
    """

    :param val_set: the validation set that we want to resimpling
    :param train_set:
    :param threshold:
    :return:
    """
    #if we dont precise the validation set, the suppression will be on the whole dataset (training set)
    if val_set == None:
        val_set = train_set
    # multiprocessing
    pool = Pool()
    # if index[i] is ture, then we will delete it from the dataset
    index_to_del = pool.map(partial(_del_val, train_set=train_set, threshold=threshold), val_set)

    return index_to_del


def loadKwargs(log_dir):
    with open(os.path.join(log_dir, 'args.json')) as data:
        rl_kwargs = json.load(data)
    with open(os.path.join(log_dir, 'env_globals.json')) as data:
        env_kwargs = json.load(data)
    return rl_kwargs, env_kwargs

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Train script for RL algorithms")
    parser.add_argument('--log-dir', type=str, default='logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')
    parser.add_argument('--data-path', type=str, default='data/test_dataset/')
    args, unknown = parser.parse_known_args()

    rl_kwargs, env_kwargs = loadKwargs(args.log_dir)
    srl_model_path = env_kwargs['srl_model_path']
    tt()
    #srl_model_path = 'srl_zoo/logs/test_dataset/19-06-26_23h44_20_custom_cnn_ST_DIM200_inverse_autoencoder/srl_model.pth'


    dataSrlLoad(data_folder=args.data_path, srl_model_path=srl_model_path)
    print("OK")
