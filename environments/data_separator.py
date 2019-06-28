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
from functools import partial
from multiprocessing import Pool
from ipdb import set_trace as tt


from state_representation.models import loadSRLModel, getSRLDim
from srl_zoo.utils import loadData
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader, DataLoader

sns.set()

#os.chdir('/home/tete/Robotics-branches/robotics-rl-srl-two/logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')

BATCH_SIZE = 64
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

def dataSrlLoad(data, srl_model=None, state_dim=2, pca_mode=True, normalized=True):
    """

    :param data_folder: (str) the path to the dataset we want to sample
    :param srl_model_path: (str)
    :return: the dataset after the srl evaluation and a pca preprocessd,
             it self, a random sampled training set, validation set
    """

    # load images and other data
    training_data, ground_truth, true_states = data
    images_path = ground_truth['images_path']
    ground_truth_states_dim = true_states.shape[1]

    # we change the path to the local path at the toolbox level
    images_path_copy = ["srl_zoo/data/" + images_path[k] for k in range(images_path.shape[0])]
    images_path = np.array(images_path_copy)

    num_samples = images_path.shape[0]-1  # number of samples

    # indices for all time steps where the episode continues
    #indices = np.array([i for i in range(num_samples-1) if not episode_starts[i + 1]], dtype='int64')
    indices = np.arange(num_samples)

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
        pca_srl_data = PCA(srl_data, dim=ground_truth_states_dim)
    else:
        pca_srl_data = srl_data
    if normalized: # Normilized into -0.5 to +0.5
        for k in range(pca_srl_data.shape[1]):
            pca_srl_data[:, k] = (pca_srl_data[:, k] - np.min(pca_srl_data[:, k])) / (
                        np.max(pca_srl_data[:, k]) - np.min(pca_srl_data[:, k])) - 0.5

    training_indices = np.concatenate(minibatchlist)

    return pca_srl_data, training_indices

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
    import numpy as np
    for p_train in train_set:
        if(1.e-10<np.linalg.norm(p_val-p_train) < threshold):
            # we will delete the data point
            return True
    return False



def index_save(data_set, threshold):
    pbar = tqdm(total = len(data_set))
    deleted = np.zeros(len(data_set)).astype(bool)
    for t, test_point in enumerate(data_set):
        pbar.update(1)
        for k, data_point in enumerate(data_set):
            if(not deleted[k] and 1.e-5<np.linalg.norm(test_point-data_point)<threshold):
                deleted[t] = True
    index = [i for i in range(len(deleted)) if not deleted[i]]
    del_index = [i for i in range(len(deleted)) if deleted[i]]
    return deleted, index, del_index


def dataSelection(data_folder, srl_model_path=None, threshold=0.003):
    state_dim = getSRLDim(srl_model_path)
    srl_model = loadSRLModel(srl_model_path, th.cuda.is_available(), state_dim, env_object=None)

    # load images and other data
    print('Loading data for separation ')
    training_data, ground_truth, true_states, _ = loadData(data_folder, absolute_path=True)
    images_path = ground_truth['images_path']
    ground_truth_states_dim = true_states.shape[1]

    # we change the path to the local path at the toolbox level
    images_path_copy = ["srl_zoo/data/" + images_path[k] for k in range(images_path.shape[0])]
    images_path = np.array(images_path_copy)
    ground_truth_load = np.load(data_folder+ "/ground_truth.npz")
    preprocessed_load = np.load(data_folder+ "/preprocessed_data.npz")

    pca_srl_data, training_indices = dataSrlLoad(data=(training_data, ground_truth, true_states),
                srl_model=srl_model,state_dim=state_dim, pca_mode=True, normalized=True)

    # re- sampling data to have uniform distribution
    print('Resampling data to have a more uniform distribution')
    delete, left_index, del_index = index_save(data_set=pca_srl_data[:16000], threshold=threshold)

    # ground_truth

    ground_truth_data = {}
    preprocessed_data = {}
    if(ground_truth_load['target_positions'].shape[0] == ground_truth_load['ground_truth_states'].shape[0]):
        #This means that the target is moving
        for arr in ground_truth_load.files:
            gt_arr = ground_truth_load[arr]
            ground_truth_data[arr] = gt_arr[left_index]
    else:
        for arr in ground_truth_load.files:
            if(arr != 'target_positions'):
                gt_arr = ground_truth_load[arr]
                ground_truth_data[arr] = gt_arr[left_index]
    for arr in preprocessed_load.files:
        pr_arr = preprocessed_load[arr]
        preprocessed_data[arr] = pr_arr[left_index]

    np.savez(data_folder+ "/preprocessed_data.npz", **preprocessed_data)
    np.savez(data_folder+ "/ground_truth.npz", **ground_truth_data)
    for idx, image in enumerate(images_path):
        if(not idx in left_index):
            try:
                os.remove(image+'.jpg')
            except:
                print("No file named: {}", image+'.jpg')
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
    #srl_model_path = 'srl_zoo/logs/test_dataset/19-06-26_23h44_20_custom_cnn_ST_DIM200_inverse_autoencoder/srl_model.pth'

    #pca_srl_data, index = dataSrlLoad(data_folder=args.data_path, srl_model_path=srl_model_path)
    dataSelection(data_folder=args.data_path, srl_model_path=srl_model_path)
