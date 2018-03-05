from collections import OrderedDict

import numpy as np
import tensorflow as tf

from pytorch_agents.visualize import load_csv


def createTensorflowSession():
    """
    Create tensorflow session with specific argument
    to prevent it from taking all gpu memory
    """
    # Let Tensorflow choose the device
    config = tf.ConfigProto(allow_soft_placement=True)
    # Prevent tensorflow from taking all the gpu memory
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


def computeMeanReward(log_dir, last_n_episodes):
    """
    Compute the mean reward for the last n episodes
    :param log_dir: (str)
    :param last_n_episodes: (int)
    :return: (bool, numpy array)
    """
    result, _ = load_csv(log_dir)
    if len(result) == 0:
        return False, 0
    y = np.array(result)[:, 1]
    return True, y[-last_n_episodes:].mean()


def safeJson(data):
    """
    Check if an object is json serializable
    :param data: (python object)
    :return: (bool)
    """
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safeJson(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safeJson(v) for k, v in data.items())
    return False


def filterJSONSerializableObjects(input_dict):
    """
    Filter and sort entries of a dictionnary
    to save it as a json
    :param input_dict: (dict)
    :return: (OrderedDict)
    """
    output_dict = OrderedDict()
    for key in sorted(input_dict.keys()):
        if safeJson(input_dict[key]):
            output_dict[key] = input_dict[key]
    return output_dict
