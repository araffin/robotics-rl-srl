from collections import OrderedDict

import numpy as np
import tensorflow as tf
import pickle

from pytorch_agents.visualize import load_csv
from baselines.common.vec_env.vec_normalize import VecNormalize

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


def isJsonSafe(data):
    """
    Check if an object is json serializable
    :param data: (python object)
    :return: (bool)
    """
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(isJsonSafe(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and isJsonSafe(v) for k, v in data.items())
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
        if isJsonSafe(input_dict[key]):
            output_dict[key] = input_dict[key]
    return output_dict


class WrapVecNormalize(VecNormalize):
    
    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def saveRunningAverage(self, fname):
        if fname == 'ret_rms':
            avg = self.ret_rms
        else:
            avg = self.ob_rms            
        pickle.dump( avg, open(fname + ".f",'wb'))

    def loadRunningAverage(self, file_r_avg):
        if file_r_avg == 'ret_rms':
            self.ret_rms = pickle.load( open(file_r_avg + '.f', 'rb'))
        else:
            self.ob_rms = pickle.load( open(file_r_avg + '.f', 'rb'))