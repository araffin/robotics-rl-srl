import json
import pickle as pkl

import numpy as np
import torch as th
import cv2

from srl_zoo.models import CustomCNN, ConvolutionalNetwork, SRLModules
from srl_zoo.preprocessing import preprocessImage, getNChannels
import srl_zoo.preprocessing as preprocessing
from srl_zoo.utils import printGreen, printYellow

NOISE_STD = 1e-6  # To avoid NaN for SRL


def getSRLDim(path=None, env_object=None):
    """
    Get the dim of SRL model
    :param path: (str) Path to a srl model
    :param env_object: (gym env object)
    :return: (int)
    """
    if path is not None:
        # Get path to the log folder
        log_folder = '/'.join(path.split('/')[:-1]) + '/'

        with open(log_folder + 'exp_config.json', 'r') as f:
            exp_config = json.load(f)
        try:
            return exp_config['state-dim']
        except KeyError:
            # Old format
            return exp_config['state_dim']
    else:
        return env_object.getGroundTruthDim()


def loadSRLModel(path=None, cuda=False, state_dim=None, env_object=None):
    """
    Load a trained SRL model, it will try to guess the model type from the path
    :param path: (str) Path to a srl model
    :param cuda: (bool)
    :param state_dim: (int)
    :param env_object: (gym env object)
    :return: (srl model)
    """

    model_type, losses, n_actions, model = None, None, None, None

    if path is not None:
        # Get path to the log folder
        log_folder = '/'.join(path.split('/')[:-1]) + '/'
        with open(log_folder + 'exp_config.json', 'r') as f:
            exp_config = json.load(f)
        try:
            state_dim = exp_config['state-dim']
            losses = exp_config.get('losses', None)
            n_actions = exp_config.get('n_actions', 6)
            model_type = exp_config['model-type']
            use_multi_view = exp_config.get('multi-view', False)
        except KeyError:
            # Old format
            state_dim = exp_config['state_dim']
    else:
        assert env_object is not None or state_dim > 0, \
            "When learning states, state_dim must be > 0. Otherwise, set SRL_MODEL_PATH \
            to a srl_model.pth file with learned states."

    if path is not None:
        if 'baselines' in path:
            if 'pca' in path:
                model_type = 'pca'
                model = SRLPCA(state_dim)

    assert model_type is not None or model is not None, \
        "Model type not supported. In order to use loadSRLModel, a path to an SRL model must be given."

    if model is None:
        if use_multi_view:
            if "triplet" in losses:
                preprocessing.preprocess.N_CHANNELS = 9
            else:
                preprocessing.preprocess.N_CHANNELS = 6

        model = SRLNeuralNetwork(state_dim, cuda, model_type, n_actions=n_actions, losses=losses)

    model_name = model_type
    if 'baselines' not in path:
        model_name += " with " + ", ".join(losses)
    printGreen("\nSRL: Using {} \n".format(model_name))

    if path is not None:
        printYellow("Loading trained model...{}".format(path))
        model.load(path)
    return model


class SRLBaseClass(object):
    """Base class for state representation learning models"""

    def __init__(self, state_dim, cuda=False):
        """
        :param state_dim: (int)
        :param cuda: (bool)
        """
        super(SRLBaseClass, self).__init__()
        self.state_dim = state_dim
        self.cuda = cuda
        self.model = None

    def load(self, path):
        """
        Load a trained SRL model
        :param path: (str)
        """
        raise NotImplementedError("load() not implemented")

    def getState(self, observation):
        """
        Predict the state for a given observation
        """
        raise NotImplementedError("getState() not implemented")


class SRLNeuralNetwork(SRLBaseClass):
    """SRL using a neural network as a state representation model"""

    def __init__(self, state_dim, cuda, model_type="custom_cnn", n_actions=None, losses=None):
        """
        :param state_dim: (int)
        :param cuda: (bool)
        :param model_type: (string)
        :param n_actions: action space dimensions (int)
        :param losses: list of optimized losses defining the model (list of string)
        """
        super(SRLNeuralNetwork, self).__init__(state_dim, cuda)

        self.model_type = model_type
        if "supervised" in losses:
            if model_type == "cnn":
                self.model = CustomCNN(state_dim)
            elif model_type == "resnet":
                self.model = ConvolutionalNetwork(state_dim)
        else:
            self.model = SRLModules(state_dim=state_dim, action_dim=n_actions, model_type=model_type,
                                    cuda=self.cuda, losses=losses)
        self.model.eval()

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")
        self.model = self.model.to(self.device)


    def load(self, path):
        """
        :param path: (str)
        """
        self.model.load_state_dict(th.load(path))

    def getState(self, observation):
        """
        :param observation: (numpy tensor)
        :return: (numpy matrix)
        """
        if getNChannels() > 3:
            observation[:, :, :3] = cv2.cvtColor(observation[:, :, :3], cv2.COLOR_RGB2BGR)
            observation[:, :, 3:] = cv2.cvtColor(observation[:, :, 3:], cv2.COLOR_RGB2BGR)
            observation = np.dstack((preprocessImage(observation[:, :, :3]), preprocessImage(observation[:, :, 3:])))
        else:
            # preprocessImage expects a BGR image
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
            observation = preprocessImage(observation)

        # Create 4D Tensor
        observation = observation.reshape(1, *observation.shape)
        # Channel first
        observation = np.transpose(observation, (0, 3, 2, 1))
        observation = th.from_numpy(observation).float().to(self.device)

        with th.no_grad():
            state = self.model.getStates(observation)[0]
        return state.to(th.device("cpu")).detach().numpy()


class SRLPCA(SRLBaseClass):
    """PCA as a state representation"""

    def __init__(self, state_dim):
        super(SRLPCA, self).__init__(state_dim)

    def load(self, path):
        """
        :param path: (str)
        """
        try:
            with open(path, "rb") as f:
                self.model = pkl.load(f)
        except UnicodeDecodeError:
            # Load pickle files saved with python 2
            with open(path, "rb") as f:
                self.model = pkl.load(f, encoding='latin1')

    def getState(self, observation):
        """
        :param observation: (numpy tensor)
        :return: (numpy matrix)
        """
        observation = observation[None]  # Add a dimension
        # n_features = width * height * n_channels
        n_features = np.prod(observation.shape[1:])
        # Convert to a 1D array
        observation = observation.reshape(-1, n_features)
        return self.model.transform(observation)[0]
