import json
import pickle as pkl

import numpy as np
import torch as th
from torch.autograd import Variable

from srl_priors.models import SRLCustomCNN, SRLConvolutionalNetwork, CNNAutoEncoder, CustomCNN
from srl_priors.preprocessing import preprocessImage
from srl_priors.utils import printGreen, printYellow

NOISE_STD = 1e-6  # To avoid NaN for SRL


def loadSRLModel(path=None, cuda=False, state_dim=None, env_object=None):
    """
    :param path: (str)
    :param cuda: (bool)
    :param state_dim: (int)
    :param env_object: (gym env object)
    :return: (srl model)
    """
    model_type = None
    model = None
    if path is not None:
        log_folder = '/'.join(path.split('/')[:-1]) + '/'

        with open(log_folder + 'exp_config.json', 'r') as f:
            state_dim = json.load(f)['state_dim']
    else:
        assert env_object is not None or state_dim > 0, "When learning states, state_dim must be > 0"

    if env_object is not None:
        model_type = 'ground truth'
        model = SRLGroundTruth(env_object)

    if path is not None:
        if 'baselines' in path:
            if 'pca' in path:
                model_type = 'pca'
                model = SRLPCA(state_dim)
            elif 'supervised' in path and 'custom_cnn' in path:
                model_type = 'supervised_custom_cnn'
            elif 'autoencoder' in path:
                model_type = 'autoencoder'
        else:
            if 'custom_cnn' in path:
                model_type = 'custom_cnn'
            else:
                model_type = 'resnet'

    assert model_type is not None or model is not None, "Model type not supported. In order to use loadSRLModel, a path to an SRL model must be given, or ground truth must be used."

    if model is None:
        model = SRLNeuralNetwork(state_dim, cuda, model_type)

    printGreen("\n Using {} \n".format(model_type))

    if path is not None:
        printYellow("Loading trained model...{}".format(path))
        model.load(path)
    return model


class SRLBaseClass(object):
    """Base class for state representation learning models"""

    def __init__(self, state_dim, cuda=False):
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


class SRLGroundTruth(SRLBaseClass):
    def __init__(self, env_object, state_dim=3):
        super(SRLGroundTruth, self).__init__(state_dim)
        self.env_object = env_object

    def load(self, path=None):
        pass

    def getState(self, observation=None):
        """
        :param observation: (numpy tensor)
        :return: (numpy matrix) TODO: Why not np.array? or why 2d?
        """
        return self.env_object.getArmPos()


class SRLNeuralNetwork(SRLBaseClass):
    """SRL using a neural network as a state representation model"""

    def __init__(self, state_dim, cuda, model_type="custom_cnn"):
        super(SRLNeuralNetwork, self).__init__(state_dim, cuda)

        assert model_type in ['resnet', 'custom_cnn', 'supervised_custom_cnn', 'autoencoder'], \
            "Model type not supported: {}".format(model_type)
        self.model_type = model_type

        if model_type == "custom_cnn":
            self.model = SRLCustomCNN(state_dim, self.cuda, noise_std=NOISE_STD)
        elif model_type == "supervised_custom_cnn":
            self.model = CustomCNN(state_dim)
        elif model_type == "resnet":
            self.model = SRLConvolutionalNetwork(state_dim, self.cuda, noise_std=NOISE_STD)
        elif model_type == "autoencoder":
            self.model = CNNAutoEncoder(self.state_dim)

        self.model.eval()

        if self.cuda:
            self.model.cuda()

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
        observation = preprocessImage(observation)
        # Create 4D Tensor
        observation = observation.reshape(1, *observation.shape)
        # Channel first
        observation = np.transpose(observation, (0, 3, 2, 1))
        observation = Variable(th.from_numpy(observation), volatile=True)
        if self.cuda:
            observation = observation.cuda()

        if self.model_type != "autoencoder":
            state = self.model(observation)
        else:
            state, _ = self.model(observation)

        if self.cuda:
            state = state.cpu()
        return state.data.numpy()


class SRLPCA(SRLBaseClass):
    """PCA as a state representation"""

    def __init__(self, state_dim):
        super(SRLPCA, self).__init__(state_dim)

    def load(self, path):
        """
        :param path: (str)
        """
        try:
            with open(path, "wb") as f:
                self.model = pkl.load(f)
        except UnicodeDecodeError:
            # Load pickle files saved with python 2
            with open(path, "wb") as f:
                self.model = pkl.load(f, encoding='latin1')

    def getState(self, observation):
        """
        :param observation: (numpy tensor)
        :return: (numpy matrix)
        """
        n_features = np.prod(observation.shape[1:])
        observation.reshape(-1, n_features)
        return self.model.transform(observation)
