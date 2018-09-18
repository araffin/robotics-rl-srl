import json
import pickle as pkl

import numpy as np
import torch as th

import srl_zoo.preprocessing as preprocessing
from srl_zoo.models import CustomCNN, ConvolutionalNetwork, SRLModules, SRLModulesSplit
from srl_zoo.preprocessing import preprocessImage, getNChannels
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

        state_dim = exp_config.get('state-dim', None)
        losses = exp_config.get('losses', None)  # None in the case of baseline models (pca, supervised)
        n_actions = exp_config.get('n_actions', None)  # None in the case of baseline models (pca, supervised)
        model_type = exp_config.get('model-type', None)
        use_multi_view = exp_config.get('multi-view', False)
        inverse_model_type = exp_config.get('inverse-model-type', 'linear')

        assert state_dim is not None, \
            "Please make sure you are loading an up to date model with a conform exp_config file."

        # WARNING: split_dimensions should be loaded as an OrderedDict to keep the order
        # which is not the case for now. However, this is not a problem in test mode
        # (only predicting states, not training)
        split_dimensions = exp_config.get('split-dimensions')
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
    assert not (losses is None and not model_type == 'pca'), \
        "Please make sure you are loading an up to date model with a conform exp_config file."
    assert not (n_actions is None and not (model_type == 'pca' or 'supervised' in losses)), \
        "Please make sure you are loading an up to date model with a conform exp_config file."

    if model is None:
        if use_multi_view:
            preprocessing.preprocess.N_CHANNELS = 6

        model = SRLNeuralNetwork(state_dim, cuda, model_type, n_actions=n_actions, losses=losses,
                                 split_dimensions=split_dimensions, inverse_model_type=inverse_model_type)

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

    def getState(self, observation, env_id=0):
        """
        Predict the state for a given observation

        :param observation: (numpy Number) the input observation
        :param env_id: (int) the environment ID for multi env systems (default=0)
        :return: (numpy Number)
        """
        raise NotImplementedError("getState() not implemented")


class SRLNeuralNetwork(SRLBaseClass):
    """SRL using a neural network as a state representation model"""

    def __init__(self, state_dim, cuda, model_type="custom_cnn", n_actions=None, losses=None, split_dimensions=None,
                 inverse_model_type="linear"):
        """
        :param state_dim: (int)
        :param cuda: (bool)
        :param model_type: (string)
        :param n_actions: action space dimensions (int)
        :param losses: list of optimized losses defining the model (list of string)
        :param split_dimensions: (OrderedDict) Number of dimensions for the different losses
        :param inverse_model_type: (string)
        """
        super(SRLNeuralNetwork, self).__init__(state_dim, cuda)

        self.model_type = model_type
        if "supervised" in losses:
            if "cnn" in model_type:
                self.model = CustomCNN(state_dim)
            elif model_type == "resnet":
                self.model = ConvolutionalNetwork(state_dim)
        # TODO: convert split_dimensions to OrderedDict when loading config
        # for now, using SRLModules for both split and combination (same networks)
        # elif isinstance(split_dimensions, OrderedDict):
        #     self.model = SRLModulesSplit(state_dim=state_dim, action_dim=n_actions, model_type=model_type,
        #                                  cuda=self.cuda, losses=losses, split_dimensions=split_dimensions,
        #                                  inverse_model_type=inverse_model_type)
        else:
            self.model = SRLModules(state_dim=state_dim, action_dim=n_actions, model_type=model_type,
                                    cuda=self.cuda, losses=losses, inverse_model_type=inverse_model_type)
        self.model.eval()

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")
        self.model = self.model.to(self.device)

    def load(self, path):
        self.model.load_state_dict(th.load(path))

    def getState(self, observation, env_id=0):
        if getNChannels() > 3:
            observation = np.dstack((preprocessImage(observation[:, :, :3], convert_to_rgb=False),
                                     preprocessImage(observation[:, :, 3:], convert_to_rgb=False)))
        else:
            observation = preprocessImage(observation, convert_to_rgb=False)

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
        try:
            with open(path, "rb") as f:
                self.model = pkl.load(f)
        except UnicodeDecodeError:
            # Load pickle files saved with python 2
            with open(path, "rb") as f:
                self.model = pkl.load(f, encoding='latin1')

    def getState(self, observation, env_id=0):
        observation = observation[None]  # Add a dimension
        # n_features = width * height * n_channels
        n_features = np.prod(observation.shape[1:])
        # Convert to a 1D array
        observation = observation.reshape(-1, n_features)
        return self.model.transform(observation)[0]
