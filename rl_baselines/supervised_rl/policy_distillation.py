import time
import pickle

import numpy as np
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from rl_baselines.base_classes import BaseRLObject
from environments import ThreadingType
from environments.registry import registered_env
from environments.utils import makeEnv
from rl_baselines.utils import loadRunningAverage, MultiprocessSRLModel, softmax
from srl_zoo.utils import printYellow


############ DEV ################
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader
from srl_zoo.utils import loadData
from sklearn.model_selection import train_test_split
BATCH_SIZE = 32
TEST_BATCH_SIZE = 256

class MLP(nn.Module):
    def __init__(self, output_size, input_size, hidden_size=400):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):

        input=input.view(-1, self.input_size)

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class PolicyDistillationModel(BaseRLObject):
    """
    Implementation of PolicyDistillation
    """
    def __init__(self):
        super(PolicyDistillationModel, self).__init__()
        # pytorch model to distillate the policy
        #self.model = MLP(input_size=1, hidden_size=400, output_size=1)

    def save(self, save_path, _locals=None):
        assert self.M is not None, "Error: must train or load model before use"
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, load_path, args=None):
        with open(load_path, "rb") as f:
            class_dict = pickle.load(f)
        loaded_model = PolicyDistillationModel()
        loaded_model.__dict__ = class_dict
        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--nothing4instance', help='Number of population (each one has 2 threads)', type=bool,
                            default=True)

        return parser

    def getActionProba(self, observation, dones=None, delta=0):
        """
        returns the action probability distribution, from a given observation.
        :param observation: (numpy int or numpy float)
        :param dones: ([bool])
        :param delta: (numpy float or float) The exploration noise applied to the policy, set to 0 for no noise.
        :return: (numpy float)
        """
        assert self.model is not None, "Error: must train or load model before use"
        action = self.model.forward(observation)
        return softmax(action)

    def getAction(self, observation, dones=None, delta=0):
        """
        From an observation returns the associated action
        :param observation: (numpy int or numpy float)
        :param dones: ([bool])
        :param delta: (numpy float or float) The exploration noise applied to the policy, set to 0 for no noise.
        :return: (numpy float)
        """
        assert self.model is not None, "Error: must train or load model before use"

        self.model.eval()
        return np.argmax(self.model.forward(observation))

    def loss_fn_kd(self, outputs, labels, teacher_outputs):
        """

        :param outputs: output from the student model
        :param labels: label
        :param teacher_outputs: output from the teacher_outputs model
        :return: loss
        """


        """
        inspired from : https://github.com/peterliht/knowledge-distillation-pytorch
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = 0.9
        T = 0.01 # temperature empirically found in "policy distillation"
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                 F.softmax(teacher_outputs / T, dim=1))

        # formula from https://github.com/peterliht/knowledge-distillation-pytorch
        # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
        #                          F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
        #           F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss


    def train(self, args, callback, env_kwargs=None, train_kwargs=None):

        #env = self.makeEnv(args, env_kwargs)
        N_EPOCHS = args.epochs_distillation
        print("mopdel state dim: ",  self.model.output_size)
        self.model = MLP(input_size=env_kwargs["state_dim"], hidden_size=400,
                         output_size=env_kwargs["N_DISCRETE_ACTIONS"])

        #criterion = nn.MSELoss()
        criterion = self.loss_fn_kd()


        print("We assumed SRL training already done")

        print('Loading data for distillation ')
        training_data, ground_truth, true_states, _ = loadData(args.teacher_data_folder)
        rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
        images_path = ground_truth['images_path']
        actions = training_data['actions']

        limit = args.training_set_size
        actions = actions[:limit]

        true_states = true_states.astype(np.float32)
        x_indices = np.arange(len(true_states)).astype(np.int64)

        # Split into train/validation set
        x_train, x_val, y_train, y_val = train_test_split(x_indices, true_states,
                                                          test_size=0.33, random_state=self.seed)

        train_loader = SupervisedDataLoader(x_train, y_train, images_path, batch_size=BATCH_SIZE,
                                            max_queue_len=4, shuffle=True)
        val_loader = SupervisedDataLoader(x_val, y_val, images_path, batch_size=TEST_BATCH_SIZE,
                                          max_queue_len=1, shuffle=False)

        epoch_train_loss = [[] for _ in range(N_EPOCHS)]
        epoch_val_loss = [[] for _ in range(N_EPOCHS)]

        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            train_loss, val_loss = 0, 0
            pbar = tqdm(total=len(train_loader))
            self.model.train()  # Restore train mode

            print("The train_loader should contains observation and target_action ")
            for obs, target_action in train_loader:
                obs, target_action = obs.to(self.device), target_action.to(self.device)

                pred_action = self.model(obs)
                self.optimizer.zero_grad()
                loss = criterion(pred_action, target_action.detach())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                epoch_train_loss[epoch].append(loss.item())

