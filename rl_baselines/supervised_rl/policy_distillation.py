import numpy as np
import pickle
import torch as th

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

from rl_baselines.base_classes import BaseRLObject
from rl_baselines.utils import loadRunningAverage, MultiprocessSRLModel, softmax
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader, DataLoader
from srl_zoo.utils import loadData
from state_representation.models import loadSRLModel, getSRLDim
from state_representation.registry import registered_srl, SRLType

N_WORKERS = 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
VALIDATION_SIZE = 0.2  # 20% of training data for validation
MAX_BATCH_SIZE_GPU = 256  # For plotting, max batch_size before having memory issues


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
        inspired from : https://github.com/peterliht/knowledge-distillation-pytorch
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2

        Hyperparameters: temperature and alpha
        :param outputs: output from the student model
        :param labels: label
        :param teacher_outputs: output from the teacher_outputs model
        :return: loss
        """

        alpha = 0.9
        T = 0.01  # temperature empirically found in "policy distillation"
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                 F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels) * (1. - alpha)
        return KD_loss

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):

        N_EPOCHS = args.epochs_distillation
        self.seed = args.seed
        self.batch_size = BATCH_SIZE
        print("We assumed SRL training already done")

        print('Loading data for distillation ')
        training_data, ground_truth, true_states, _ = loadData(args.teacher_data_folder, complete=True)
        rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
        images_path = ground_truth['images_path']
        images_path_copy = ["srl_zoo/data/" + images_path[k] for k in range(images_path.shape[0])]
        images_path = np.array(images_path_copy)
        actions = training_data['actions']
        actions_proba = training_data['actions_proba']
        limit = args.distillation_training_set_size
        actions = actions[:limit]

        num_samples = images_path.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                         for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size)]
        data_loader = DataLoader(minibatchlist, images_path, n_workers=N_WORKERS, multi_view=False,
                                 use_triplets=False, is_training=True, complete_path=True)

        test_minibatchlist = DataLoader.createTestMinibatchList(len(images_path), MAX_BATCH_SIZE_GPU)
        test_data_loader = DataLoader(test_minibatchlist, images_path, n_workers=N_WORKERS, multi_view=False,
                                      use_triplets=False, max_queue_len=1, is_training=False, complete_path=True)

        # Number of minibatches used for validation:
        n_val_batches = np.round(VALIDATION_SIZE * len(minibatchlist)).astype(np.int64)
        val_indices = np.random.permutation(len(minibatchlist))[:n_val_batches]
        # Print some info
        print("{} minibatches for training, {} samples".format(len(minibatchlist) - n_val_batches,
                                                               (len(minibatchlist) - n_val_batches) * BATCH_SIZE))
        print("{} minibatches for validation, {} samples".format(n_val_batches, n_val_batches * BATCH_SIZE))
        assert n_val_batches > 0, "Not enough sample to create a validation set"

        # Stats about actions
        if not args.continuous_actions:
            print('Discrete action space:')
            action_set = set(actions)
            n_actions = int(np.max(actions) + 1)
            print("{} unique actions / {} actions".format(len(action_set), n_actions))
            n_pairs_per_action = np.zeros(n_actions, dtype=np.int64)
            n_obs_per_action = np.zeros(n_actions, dtype=np.int64)
            for i in range(n_actions):
                n_obs_per_action[i] = np.sum(actions == i)

            print("Number of observations per action")
            print(n_obs_per_action)

        else:
            print('Continuous action space:')
            print('Action dimension: {}'.format(self.dim_action))

        assert env_kwargs is not None and registered_srl[args.srl_model][0] == SRLType.SRL, \
            "Please specify a valid srl model for training your policy !"

        self.state_dim = getSRLDim(env_kwargs.get("srl_model_path", None))
        self.srl_model = loadSRLModel(env_kwargs.get("srl_model_path", None),
                                                   th.cuda.is_available(), self.state_dim, env_object=None)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.policy = MLP(input_size=self.state_dim, hidden_size=400, output_size=n_actions)
        if th.cuda.is_available():
            self.policy.cuda()

        learnable_params = [param for param in self.policy.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=1e-3)

        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            val_loss = 0
            pbar = tqdm(total=len(minibatchlist))

            for minibatch_num, (minibatch_idx, obs, _, _, _) in enumerate(data_loader):

                obs = obs.to(self.device)
                validation_mode = minibatch_idx in val_indices
                if validation_mode:
                    self.policy.eval()
                else:
                    self.policy.train()

                # Actions associated to the observations of the current minibatch
                actions_st = actions[minibatchlist[minibatch_idx]]
                actions_proba_st = actions_proba[minibatchlist[minibatch_idx]]

                if not args.continuous_actions:
                    # Discrete actions, rearrange action to have n_minibatch ligns and one column,
                    # containing the int action
                    #print("shapes:", actions_st.shape, actions_proba_st.shape)
                    actions_st = th.from_numpy(actions_st).requires_grad_(False).to(self.device)
                    actions_proba_st = th.from_numpy(actions_proba_st).requires_grad_(False).to(self.device)
                else:
                    # Continuous actions, rearrange action to have n_minibatch ligns and dim_action columns
                    actions_st = th.from_numpy(actions_st).view(-1, self.dim_action).requires_grad_(False).to(
                        self.device)

                state = self.srl_model.model.getStates(obs).to(self.device).detach()
                pred_action = self.policy(state)
                self.optimizer.zero_grad()
                loss = self.loss_fn_kd(pred_action, actions_st, actions_proba_st)

                loss.backward()
                if validation_mode:
                    val_loss += loss.item()
                    # We do not optimize on validation data
                    # so optimizer.step() is not called
                else:
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_batches += 1
                pbar.update(1)
            train_loss = epoch_loss / float(epoch_batches)
            val_loss /= float(n_val_batches)
            pbar.close()
            print("Epoch {:3}/{}, train_loss:{:.4f} val_loss:{:.4f}".format(epoch + 1, N_EPOCHS, train_loss, val_loss))
