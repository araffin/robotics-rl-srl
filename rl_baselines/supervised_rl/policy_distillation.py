import numpy as np
import pickle
import torch as th

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

from rl_baselines.base_classes import BaseRLObject
from srl_zoo.models.models import CustomCNN
from srl_zoo.preprocessing.data_loader import SupervisedDataLoader, DataLoader
from srl_zoo.utils import loadData
from state_representation.models import loadSRLModel, getSRLDim

N_WORKERS = 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
VALIDATION_SIZE = 0.2  # 20% of training data for validation
MAX_BATCH_SIZE_GPU = 256  # For plotting, max batch_size before having memory issues
RENDER_HEIGHT = 224
RENDER_WIDTH = 224


class MLPPolicy(nn.Module):
    def __init__(self, output_size, input_size, hidden_size=400):
        super(MLPPolicy, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):

        input = input.view(-1, self.input_size)
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


class CNNPolicy(nn.Module):
    def __init__(self, output_size):
        super(CNNPolicy, self).__init__()
        self.model = CustomCNN(state_dim=output_size)

    def forward(self, input):
        return F.softmax(self.model(input), dim=1)


class PolicyDistillationModel(BaseRLObject):
    """
    Implementation of PolicyDistillation
    """
    def __init__(self):
        super(PolicyDistillationModel, self).__init__()

    def save(self, save_path, _locals=None):
        assert self.model is not None, "Error: must train or load model before use"
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
        if len(observation.shape) > 2:
            observation = np.transpose(observation, (0, 3, 2, 1))
        observation = th.from_numpy(observation).float().requires_grad_(False).to(self.device)
        action = self.model.forward(observation).detach().cpu().numpy()
        return action #softmax(action)

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
        if len(observation.shape) > 2:
            observation = np.transpose(observation, (0, 3, 2, 1))
        observation = th.from_numpy(observation).float().requires_grad_(False).to(self.device)
        return [np.argmax(self.model.forward(observation).detach().cpu().numpy())]

    def loss_fn_kd(self, outputs, teacher_outputs):
        """
        inspired from : https://github.com/peterliht/knowledge-distillation-pytorch
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2

        Hyperparameters: temperature and alpha
        :param outputs: output from the student model
        :param teacher_outputs: output from the teacher_outputs model
        :return: loss
        """
        return (outputs - teacher_outputs).pow(2).sum(1).mean()

    def loss_mse(self, outputs, teacher_outputs):
        MSELoss = nn.MSELoss()(outputs, teacher_outputs)

        return MSELoss

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):

        N_EPOCHS = args.epochs_distillation
        self.seed = args.seed
        self.batch_size = BATCH_SIZE
        print("We assumed SRL training already done")

        print('Loading data for distillation ')
        training_data, ground_truth, true_states, _ = loadData(args.teacher_data_folder, absolute_path=True)
        rewards, episode_starts = training_data['rewards'], training_data['episode_starts']

        images_path = ground_truth['images_path']
        actions = training_data['actions']
        actions_proba = training_data['actions_proba']

        if args.distillation_training_set_size > 0:
            limit = args.distillation_training_set_size
            actions = actions[:limit]
            images_path = images_path[:limit]
            episode_starts = episode_starts[:limit]

        images_path_copy = ["srl_zoo/data/" + images_path[k] for k in range(images_path.shape[0])]
        images_path = np.array(images_path_copy)

        num_samples = images_path.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                         for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size)]
        data_loader = DataLoader(minibatchlist, images_path, n_workers=N_WORKERS, multi_view=False,
                                 use_triplets=False, is_training=True, absolute_path=True)

        test_minibatchlist = DataLoader.createTestMinibatchList(len(images_path), MAX_BATCH_SIZE_GPU)
        test_data_loader = DataLoader(test_minibatchlist, images_path, n_workers=N_WORKERS, multi_view=False,
                                      use_triplets=False, max_queue_len=1, is_training=False, absolute_path=True)

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
            n_obs_per_action = np.zeros(n_actions, dtype=np.int64)
            for i in range(n_actions):
                n_obs_per_action[i] = np.sum(actions == i)

            print("Number of observations per action")
            print(n_obs_per_action)

        else:
            print('Continuous action space:')
            print('Action dimension: {}'.format(self.dim_action))

        # Here the default SRL model is assumed to be raw_pixels
        self.state_dim = RENDER_HEIGHT * RENDER_WIDTH * 3
        self.srl_model = None

        # TODO: add sanity checks & test for all possible SRL for distillation
        if env_kwargs["srl_model"] == "raw_pixels":
            self.model = CNNPolicy(n_actions)
        else:
            self.state_dim = getSRLDim(env_kwargs.get("srl_model_path", None))
            self.srl_model = loadSRLModel(env_kwargs.get("srl_model_path", None),
                                          th.cuda.is_available(), self.state_dim, env_object=None)
            self.model = MLPPolicy(n_actions, self.state_dim)

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        if th.cuda.is_available():
            self.model.cuda()

        learnable_params = self.model.parameters()
        self.optimizer = th.optim.Adam(learnable_params, lr=1e-3)

        best_error = np.inf
        best_model_path = "{}/distillation_model.pkl".format(args.log_dir)

        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            val_loss = 0
            pbar = tqdm(total=len(minibatchlist))

            for minibatch_num, (minibatch_idx, obs, _, _, _) in enumerate(data_loader):

                obs = obs.to(self.device)
                validation_mode = minibatch_idx in val_indices
                if validation_mode:
                    self.model.eval()
                else:
                    self.model.train()

                # Actions associated to the observations of the current minibatch
                actions_st = actions[minibatchlist[minibatch_idx]]
                actions_proba_st = actions_proba[minibatchlist[minibatch_idx]]

                if not args.continuous_actions:
                    # Discrete actions, rearrange action to have n_minibatch ligns and one column,
                    # containing the int action
                    actions_st = th.from_numpy(actions_st).requires_grad_(False).to(self.device)
                    actions_proba_st = th.from_numpy(actions_proba_st).requires_grad_(False).to(self.device)
                else:
                    # Continuous actions, rearrange action to have n_minibatch ligns and dim_action columns
                    actions_st = th.from_numpy(actions_st).view(-1, self.dim_action).requires_grad_(False).to(
                        self.device)

                state = obs.detach() if self.srl_model is None \
                    else self.srl_model.model.getStates(obs).to(self.device).detach()
                pred_action = self.model.forward(state)
                #loss = self.loss_fn_kd(pred_action, actions_proba_st)
                loss = self.loss_mse(pred_action, actions_proba_st.float())

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
                self.optimizer.zero_grad()

            train_loss = epoch_loss / float(epoch_batches)
            val_loss /= float(n_val_batches)
            pbar.close()
            print("Epoch {:3}/{}, train_loss:{:.4f} val_loss:{:.4f}".format(epoch + 1, N_EPOCHS, train_loss, val_loss))

            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                self.save(best_model_path)
