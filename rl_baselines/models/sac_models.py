import torch as th
import torch.nn as nn
import torch.nn.functional as F


def encodeOneHot(tensor, n_dim):
    """
    One hot encoding for a given tensor
    :param tensor: (th Tensor)
    :param n_dim: (int) Number of dimensions
    :return: (th.Tensor)
    """
    encoded_tensor = th.Tensor(tensor.shape[0], n_dim).zero_().to(tensor.device)
    return encoded_tensor.scatter_(1, tensor, 1.)


class NatureCNN(nn.Module):
    """
    CNN from Nature paper.
    :param n_channels: (int)
    """

    def __init__(self, n_channels):
        super(NatureCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # TODO: check the padding
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )
        self.fc = nn.Linear(36864, 512)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class MLPPolicy(nn.Module):
    """
    :param input_dim: (int)
    :param out_dim: (int)
    :param hidden_dim: (int)
    """

    def __init__(self, input_dim, out_dim, hidden_dim=128):
        super(MLPPolicy, self).__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden_dim, int(out_dim))
        self.logstd_head = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, x):
        x = self.policy_net(x)
        return self.mean_head(x), self.logstd_head(x)


class MLPValueNetwork(nn.Module):
    """
    :param input_dim: (int)
    :param hidden_dim: (int)
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(MLPValueNetwork, self).__init__()

        self.value_net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.value_net(x)


class MLPQValueNetwork(nn.Module):
    """
    :param input_dim: (int)
    :param n_actions: (int)
    :param continuous_actions: (bool)
    :param hidden_dim: (int)
    """

    def __init__(self, input_dim, n_actions, continuous_actions, hidden_dim=128):
        super(MLPQValueNetwork, self).__init__()

        self.continuous_actions = continuous_actions
        self.n_actions = n_actions
        self.q_value_net = nn.Sequential(
            nn.Linear(int(input_dim) + int(n_actions), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        """
        :param obs: (th.Tensor)
        :param action: (th.Tensor)
        :return: (th.Tensor)
        """
        if not self.continuous_actions:
            action = encodeOneHot(action.unsqueeze(1).long(), self.n_actions)

        return self.q_value_net(th.cat([obs, action], dim=1))
