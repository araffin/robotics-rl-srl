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

def channelFirst(tensor):
    """
    Permute the dimension to match pytorch convention
    for images.
    """
    return tensor.permute(0, 3, 1, 2)


class NatureCNN(nn.Module):
    """CNN from Nature paper."""
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


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class CustomCNN(nn.Module):
    """
    Convolutional Neural Network
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    """

    def __init__(self, n_channels):
        super(CustomCNN, self).__init__()
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.conv_layers = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.fc = nn.Linear(6 * 6 * 64, 512)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MLPPolicy(nn.Module):
    """
    :param input_dim: (int)
    :param hidden_dim: (int)
    :param out_dim: (int)
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
        if not self.continuous_actions:
            action = encodeOneHot(action.unsqueeze(1).long(), self.n_actions)

        return self.q_value_net(th.cat([obs, action], dim=1))



class CNNPolicy(nn.Module):
    """
    :param input_dim: (int)
    :param hidden_dim: (int)
    :param out_dim: (int)
    """

    def __init__(self, input_dim, out_dim, hidden_dim=128):
        super(CNNPolicy, self).__init__()

        self.nature_cnn = NatureCNN()

        self.policy_net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden_dim, int(out_dim))
        self.logstd_head = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, x):
        x = self.nature_cnn(x)
        x = self.policy_net(x)
        return self.mean_head(x), self.logstd_head(x)


class CNNValueNetwork(nn.Module):
    """
    :param input_dim: (int)
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(CNNValueNetwork, self).__init__()

        self.nature_cnn = NatureCNN()

        self.value_net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.nature_cnn(x)
        return self.value_net(x)


class CNNQValueNetwork(nn.Module):
    """
    :param input_dim: (int)
    """

    def __init__(self, input_dim, n_actions, continuous_actions, hidden_dim=128):
        super(CNNQValueNetwork, self).__init__()

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
        x = self.nature_cnn(obs)

        if not self.continuous_actions:
            action = encodeOneHot(action.unsqueeze(1).long(), self.n_actions)

        return self.q_value_net(th.cat([x, action], dim=1))
