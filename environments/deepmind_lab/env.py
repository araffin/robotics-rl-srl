import numpy as np
import torch.multiprocessing as multiprocessing
import timeit
import time
import torch
from environments.srl_env import SRLGymEnv
from gym import spaces
import deepmind_lab
from environments.deepmind_lab.constants import *
from matplotlib import pyplot as plt


translated = {'forward':0, 'right':1, 'left': 2}
translated_a = {0:'forward', 1:'right', 2:'left'}

def translate_action(action):

    action = translated_a[action]

    translated_action = actions_dict[action]

    return translated_action

actions_dict = {
"left":np.array([-20,0,0,0,0,0,0], dtype=np.intc),
"right":np.array([20,0,0,0,0,0,0], dtype=np.intc),
"forward":np.array([0,0,0,1,0,0,0], dtype=np.intc)
  }


def getGlobals():
    """
    :return: (dict)
    """
    return globals()


class DeepmindLab_v0(SRLGymEnv):

    def __init__(self, env_rank=0, srl_pipe=None, srl_model="raw_pixels", **_kwargs):
        super(DeepmindLab_v0, self).__init__(srl_model=srl_model,
                                        relative_pos=False,
                                        env_rank=env_rank,
                                        srl_pipe=srl_pipe)

        self.vae = torch.load(VAE_PATH, map_location={'cuda:0': 'cpu'})
        print('VAE loaded is:', VAE_PATH)

        self.action_space = spaces.Discrete(ACTION_SPACE)

        # SRL model
        if self.srl_model != "raw_pixels":
            if self.srl_model == "ground_truth":
                self.state_dim = self.getGroundTruthDim()
            if self.srl_model == "vae":
                self.state_dim = Z_DIM
            self.dtype = np.float32
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=self.dtype)
        else:
            self.dtype = np.uint8
            self.observation_space = spaces.Box(low=0, high=255, shape=INPUT_DIM,
                                                dtype=self.dtype)

        self.env = deepmind_lab.Lab('simple_demo', ['RGB_INTERLEAVED'],{'fps': '3000', 'width': '128', 'height': '128'})
        self.env.reset()

        self.ep_index = 0
        self.num_ts = 0

        self.render = False

        if self.render:
            fig = plt.figure()
            data = np.zeros((128, 128, 3))
            self.im = plt.imshow(data)
            plt.axis('off')
            plt.grid(b=None)



    def make(self):

        self.env = deepmind_lab.Lab('simple_demo', ['RGB_INTERLEAVED'],{'fps': '3000', 'width': '128', 'height': '128'})

        return

    def reset(self):

        self.num_ts = 0

        self.env.reset()

        # Initialize with 500 no-op.
        self.env.step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc), num_steps=500)

        observation = self.env.observations()['RGB_INTERLEAVED'][42:106,32:96]

        if self.srl_model == "vae" or self.srl_model=='raw_pixels_1D':
            observation = self.getSRLState(observation)

        return observation

    def step(self, action):

        _action = translate_action(action)

        reward = self.env.step(_action, num_steps=1)

        observation = self.env.observations()['RGB_INTERLEAVED'][42:106,32:96]

        if self.render:
            self.im.set_data(self.env.observations()['RGB_INTERLEAVED'])
            plt.pause(0.05)

        self.num_ts += 1

        if self.num_ts > HORIZON:
            done = True
        else:
            done = False

        if done:
            self.ep_index += 1

        if self.srl_model == "vae":
            observation = self.getSRLState(observation)

        return observation, reward, done, {}

    def getTargetPos(self):

        return None

    def getGroundTruthDim(self):

        return 2

    def getGroundTruth(self):

        return None

    def render(self, mode='human'):

        return self.env.observations()['RGB_INTERLEAVED'][42:106,32:96]

    def close(self):

        return None

    def getSRLState(self, observation):

        o_t = observation/255.

        # VAE predict
        o_t = o_t.reshape((1,) + INPUT_DIM).transpose((0,3,1,2)) # reshape into (1,3,64,64)
        if GPU_MODE:
            o_t = torch.from_numpy(o_t).cuda()
        else:
            o_t = torch.from_numpy(o_t).float()
        z_t = self.vae.forward(o_t, encode=True) # get latent vector

        feature = z_t

        recon = self.vae.forward(z_t, decode=True)
        np.save('recon', recon.detach().cpu().numpy())

        return feature.detach().cpu().numpy()