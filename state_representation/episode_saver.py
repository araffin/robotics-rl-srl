import os
import json
import time

import cv2
import numpy as np

from srl_zoo.utils import printYellow
from rl_baselines.utils import filterJSONSerializableObjects
from state_representation.client import SRLClient


class EpisodeSaver(object):
    """
    Save the experience data from a gym env to a file
    and notify the srl server so it learns from the gathered data
    :param name: (str)
    :param max_dist: (float)
    :param state_dim: (int)
    :param globals_: (dict) Environments globals
    :param learn_every: (int)
    :param learn_states: (bool)
    :param path: (str)
    :param relative_pos: (bool)
    """

    def __init__(self, name, max_dist, state_dim=-1, globals_=None, learn_every=3, learn_states=False,
                 path='data/', relative_pos=False):
        super(EpisodeSaver, self).__init__()
        self.name = name
        self.data_folder = path + name
        self.path = path
        try:
            os.makedirs(self.data_folder)
        except OSError:
            printYellow("Folder already exist")

        self.actions = []
        self.rewards = []
        self.images = []
        self.target_positions = []
        self.episode_starts = []
        self.ground_truth_states = []
        self.images_path = []
        self.episode_step = 0
        self.episode_idx = -1
        self.episode_folder = None
        self.episode_success = False
        self.state_dim = state_dim
        self.learn_states = learn_states
        self.learn_every = learn_every  # Every n episodes, learn a state representation
        self.srl_model_path = ""
        self.n_steps = 0
        self.max_steps = 10000

        self.dataset_config = {'relative_pos': relative_pos, 'max_dist': str(max_dist)}
        with open("{}/dataset_config.json".format(self.data_folder), "w") as f:
            json.dump(self.dataset_config, f)

        if globals_ is not None:
            # Save environments parameters
            with open("{}/env_globals.json".format(self.data_folder), "w") as f:
                json.dump(filterJSONSerializableObjects(globals_), f)

        if self.learn_states:
            self.socket_client = SRLClient(self.name)
            self.socket_client.waitForServer()

    def saveImage(self, observation):
        """
        Write an image to disk
        :param observation: (numpy matrix) BGR image
        """
        image_path = "{}/{}/frame{:06d}".format(self.data_folder, self.episode_folder, self.episode_step)
        relative_image_path = "{}/{}/frame{:06d}".format(self.name, self.episode_folder, self.episode_step)
        self.images_path.append(relative_image_path)
        
        # in the case of dual/multi-camera
        if observation.shape[2] > 3:
            observation1 = cv2.cvtColor(observation[:, :, :3], cv2.COLOR_BGR2RGB)
            observation2 = cv2.cvtColor(observation[:, :, 3:], cv2.COLOR_BGR2RGB)

            cv2.imwrite("{}_1.jpg".format(image_path), observation1)
            cv2.imwrite("{}_2.jpg".format(image_path), observation2)
        else:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            cv2.imwrite("{}.jpg".format(image_path), observation)

    def reset(self, observation, target_pos, ground_truth):
        """
        Called when starting a new episode
        :param observation: (numpy matrix) BGR Image
        :param target_pos: (numpy array)
        :param ground_truth: (numpy array)
        """
        # only reset if the array is empty, or the a reset has not already occured
        if len(self.episode_starts) == 0 or self.episode_starts[-1] is False:
            self.episode_idx += 1

            if self.learn_states and (self.episode_idx + 1) % self.learn_every == 0 and self.n_steps <= self.max_steps:
                print("Learning a state representation ...")
                start_time = time.time()
                ok, self.srl_model_path = self.socket_client.waitForSRLModel(self.state_dim)
                print("Took {:.2f}s".format(time.time() - start_time))

            self.episode_step = 0
            self.episode_success = False
            self.episode_folder = "record_{:03d}".format(self.episode_idx)
            os.makedirs("{}/{}".format(self.data_folder, self.episode_folder), exist_ok=True)

            self.episode_starts.append(True)
            self.target_positions.append(target_pos)
            self.ground_truth_states.append(ground_truth)
            self.saveImage(observation)

    def step(self, observation, action, reward, done, ground_truth_state):
        """
        :param observation: (numpy matrix) BGR Image
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        :param ground_truth_state: (numpy array)
        """
        self.episode_step += 1
        self.n_steps += 1
        self.rewards.append(reward)
        self.actions.append(action)
        if reward > 0:
            self.episode_success = True

        if not done:
            self.episode_starts.append(False)
            self.ground_truth_states.append(ground_truth_state)
            self.saveImage(observation)
        else:
            # Save the gathered data at the end of each episode
            self.save()

    def save(self):
        """
        Write data and ground truth to disk
        """
        # Sanity checks
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.episode_starts)
        assert len(self.actions) == len(self.images_path)
        assert len(self.actions) == len(self.ground_truth_states)
        assert len(self.target_positions) == self.episode_idx + 1

        data = {
            'rewards': np.array(self.rewards),
            'actions': np.array(self.actions),
            'episode_starts': np.array(self.episode_starts)
        }

        ground_truth = {
            'target_positions': np.array(self.target_positions),
            'ground_truth_states': np.array(self.ground_truth_states),
            'images_path': np.array(self.images_path)
        }
        print("Saving preprocessed data...")
        np.savez('{}/preprocessed_data.npz'.format(self.data_folder), **data)
        np.savez('{}/ground_truth.npz'.format(self.data_folder), **ground_truth)


class LogRLStates(object):
    """
    Save the experience data (states, normalized states, actions, rewards) from a gym env to a file
    during RL training. It is useful to debug SRL models.
    :param log_folder: (str)
    """

    def __init__(self, log_folder):
        super(LogRLStates, self).__init__()

        self.log_folder = log_folder + 'log_srl/'
        try:
            os.makedirs(self.log_folder)
        except OSError:
            printYellow("Folder already exist")

        self.actions = []
        self.rewards = []
        self.states = []
        self.normalized_states = []

    def reset(self, normalized_state, state):
        """
        Called when starting a new episode
        :param normalized_state: (numpy array)
        :param state: (numpy array)
        """
        # self.episode_starts.append(True)
        self.normalized_states.append(normalized_state)
        self.states.append(np.squeeze(state))

    def step(self, normalized_state, state, action, reward, done):
        """
        :param normalized_state: (numpy array)
        :param state: (numpy array)
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        """
        self.rewards.append(reward)
        self.actions.append(action)

        if not done:
            self.normalized_states.append(normalized_state)
            self.states.append(np.squeeze(state))
        else:
            # Save the gathered data at the end of each episode
            self.save()

    def save(self):
        """
        Write data to disk
        """
        # Sanity checks
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.normalized_states)
        assert len(self.actions) == len(self.states)

        data = {
            'rewards': np.array(self.rewards),
            'actions': np.array(self.actions),
            'states': np.array(self.states),
            'normalized_states': np.array(self.normalized_states),
        }

        np.savez('{}/full_log.npz'.format(self.log_folder), **data)
        np.savez('{}/states_rewards.npz'.format(self.log_folder),
                 **{'states': data['states'], 'rewards': data['rewards']})
        np.savez('{}/normalized_states_rewards.npz'.format(self.log_folder),
                 **{'states': data['normalized_states'], 'rewards': data['rewards']})
