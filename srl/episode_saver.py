import os
import json

import cv2
import numpy as np

from .client import SRLClient


class EpisodeSaver(object):
    """
    Save the experience data from a gym env to a file
    :param name: (str)
    :param max_dist: (float)
    :param path: (str)
    :param relative_pos: (bool)
    """

    def __init__(self, name, max_dist, path='srl_priors/data/', relative_pos=False):
        super(EpisodeSaver, self).__init__()
        self.name = name
        self.data_folder = path + name
        try:
            os.makedirs(self.data_folder)
        except OSError:
            print("Folder already exist")
            
        self.actions = []
        self.rewards = []
        self.images = []
        self.button_positions = []
        self.episode_starts = []
        self.arm_states = []
        self.images_path = []
        self.episode_step = 0
        self.episode_idx = -1
        self.episode_folder = None
        self.episode_success = False

        # TODO: convert max dist (to button) to lower/upper bound
        self.dataset_config = {'relative_pos': relative_pos, 'max_dist': str(max_dist)}
        with open("{}/dataset_config.json".format(self.data_folder), "w") as f:
            json.dump(self.dataset_config, f)

        socket_client = SRLClient()
        socket_client.waitForServer()


    def saveImage(self, observation):
        """
        :param observation: (numpy matrix) BGR image
        """
        image_path = "{}/{}/frame{:06d}.jpg".format(self.name, self.episode_folder, self.episode_step)
        self.images_path.append(image_path)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        cv2.imwrite("data/{}".format(image_path), observation)

    def reset(self, observation, button_pos, arm_state):
        """
        :param observation: (numpy matrix) BGR Image
        :param button_pos: ([float])
        :param arm_state: ([float])
        """
        self.episode_idx += 1
        self.episode_step = 0
        self.episode_success = False
        self.episode_folder = "record_{:03d}".format(self.episode_idx)
        os.makedirs("{}/{}".format(self.data_folder, self.episode_folder), exist_ok=True)

        self.episode_starts.append(True)
        self.button_positions.append(button_pos)
        self.arm_states.append(arm_state)
        self.saveImage(observation)

    def step(self, observation, action, reward, done, arm_state):
        """
        :param observation: (numpy matrix) BGR Image
        :param action: (int)
        :param reward: (float)
        :param done: (bool) whether the episode is done or not
        :param arm_state: ([float])
        """
        self.episode_step += 1
        self.rewards.append(reward)
        self.actions.append(action)
        if reward > 0:
            self.episode_success = True

        if not done:
            self.episode_starts.append(False)
            self.arm_states.append(arm_state)
            self.saveImage(observation)
        else:
            # Save the gathered data at the end of each episode
            self.save()

    def save(self):
        """
        Write data and ground truth to disk
        """
        assert len(self.actions) == len(self.rewards)
        assert len(self.actions) == len(self.episode_starts)
        assert len(self.actions) == len(self.images_path)
        assert len(self.actions) == len(self.arm_states)
        assert len(self.button_positions) == self.episode_idx + 1

        data = {
            'rewards': np.array(self.rewards),
            'actions': np.array(self.actions),
            'episode_starts': np.array(self.episode_starts)
        }

        ground_truth = {
            'button_positions': np.array(self.button_positions),
            'arm_states': np.array(self.arm_states),
            # 'actions_deltas': action_to_idx.keys(),
            'images_path': np.array(self.images_path)
        }
        print("Saving preprocessed data...")
        np.savez('{}/preprocessed_data.npz'.format(self.data_folder), **data)
        np.savez('{}/ground_truth.npz'.format(self.data_folder), **ground_truth)
