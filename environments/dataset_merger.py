from __future__ import division, absolute_import, print_function

import glob
import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm

# List of all possible labels identifying a task,
#   for experiments in Continual Learning scenari.
CONTINUAL_LEARNING_LABELS = ['CC', 'SC', 'EC', 'SQC']
CL_LABEL_KEY = "continual_learning_label"


def main():
    parser = argparse.ArgumentParser(description='Dataset Manipulator: useful to merge two datasets by concatenating '
                                                 + 'episodes. PS: Deleting sources after merging into the destination '
                                                 + 'folder.')
    parser.add_argument('--continual-learning-labels', type=str, nargs=2, metavar=('label_1', 'label_2'),
                        default=argparse.SUPPRESS, help='Labels for the continual learning RL distillation task.')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force the merge, even if it overrides something else,' 
                             ' including the destination if it exist')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--merge', type=str, nargs=3, metavar=('source_1', 'source_2', 'destination'),
                       default=argparse.SUPPRESS,
                       help='Merge two datasets by appending the episodes, deleting sources right after.')

    args = parser.parse_args()

    if 'merge' in args:
        # let make sure everything is in order
        assert os.path.exists(args.merge[0]), "Error: dataset '{}' could not be found".format(args.merge[0])

        # If the merge file exists already, delete it for the convenince of updating student's policy
        if os.path.exists(args.merge[2]) or os.path.exists(args.merge[2] + '/'):
            assert args.force, "Error: destination directory '{}' already exists".format(args.merge[2])
            shutil.rmtree(args.merge[2])

        if 'continual_learning_labels' in args:
            assert args.continual_learning_labels[0] in CONTINUAL_LEARNING_LABELS \
                   and args.continual_learning_labels[1] in CONTINUAL_LEARNING_LABELS, \
                   "Please specify a valid Continual learning label to each dataset to be used for RL distillation !"

        # create the output
        os.mkdir(args.merge[2])

        # copy files from first source
        os.rename(args.merge[0] + "/dataset_config.json", args.merge[2] + "/dataset_config.json")
        os.rename(args.merge[0] + "/env_globals.json", args.merge[2] + "/env_globals.json")

        for record in sorted(glob.glob(args.merge[0] + "/record_[0-9]*/*")):
            s = args.merge[2] + "/" + record.split("/")[-2] + '/' + record.split("/")[-1]
            os.renames(record, s)

        num_episode_dataset_1 = int(record.split("/")[-2][7:]) + 1

        # copy files from second source
        for record in sorted(glob.glob(args.merge[1] + "/record_[0-9]*/*")):
            episode = str(num_episode_dataset_1 + int(record.split("/")[-2][7:]))
            new_episode = record.split("/")[-2][:-len(episode)] + episode
            s = args.merge[2] + "/" + new_episode + '/' + record.split("/")[-1]
            os.renames(record, s)
        num_episode_dataset_2 = int(record.split("/")[-2][7:]) + 1

        # load and correct ground_truth
        ground_truth = {}
        ground_truth_load = np.load(args.merge[0] + "/ground_truth.npz")
        ground_truth_load_2 = np.load(args.merge[1] + "/ground_truth.npz")
        ground_truth["images_path"] = []
        num_episode_dataset = num_episode_dataset_1

        index_slash = args.merge[2].find("/")
        index_margin_str = len("/record_")
        directory_str = args.merge[2][index_slash+1:]

        for idx_, gt_load in enumerate([ground_truth_load, ground_truth_load_2], 1):
            for arr in gt_load.files:
                if arr == "images_path":
                    # here, we want to rename just the folder containing the records, hence the black magic

                    for i in tqdm(range(len(gt_load["images_path"])),
                                  desc="Update of paths (Folder " + str(1+idx_) + ")"):
                        # find the "record_" position
                        path = gt_load["images_path"][i]
                        end_pos = path.find("/record_")
                        inter_pos = path.find("/frame")  # pos in the complete path.

                        if idx_ > 1:
                            episode = str(num_episode_dataset_1 + int(path[end_pos + index_margin_str: inter_pos]))
                            episode = episode.zfill(3)
                            new_record_path = "/record_" + episode + path[inter_pos:]
                        else:
                            new_record_path = path[end_pos:]
                        ground_truth["images_path"].append(directory_str + new_record_path)
                else:
                    # anything that isnt image_path, we dont need to change
                    gt_arr = gt_load[arr]

                    if idx_ > 1:
                        num_episode_dataset = num_episode_dataset_2

                    # HERE check before overwritting that the target is random !+
                    if gt_load[arr].shape[0] < num_episode_dataset:
                        gt_arr = np.repeat(gt_load[arr], num_episode_dataset, axis=0)

                    if idx_ > 1:
                        if gt_arr.shape == ground_truth[arr].shape:
                            ground_truth[arr] = np.concatenate((ground_truth[arr], gt_arr), axis=0)

                    else:
                        ground_truth[arr] = gt_arr

        # save the corrected ground_truth
        np.savez(args.merge[2] + "/ground_truth.npz", **ground_truth)

        # load and correct the preprocessed data (actions, rewards etc)
        preprocessed = {}
        preprocessed_load = np.load(args.merge[0] + "/preprocessed_data.npz")
        preprocessed_load_2 = np.load(args.merge[1] + "/preprocessed_data.npz")

        dataset_1_size = preprocessed_load["actions"].shape[0]
        dataset_2_size = preprocessed_load_2["actions"].shape[0]

        # Concatenating additional information: indices of episode start, action probabilities, CL labels...
        for idx, prepro_load in enumerate([preprocessed_load, preprocessed_load_2]):
            for arr in prepro_load.files:
                pr_arr = prepro_load[arr]

                to_class = None
                if arr == "episode_starts":
                    to_class = bool
                elif arr == "actions_proba":
                    to_class = float
                else:
                    to_class = int
                if preprocessed.get(arr, None) is None:
                    preprocessed[arr] = pr_arr.astype(to_class)
                else:
                    preprocessed[arr] = np.concatenate((preprocessed[arr].astype(to_class),
                                                        pr_arr.astype(to_class)), axis=0)
            if 'continual_learning_labels' in args:
                if preprocessed.get(CL_LABEL_KEY, None) is None:
                    preprocessed[CL_LABEL_KEY] = \
                        np.array([args.continual_learning_labels[idx] for _ in range(dataset_1_size)])
                else:
                    preprocessed[CL_LABEL_KEY] = \
                        np.concatenate((preprocessed[CL_LABEL_KEY], np.array([args.continual_learning_labels[idx]
                                                                              for _ in range(dataset_2_size)])), axis=0)

        np.savez(args.merge[2] + "/preprocessed_data.npz", ** preprocessed)

        # remove the old folders
        shutil.rmtree(args.merge[0])
        shutil.rmtree(args.merge[1])


if __name__ == '__main__':
    main()
