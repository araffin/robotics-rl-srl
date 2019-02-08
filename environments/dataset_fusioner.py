from __future__ import division, absolute_import, print_function

import glob
import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm


# TODO add merge (useful when you want to add events by hand)
#      add update-max-distance (when you want to change the max dist)
import cv2
def main():
    parser = argparse.ArgumentParser(description='Dataset Manipulator: useful to fusion two datasets by concatenating '
                                                 'episodes. PS: Deleting sources after fusion into destination folder.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--merge', type=str, nargs=3, metavar=('source_1', 'source_2', 'destination'),
                       default=argparse.SUPPRESS,
                       help='Fusion two datasets by appending the episodes, deleting sources right after.')

    args = parser.parse_args()

    if 'merge' in args:
        # let make sure everything is in order
        assert os.path.exists(args.merge[0]), "Error: dataset '{}' could not be found".format(args.merge[0])
        assert (not os.path.exists(args.merge[2])), "Error: dataset '{}' already exists, cannot rename '{}' to '{}'"\
                                                          .format(args.merge[2], args.merge[0], args.merge[2])
        # create the output
        print(args)
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

        for idx_, gt_load in enumerate([ground_truth_load, ground_truth_load_2], 1):
            for arr in gt_load.files:
                if arr == "images_path":
                    # here, we want to rename just the folder containing the records, hence the black magic
                    # find the "record_" position
                    path = gt_load["images_path"][0]
                    end_pos = path.find("/record_")
                    index_slash = args.merge[2].find("/")
                    inter_pos = path[end_pos:][8:].find("f") + end_pos + 8 #pos in the complete path.
                    for i in tqdm(range(len(gt_load["images_path"]))):
                        path = gt_load["images_path"][i]
                        if idx_ > 1:
                            episode = str(num_episode_dataset_1 + int(path[end_pos+8:inter_pos - 1]))
                            episode = episode.zfill(3)
                            new_record_path = "/record_" + episode + "/" + path[inter_pos:]
                            
                        else:
                            new_record_path = path[end_pos:]
                        ground_truth["images_path"].append(args.merge[2][index_slash+1:] + new_record_path)
                else:
                    # anything that isnt image_path, we dont need to change
                    gt_arr = gt_load[arr]

                    if idx_ > 1:
                        num_episode_dataset = num_episode_dataset_2

                    # HERE check before overwritting that the target is random !+
                    if gt_load[arr].shape[0] < num_episode_dataset:
                        gt_arr = np.repeat(gt_load[arr], num_episode_dataset, axis=0)

                    if idx_ > 1:
                        ground_truth[arr] = np.concatenate((ground_truth[arr], gt_arr), axis=0)
                    else:
                        ground_truth[arr] = gt_arr

        # save the corrected ground_truth
        np.savez(args.merge[2] + "/ground_truth.npz", **ground_truth)

        # load and correct the preprocessed data (actions, rewards etc)
        preprocessed = {}
        preprocessed_load = np.load(args.merge[0] + "/preprocessed_data.npz")
        preprocessed_load_2 = np.load(args.merge[1] + "/preprocessed_data.npz")

        for prepro_load in [preprocessed_load, preprocessed_load_2]:
            for arr in prepro_load.files:
                pr_arr = prepro_load[arr]
                preprocessed[arr] = np.concatenate((preprocessed.get(arr, []), pr_arr), axis=0)
                if arr == "episode_starts":
                    preprocessed[arr] = preprocessed[arr].astype(bool)
                else:
                    preprocessed[arr] = preprocessed[arr].astype(int)

        np.savez(args.merge[2] + "/preprocessed_data.npz", ** preprocessed)

        # remove the old folders
        shutil.rmtree(args.merge[0])
        shutil.rmtree(args.merge[1])


if __name__ == '__main__':
    main()
