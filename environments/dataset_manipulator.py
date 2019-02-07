from __future__ import division, absolute_import, print_function

import glob
import argparse
import os
import shutil

import numpy as np

# TODO add merge (useful when you want to add events by hand)
#      add update-max-distance (when you want to change the max dist)
import cv2
def main():
    parser = argparse.ArgumentParser(description='Dataset Manipulator: useful to fusion two datasets by concatenating '
                                                 'episodes.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rename', type=str, nargs=3, metavar=('source_1', 'source_2', 'destination'),
                       default=argparse.SUPPRESS,
                       help='Fusion two datasets appending the episodes.')

    args = parser.parse_args()

    if 'rename' in args:
        # let make sure everything is in order
        assert os.path.exists(args.rename[0]), "Error: dataset '{}' could not be found".format(args.rename[0])
        assert (not os.path.exists(args.rename[2])), "Error: dataset '{}' already exists, cannot rename '{}' to '{}'"\
                                                          .format(args.rename[2], args.rename[0], args.rename[2])
        # create the output
        print(args)
        os.mkdir(args.rename[2])

        # copy files from first source
        os.rename(args.rename[0] + "/dataset_config.json", args.rename[2] + "/dataset_config.json")
        os.rename(args.rename[0] + "/env_globals.json", args.rename[2] + "/env_globals.json")

        for record in sorted(glob.glob(args.rename[0] + "/record_[0-9]*/*")):
            s = args.rename[2] + "/" + record.split("/")[-2] + '/' + record.split("/")[-1]
            print("s: ", s, record.split("/")[-2])
            os.renames(record, s)

        num_episode_dataset_1 = int(record.split("/")[-2][7:]) + 1

        # copy files from second source
        for record in sorted(glob.glob(args.rename[1] + "/record_[0-9]*/*")):
            episode = str(num_episode_dataset_1 + int(record.split("/")[-2][7:]))
            new_episode = record.split("/")[-2][:-len(episode)] + episode
            s = args.rename[2] + "/" + new_episode + '/' + record.split("/")[-1]
            os.renames(record, s)
        num_episode_dataset_2 = int(record.split("/")[-2][7:]) + 1


        print("num episodes:", num_episode_dataset_1, num_episode_dataset_2)

        # load and correct ground_truth
        ground_truth = {}
        ground_truth_load = np.load(args.rename[0] + "/ground_truth.npz")
        ground_truth_load_2 = np.load(args.rename[1] + "/ground_truth.npz")

        for arr in ground_truth_load.files:
            if arr == "images_path":
                ground_truth["images_path"] = []
                # here, we want to rename just the folder containing the records, hence the black magic
                for i in range(len(ground_truth_load["images_path"])):
                    path = ground_truth_load["images_path"][i]
                    end_pos = path.find("/record_")
                    index_slash = args.rename[2].find("/")

                    ground_truth["images_path"].append(args.rename[2][index_slash+1:] + path[end_pos:])
            else:
                # anything that isnt image_path, we dont need to change
                gt_arr = ground_truth_load[arr]
                print("type 1 : ", type(gt_arr))

                # HERE check before overwritting that the target is random !+
                if ground_truth_load[arr].shape[0] < num_episode_dataset_1:
                    gt_arr = np.repeat(ground_truth_load[arr], num_episode_dataset_1,  axis=0)
                ground_truth[arr] = gt_arr

        for arr in ground_truth_load_2.files:
            if arr == "images_path":
                # here, we want to rename just the folder containing the records, hence the black magic
                for i in range(len(ground_truth_load_2["images_path"])):
                    path = ground_truth_load_2["images_path"][i]
                    end_pos = path.find("/record_")
                    inter_pos = path[end_pos:][8:].find("f")
                    episode = str(num_episode_dataset_1 + int(path[end_pos:][8:][:inter_pos-1]))
                    episode = episode.zfill(3)
                    new_record_path = "/record_" + episode + "/" + path[end_pos:][8:][inter_pos:]

                    index_slash = args.rename[2].find("/")
                    ground_truth["images_path"].append( args.rename[2][index_slash+1:] + new_record_path)
            else:
                # anything that isnt image_path, we dont need to change
                # HERE check before overwritting that the target is random !+
                gt_arr = ground_truth_load_2[arr]
                if ground_truth_load_2[arr].shape[0] < num_episode_dataset_2:
                    gt_arr = np.repeat(ground_truth_load_2[arr], num_episode_dataset_2, axis=0)
                ground_truth[arr] = np.concatenate((ground_truth[arr], gt_arr), axis=0)

        # save the corrected ground_truth
        np.savez(args.rename[2] + "/ground_truth.npz", **ground_truth)

        # load and correct the preprocessed data (actions, rewards etc)
        preprocessed = {}
        preprocessed_load = np.load(args.rename[0] + "/preprocessed_data.npz")
        preprocessed_load_2 = np.load(args.rename[1] + "/preprocessed_data.npz")

        for prepro_load in [preprocessed_load, preprocessed_load_2]:
            for arr in prepro_load.files:
                pr_arr = prepro_load[arr]
                preprocessed[arr] = np.concatenate((preprocessed.get(arr, []), pr_arr), axis=0)
                if arr == "episode_starts":
                    preprocessed[arr] = preprocessed[arr].astype(bool)
                else:
                    preprocessed[arr] = preprocessed[arr].astype(int)

        np.savez(args.rename[2] + "/preprocessed_data.npz", ** preprocessed)

        # remove the old folders
        shutil.rmtree(args.rename[0])
        shutil.rmtree(args.rename[1])

if __name__ == '__main__':
    main()