import argparse
import numpy as np
from os.path import join
import shutil

def main():
    parser = argparse.ArgumentParser(description='Change existed dataset whose ground_truth is global position to relative position')
    parser.add_argument('--data-src', type=str, default=None, help='source data folder (global position)')
    parser.add_argument('--data-dst', type=str, default=None, help='destination data folder, (relative position)')


    args = parser.parse_args()
    assert args.data_src is not None
    assert args.data_dst is not None
    ground_truth = np.load(join(args.data_src, 'ground_truth.npz'))
    preprocessed_data = np.load(join(args.data_src, 'preprocessed_data.npz'))

    shutil.copytree(args.data_src, args.data_dst)
    episode_starts = preprocessed_data['episode_starts']
    print(ground_truth.keys())
    ground_truth_states = ground_truth['ground_truth_states']
    target_position = ground_truth['target_positions']

    episode_num = -1
    
    print(ground_truth_states.shape)
    for i in range(ground_truth_states.shape[0]):
        if(episode_starts[i] == True):
            episode_num += 1
        ground_truth_states[i,:] = ground_truth_states[i,:] - target_position[episode_num]
    new_ground_truth = {}
    for key in ground_truth.keys():
        if key != 'ground_truth_states':
            new_ground_truth[key] = ground_truth[key]
    new_ground_truth['ground_truth_states'] = ground_truth_states
    np.savez(join(args.data_dst, 'ground_truth.npz'), **new_ground_truth)

if __name__ == '__main__':
    main()
