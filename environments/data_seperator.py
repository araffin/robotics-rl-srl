"""
Script to verify the states distribution
"""
import json
import os
import argparse


import numpy as np
import torch as th
from ipdb import set_trace as tt

from state_representation.models import loadSRLModel, getSRLDim




#os.chdir('/home/tete/Robotics-branches/robotics-rl-srl-two/logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00')



def dataDistribution(srl_model_path=None):
    state_dim = getSRLDim(srl_model_path)
    srl_model = loadSRLModel(srl_model_path,th.cuda.is_available(), state_dim, env_object=None)

    #model = MLPPolicy(output_size=n_actions, input_size=self.state_dim)


    return

def loadKwargs(log_dir):
    with open(os.path.join(args.log_dir, 'args.json')) as data:
        rl_kwargs = json.load(data)
    with open(os.path.join(args.log_dir, 'env_globals.json')) as data:
        env_kwargs = json.load(data)
    return rl_kwargs, env_kwargs

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Train script for RL algorithms")
    parser.add_argument('--log-dir', type=str, default='logs/teacher_policies_for_evaluation/sc/OmnirobotEnv-v0/srl_combination/ppo2/19-06-19_01h10_00'
                        )
    parser.add_argument('')
    args, unknown = parser.parse_known_args()

    rl_kwargs, env_kwargs = loadKwargs(args.log_dir)
    srl_model_path = env_kwargs['srl_model_path']

    dataDistribution(srl_model_path=srl_model_path)


    print("OK")
