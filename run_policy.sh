#!/usr/bin/env bash





policy="ppo2"
env="OmnirobotEnv-v0"


# those name can not be reuse for a unique run, in other case some folder need to be manually removed
name_circular_policy_folder="circular_on_policy"
name_reaching_policy_folder="reaching_on_policy"
merging_file="merge_CC_SC"



### 0 - Generate datasets for SRL (random policy)
# Dataset 1 (random reaching target)
python -m environments.dataset_generator --num-cpu 8 --name Omnibot_random_simple --env $env --simple-continual --num-episode 250 -f
# Dataset 2 (Circular task)
python -m environments.dataset_generator --num-cpu 8 --name Omnibot_random_circular --env $env --circular-continual --num-episode 250 -f


### 1.1) Train SRL

cd srl_zoo
# Dataset 1 (random reaching target)
python train.py --data-folder data/Omnibot_random_simple  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse
# Dataset 2 (Circular task)
python train.py --data-folder data/Omnibot_random_circular  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse

### 1.2) Train policy
cd ..

# Dataset 1 (random reaching target)
cp config/srl_models_simple.yaml config/srl_models.yaml
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --num-timesteps 500000 --env OmnirobotEnv-v0 --log-dir logs/simple/  --num-cpu 8 --simple-continual  --latest

# Dataset 2 (Circular task)
cp config/srl_models_circular.yaml config/srl_models.yaml
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/circular/  --num-cpu 8 --circular-continual  --latest




# Dataset 1 (random reaching target)
path2policy="logs/simple/OmnirobotEnv-v0/srl_combination/ppo2/"
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy $path2policy --short-episodes --save-path data/ --name reaching_on_policy -sc --latest


# Dataset 2 (Circular task)
path2policy="logs/circular/OmnirobotEnv-v0/srl_combination/ppo2/"
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy $path2policy --short-episodes --save-path data/ --name circular_on_policy -cc --latest


# Merge Datasets

#(/ ! \ it removes the generated dataset for dataset 1 and 2)

python -m environments.dataset_fusioner --merge data/$name_circular_policy_folder\/ data/$name_reaching_policy_folder\/ data/$merging_file

# Copy the merged Dataset to srl_zoo repository
cp -r data/$merging_file srl_zoo/data/$merging_file


### 2.3) Train SRL 1&2

cd srl_zoo
# Dataset 1
python train.py --data-folder data/$merging_file  -bs 32 --epochs 20 --state-dim 200 --training-set-size 30000 --losses autoencoder inverse


### 2.3) Run Distillation

# make a new log folder
mkdir logs/CL_SC_CC
cp config/srl_models_merged.yaml config/srl_models.yaml

# Merged Dataset
cd ..
python -m rl_baselines.train --algo distillation --srl-model srl_combination --env OmnirobotEnv-v0 --log-dir logs/CL_SC_CC --teacher-data-folder srl_zoo/data/$merging_file -cc --distillation-training-set-size 40000 --epochs-distillation 20 --latest

