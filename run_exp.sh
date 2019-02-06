#!/bin/bash


BS=32  # Batch size for training SRL model
ENV='OmnirobotEnv-v0'
DATASET_NAME='omnirobot_simulator'
N_ITER=1  # Number of random seeds for training RL
N_CPU=8  # Number of cpu for training PPO
N_EPISODES=10  # For generating data
N_SRL_SAMPLES=-1
N_TIMESTEPS=1000000
ENV_DIM=4  # Only for priors, state dimension
N_EPOCHS=1  # NUM_EPOCHS for training SRL model

SRL_LOG_FOLDER=logs/simulator
mkdir -p $SRL_LOG_FOLDER
#########
# DATASET
#########

python -m environments.dataset_generator --num-cpu $N_CPU --name $DATASET_NAME --num-episode $N_EPISODES --random-target --env $ENV --toward-target-timesteps-proportion 0.3
if [ $? != 0 ]; then
    printf "Error when creating dataset, halting.\n"
	exit $ERROR_CODE
fi

###########
# SRL SPLIT
###########

pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses autoencoder:1:198 reward:1:-1 inverse:2:2 --inverse-model-type linear --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL srl_splits model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model srl_splits --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir $SRL_LOG_FOLDER --seed $i
    if [ $? != 0 ]; then
        printf "Error when training RL srl_splits model, halting.\n"
        exit $ERROR_CODE
    fi
done


#################
# SRL COMBINATION
#################

pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses autoencoder:1 reward:1 inverse:2 --inverse-model-type linear --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL srl_combination model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model srl_combination --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir $SRL_LOG_FOLDER --seed $i
    if [ $? != 0 ]; then
        printf "Error when training RL srl_combination model, halting.\n"
        exit $ERROR_CODE
    fi
done