#!/bin/bash

BS=32  # Batch size for training SRL model
ENV='KukaMovingButtonGymEnv-v0'
DATASET_NAME='kuka_moving_button_big'
N_ITER=10  # Number of random seeds for training RL
N_CPU=8  # Number of cpu for training PPO
N_EPISODES=300  # For generating data
N_SRL_SAMPLES=20000
N_TIMESTEPS=5000000
ENV_DIM=3  # Only for priors, state dimension
N_EPOCHS=30  # NUM_EPOCHS for training SRL model

rm -rf logs/ICLR
rm -rf srl_zoo/data/$DATASET_NAME*
mkdir logs/ICLR/


if [ "$1" != "full" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "RUNNING DRY RUN MODE, PLEASE USE './run_exp.sh full' FOR NORMAL RUN (waiting 5s)"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo
    sleep 5s
    N_EPISODES=$N_CPU
    N_SRL_SAMPLES=200
    N_TIMESTEPS=5000
    N_EPOCHS=2
    N_ITER=1
fi


#########
# DATASET
#########

python -m environments.dataset_generator --num-cpu $N_CPU --name $DATASET_NAME --num-episode $N_EPISODES --random-target --env $ENV
if [ $? != 0 ]; then
    printf "Error when creating dataset, halting.\n"
	exit $ERROR_CODE
fi


###########
# SRL SPLIT
###########

pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses autoencoder:1:190 reward:1:-1 inverse:2:10 --inverse-model-type mlp --occlusion-percentage 0.3 --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL srl_splits model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model srl_splits --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/ICLR/
    if [ $? != 0 ]; then
        printf "Error when training RL srl_splits model, halting.\n"
        exit $ERROR_CODE
    fi
done


#################
# SRL COMBINATION
#################

pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses autoencoder:1 reward:1 inverse:2 --inverse-model-type mlp --occlusion-percentage 0.3 --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL srl_combination model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model srl_combination --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/ICLR/
    if [ $? != 0 ]; then
        printf "Error when training RL srl_combination model, halting.\n"
        exit $ERROR_CODE
    fi
done


########
# RANDOM
########

pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs 1 --state-dim 200 --training-set-size $N_SRL_SAMPLES --losses random --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL random model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model random --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/ICLR/
    if [ $? != 0 ]; then
        printf "Error when training RL random model, halting.\n"
        exit $ERROR_CODE
    fi
done


################
# ROBOTIC PRIORS
################

pushd srl_zoo
python train.py --data-folder data/$DATASET_NAME/  -bs $BS --epochs $N_EPOCHS --state-dim $ENV_DIM --training-set-size $N_SRL_SAMPLES --losses priors --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL robotic_priors model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model robotic_priors --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/ICLR/
    if [ $? != 0 ]; then
        printf "Error when training RL robotic_priors model, halting.\n"
        exit $ERROR_CODE
    fi
done


############
# SUPERVISED
############

pushd srl_zoo
python -m srl_baselines.supervised --data-folder data/$DATASET_NAME/ --epochs $N_EPOCHS --training-set-size $N_SRL_SAMPLES --relative-pos --no-display-plots
if [ $? != 0 ]; then
    printf "Error when training SRL supervised model, halting.\n"
	exit $ERROR_CODE
fi
popd
for i in `seq 1 $N_ITER`; do
    python -m rl_baselines.train --algo ppo2 --random-target --latest --srl-model supervised --num-timesteps $N_TIMESTEPS --env $ENV --num-cpu $N_CPU --no-vis --log-dir logs/ICLR/
    if [ $? != 0 ]; then
        printf "Error when training RL supervised model, halting.\n"
        exit $ERROR_CODE
    fi
done


###########################
# GROUND_TRUTH & RAW_PIXELS
###########################

python -m rl_baselines.pipeline --algo ppo2 --random-target --srl-model ground_truth raw_pixels --num-timesteps $N_TIMESTEPS --env $ENV --num-iteration $N_ITER --num-cpu $N_CPU --no-vis --log-dir logs/ICLR/
if [ $? != 0 ]; then
    printf "Error when training RL ground_truth & raw_pixels model, halting.\n"
	exit $ERROR_CODE
fi
