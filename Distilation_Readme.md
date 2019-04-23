


#  Steps for Distillation





# 1 - Train Baselines


### 0 - Generate datasets for SRL (random policy)

```
cd srl_zoo
# Dataset 1
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_circular --env OmnirobotEnv-v0 --simple-continual --num-episode 250
# Dataset 2
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_random_simple --env OmnirobotEnv-v0 --circular-continual --num-episode 250
```

### 1.1) Train SRL

```
# Dataset 1
python train.py --data-folder data/simple-continual  -bs 32 --epochs 30 --state-dim 200 --training-set-size 30000 --losses autoencoder inverse
# Dataset 2
python train.py --data-folder data/circular-continual  -bs 32 --epochs 30 --state-dim 200 --training-set-size 30000 --losses autoencoder inverse
```


### 1.2) Train policy

```
cd ..

# Dataset 1
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/circular/  --num-cpu 8 --simple-continual  --latest
# Dataset 2
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/circular/  --num-cpu 8 --circular-continual  --latest

```



# 2 - Train Distillation


### 2.1) Generate dataset on Policy


```
# Dataset 1
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 50 --num-cpu 8 --run-policy custom --log-custom-policy logs/simple-continual
# Dataset 2
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 50 --num-cpu 8 --run-policy custom --log-custom-policy logs/circular-continual
# Merge Datasets

?
```

### 2.3) Train SRL 1&2

```
# Dataset 1
python train.py --data-folder data/simple_continual_on_policy  -bs 32 --epochs 2 --state-dim 200 --training-set-size 3000 --losses autoencoder inverse
# Dataset 2
python train.py --data-folder data/circular_continual_on_policy  -bs 32 --epochs 2 --state-dim 200 --training-set-size 3000 --losses autoencoder inverse
```


### 2.3) Run Distillation

```
# Dataset 1
python -m rl_baselines.train --algo distillation --srl-model srl_combination --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/simple/ --simple-continual  --latest --teacher-data-folder srl_zoo/data/simple_continual_on_policy/
# Dataset 2
python -m rl_baselines.train --algo distillation --srl-model srl_combination --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/circular/ --circular-continual  --latest --teacher-data-folder srl_zoo/data/circular_continual_on_policy/
```
