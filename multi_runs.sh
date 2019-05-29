echo "multi run: RL raw pixels"
py36
python -m rl_baselines.train --algo ppo2 --env MobileRobotGymEnv-v0 --log-dir logs/ --srl-model raw_pixels --num-timesteps 5000000 --num-cpu=10 --seed 0 --random-target --img-shape="(3,128,128)"
python -m rl_baselines.train --algo ppo2 --env MobileRobotGymEnv-v0 --log-dir logs/ --srl-model raw_pixels --num-timesteps 5000000 --num-cpu=10 --seed 0 --random-target --img-shape="(3,64,64)"
python -m rl_baselines.train --algo ppo2 --env MobileRobotGymEnv-v0 --log-dir logs/ --srl-model raw_pixels --num-timesteps 5000000 --num-cpu=10 --seed 0 --random-target --img-shape="(3,32,32)"
