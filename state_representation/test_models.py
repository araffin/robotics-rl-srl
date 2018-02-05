import time
import json

import environments.kuka_button_gym_env as kuka_env

kuka_env.USE_SRL = True
path = 'srl_priors/logs/kuka_gym_env/modelY2018_M02_D05_H16M06S07_custom_cnn_ProTemCauRep_ST_DIM3_SEED1_priors/srl_model.pth'
# path = 'srl_priors/logs/relative_pos/baselines/supervised_custom_cnn_SEED1_EPOCHS25_BS32/srl_supervised_model.pth'
# path = 'srl_priors/logs/relative_pos/baselines/autoencoder_cnn_ST_DIM3_SEED1_NOISE0_1_EPOCHS25_BS32/srl_ae_model.pth'
kuka_env.SRL_MODEL_PATH = path


env = kuka_env.KukaButtonGymEnv(renders=True, is_discrete=True, name="kuka_gym_env")
env.seed(0)
i = 0
start_time = time.time()
for i_episode in range(1):
    observation = env.reset()
    for t in range(501):
        env.render()
        # print(observation.shape)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        i += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print("{:.2f} FPS".format(i / (time.time() - start_time)))
