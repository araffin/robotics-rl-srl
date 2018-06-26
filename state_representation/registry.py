from state_representation import SRLType
from environments.kuka_gym.kuka_button_gym_env import KukaButtonGymEnv

# format NAME: (SRLType, LIMITED_TO_ENV)
registered_srl = {
    "raw_pixels":      (SRLType.ENVIRONMENT, None),
    "ground_truth":    (SRLType.ENVIRONMENT, None),
    "joints":          (SRLType.ENVIRONMENT, [KukaButtonGymEnv]),
    "joints_position": (SRLType.ENVIRONMENT, [KukaButtonGymEnv]),
    "robotic_priors":  (SRLType.SRL, None),
    "supervised":      (SRLType.SRL, None),
    "autoencoder":     (SRLType.SRL, None),
    "vae":             (SRLType.SRL, None),
    "pca":             (SRLType.SRL, None)
}
