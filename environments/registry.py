import subprocess

import gym
from gym.envs import registry

from environments import PlottingType, ThreadingType
from environments.srl_env import SRLGymEnv
from environments.kuka_gym.kuka_button_gym_env import KukaButtonGymEnv
from environments.kuka_gym.kuka_rand_button_gym_env import KukaRandButtonGymEnv
from environments.kuka_gym.kuka_2button_gym_env import Kuka2ButtonGymEnv
from environments.kuka_gym.kuka_moving_button_gym_env import KukaMovingButtonGymEnv
from environments.mobile_robot.mobile_robot_env import MobileRobotGymEnv
from environments.mobile_robot.mobile_robot_2target_env import MobileRobot2TargetGymEnv
from environments.mobile_robot.mobile_robot_1D_env import MobileRobot1DGymEnv
from environments.mobile_robot.mobile_robot_line_target_env import MobileRobotLineTargetGymEnv
from environments.gym_baxter.baxter_env import BaxterEnv
from environments.robobo_gym.robobo_env import RoboboEnv
from environments.omnirobot_gym.omnirobot_env import OmniRobotEnv

def register(_id, **kvargs):
    if _id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(_id, **kvargs)


def isXAvailable():
    """
    check to see if running in terminal with X or not
    :return: (bool)
    """
    try:
        p = subprocess.Popen(["xset", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()
        return p.returncode == 0
    except FileNotFoundError:
        # Return False if xset is not present on the machine
        return False


registered_env = {
    "KukaButtonGymEnv-v0":            (KukaButtonGymEnv, SRLGymEnv, PlottingType.PLOT_3D, ThreadingType.PROCESS),
    "KukaRandButtonGymEnv-v0":        (KukaRandButtonGymEnv, KukaButtonGymEnv, PlottingType.PLOT_3D, ThreadingType.PROCESS),
    "Kuka2ButtonGymEnv-v0":           (Kuka2ButtonGymEnv, KukaButtonGymEnv, PlottingType.PLOT_3D, ThreadingType.PROCESS),
    "KukaMovingButtonGymEnv-v0":      (KukaMovingButtonGymEnv, KukaButtonGymEnv, PlottingType.PLOT_3D, ThreadingType.PROCESS),
    "MobileRobotGymEnv-v0":           (MobileRobotGymEnv, SRLGymEnv, PlottingType.PLOT_2D, ThreadingType.PROCESS),
    "MobileRobot2TargetGymEnv-v0":    (MobileRobot2TargetGymEnv, MobileRobotGymEnv, PlottingType.PLOT_2D, ThreadingType.PROCESS),
    "MobileRobot1DGymEnv-v0":         (MobileRobot1DGymEnv, MobileRobotGymEnv, PlottingType.PLOT_2D, ThreadingType.PROCESS),
    "MobileRobotLineTargetGymEnv-v0": (MobileRobotLineTargetGymEnv, MobileRobotGymEnv, PlottingType.PLOT_2D, ThreadingType.PROCESS),
    "Baxter-v0":                      (BaxterEnv, SRLGymEnv, PlottingType.PLOT_3D, ThreadingType.NONE),
    "RoboboGymEnv-v0":                (RoboboEnv, SRLGymEnv, PlottingType.PLOT_2D, ThreadingType.NONE),
    "OmnirobotEnv-v0":                (OmniRobotEnv, SRLGymEnv, PlottingType.PLOT_2D, ThreadingType.PROCESS),
}

# Environments only available when running in a terminal with X (hence only imported when available):
if isXAvailable():
    # Catch if X available, but GL context unavailable. 
    # This prevents SSH crashing when X is passed without GL context.
    try:
        from environments.car_racing.car_env import CarRacingEnv
        registered_env["CarRacingGymEnv-v0"] = (CarRacingEnv, SRLGymEnv, PlottingType.PLOT_2D, ThreadingType.NONE)
    except:
        pass


for name, (env_class, _, _, _) in registered_env.items():
    register(
        _id=name,
        entry_point=env_class.__module__ + ":" + env_class.__name__,
        timestep_limit=None,  # This limit is changed in the file
        reward_threshold=None  # Threshold at which the environment is considered as solved
    )
