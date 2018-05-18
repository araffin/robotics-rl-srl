"""
Enjoy script for OpenAI Baselines
"""
import argparse
import json
import os
from datetime import datetime

import yaml
from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy
from baselines.ppo2.policies import CnnPolicy, MlpPolicy
from baselines.common import tf_util, set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import deepq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

import rl_baselines.ddpg as ddpg
import rl_baselines.ars as ars
import rl_baselines.cma_es as cma_es
from rl_baselines.deepq import CustomDummyVecEnv, WrapFrameStack
from rl_baselines.utils import createTensorflowSession, computeMeanReward, CustomVecNormalize, VecFrameStack
from rl_baselines.policies import MlpPolicyDiscrete, AcerMlpPolicy, CNNPolicyContinuous
from srl_priors.utils import printYellow, printGreen
from environments.utils import makeEnv

supported_models = ['acer', 'ppo2', 'a2c', 'deepq', 'ddpg', 'ars', 'cma-es']


def parseArguments():
    """

    :return: (Arguments)
    """
    parser = argparse.ArgumentParser(description="Load trained RL model")
    parser.add_argument('--env', help='environment ID', type=str, default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--log-dir', help='folder with the saved agent model', type=str, required=True)
    parser.add_argument('--num-timesteps', type=int, default=int(1e4))
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment (show the GUI)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('--plotting', action='store_true', default=False,
                        help='display in the latent space the current observation.')
    return parser.parse_args()


def loadConfigAndSetup(load_args):
    """
    Get the training config and setup the parameters
    :param load_args: (Arguments)
    :return: (dict, str, str, dict, dict)
    """
    with open('config/srl_models.yaml', 'rb') as f:
        srl_models = yaml.load(f)

    for algo in supported_models + ['not_supported']:
        if algo in load_args.log_dir:
            break
    if algo == "not_supported":
        raise ValueError("RL algo not supported for replay")
    printGreen("\n" + algo + "\n")

    load_path = "{}/{}_model.pkl".format(load_args.log_dir, algo)

    env_globals = json.load(open(load_args.log_dir + "kuka_env_globals.json", 'r'))
    train_args = json.load(open(load_args.log_dir + "args.json", 'r'))
    # choose the right paths for the environment
    assert train_args["env"] in srl_models, \
        "Error: environment '{}', is not defined in 'config/srl_models.yaml'".format(train_args["env"])
    srl_models = srl_models[train_args["env"]]

    env_kwargs = {}
    env_kwargs["renders"] = load_args.render
    # load it, if it was defined
    if "action_repeat" in env_globals:
        env_kwargs["action_repeat"] = env_globals['action_repeat']
    elif "ACTION_REPEAT" in env_globals:
        env_kwargs["action_repeat"] = env_globals['ACTION_REPEAT']
    # Reward sparse or shaped
    env_kwargs["shape_reward"] = load_args.shape_reward

    env_kwargs["action_joints"] = train_args["action_joints"]
    env_kwargs["is_discrete"] = not train_args["continuous_actions"]
    env_kwargs["button_random"] = train_args.get('relative', False)
    # Remove up action
    if train_args["env"] == "Kuka2ButtonGymEnv-v0":
        env_kwargs["force_down"] = env_globals.get('force_down', env_globals.get('FORCE_DOWN', True))
    else:
        env_kwargs["force_down"] = env_globals.get('force_down', env_globals.get('FORCE_DOWN', False))

    if train_args["srl_model"] != "":
        train_args["policy"] = "mlp"
        path = srl_models.get(train_args["srl_model"])

        if train_args["srl_model"] == "ground_truth":
            env_kwargs["use_ground_truth"] = True
        elif train_args["srl_model"] == "joints":
            env_kwargs["use_joints"] = True
        elif train_args["srl_model"] == "joints_position":
            env_kwargs["use_ground_truth"] = True
            env_kwargs["use_joints"] = True
        elif path is not None:
            env_kwargs["use_srl"] = True
            env_kwargs["srl_model_path"] = srl_models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(train_args["srl_model"]))

    return train_args, load_path, algo, srl_models, env_kwargs


def createEnv(load_args, train_args, algo, env_kwargs, log_dir="/tmp/gym/test/"):
    """
    Create the Gym environment
    :param load_args: (Arguments)
    :param train_args: (dict)
    :param algo: (str)
    :param env_kwargs: (dict) The extra arguments for the environment
    :param log_dir: (str) Log dir for testing the agent
    :return: (str, SubprocVecEnv)
    """
    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    os.makedirs(log_dir, exist_ok=True)

    if algo not in ["deepq", "ddpg"]:
        envs = SubprocVecEnv([makeEnv(train_args['env'], load_args.seed, i, log_dir, env_kwargs=env_kwargs)
                              for i in range(load_args.num_cpu)])
        envs = VecFrameStack(envs, train_args['num_stack'])
    else:
        if load_args.num_cpu > 1:
            printYellow(algo + " does not support multiprocessing, setting num-cpu=1")
        envs = CustomDummyVecEnv([makeEnv(train_args['env'], load_args.seed, 0, log_dir, env_kwargs=env_kwargs)])

    if train_args["srl_model"] != "":
        # special case where ars type v1 is not normalized
        if algo != "ars" or train_args['algo_type'] != "v1":
            envs = CustomVecNormalize(envs, training=False)
        # Temp fix for experiments where no running average were saved
        try:
            printGreen("Loading saved running average")
            envs.loadRunningAverage(load_args.log_dir)
        except FileNotFoundError:
            envs.training = True
            printYellow("Running Average files not found for CustomVecNormalize, switching to training mode")

    if algo in ["deepq", "ddpg"]:
        # Normalize only raw pixels
        normalize = train_args['srl_model'] == ""
        envs = WrapFrameStack(envs, train_args['num_stack'], normalize=normalize)

    return log_dir, envs


def main():
    load_args = parseArguments()
    train_args, load_path, algo, srl_models, env_kwargs = loadConfigAndSetup(load_args)
    log_dir, envs = createEnv(load_args, train_args, algo, env_kwargs)

    assert not (load_args.plotting and load_args.num_cpu == 1), "Error: cannot run plotting with more than 1 CPU"

    ob_space = envs.observation_space
    ac_space = envs.action_space

    tf.reset_default_graph()
    set_global_seeds(load_args.seed)
    createTensorflowSession()

    sess = tf_util.make_session()
    printYellow("Compiling Policy function....")
    if algo == "acer":
        policy = {'cnn': AcerCnnPolicy, 'mlp': AcerMlpPolicy}[train_args["policy"]]
        # nstack is already handled in the VecFrameStack
        model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, nstack=1, reuse=False)
    elif algo == "a2c":
        policy = {'cnn': CnnPolicy, 'mlp': MlpPolicyDiscrete}[train_args["policy"]]
        model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)
    elif algo == "ppo2":
        if train_args["continuous_actions"]:
            policy = {'cnn': CNNPolicyContinuous, 'mlp': MlpPolicy}[train_args["policy"]]
        else:
            policy = {'cnn': CnnPolicy, 'mlp': MlpPolicyDiscrete}[train_args["policy"]]
        model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)
    elif algo == "ddpg":
        model = ddpg.load(load_path, sess)
    elif algo == "ars":
        model = ars.load(load_path)
    elif algo == "cma-es":
        model = cma_es.load(load_path)

    params = find_trainable_variables("model")

    tf.global_variables_initializer().run(session=sess)

    # Load weights
    if algo in ["acer", "a2c", "ppo2"]:
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        ps = sess.run(restores)
    elif algo == "deepq":
        model = deepq.load(load_path)
    elif algo == "ddpg":
        model.load(load_path)
    elif algo == "cma-es":
        model.policy.setParam(model.best_model)

    dones = [False for _ in range(load_args.num_cpu)]
    obs = envs.reset()
    # print(obs.shape)

    # plotting init
    if load_args.plotting:
        plt.pause(0.1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        old_obs = []
        line, = ax.plot([], [], [], c=[1, 0, 0, 1], label="episode 0")
        point = ax.scatter([0], [0], [0], c=[1, 0, 0, 1])
        fig.legend()

        if train_args["srl_model"] == "ground_truth":
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_zlim([-2, 2])
            delta_obs = [obs[0]]
        elif train_args["srl_model"] in ["vae", "srl_priors"]:
            # we need to rebuild the PCA representation, in order to visualize correctly in 3D
            # load the saved representations
            path = srl_models['log_folder'] + "/".join(
                srl_models.get(train_args["srl_model"]).split("/")[:-1]) + "/image_to_state.json"
            X = np.array(list(json.load(open(path, 'r')).values()))

            # train the PCA and et the limits
            pca = PCA(n_components=3)
            X_new = pca.fit_transform(X)
            ax.set_xlim([np.min(X_new[:, 0]) * 1.2, np.max(X_new[:, 0]) * 1.2])
            ax.set_ylim([np.min(X_new[:, 1]) * 1.2, np.max(X_new[:, 1]) * 1.2])
            ax.set_zlim([np.min(X_new[:, 2]) * 1.2, np.max(X_new[:, 2]) * 1.2])
            delta_obs = [pca.transform([obs[0]])[0]]
        else:
            assert False, "Error: srl_model {} not supported with plotting.".format(train_args["srl_model"])

    n_done = 0
    last_n_done = 0
    episode = 0
    for i in range(load_args.num_timesteps):
        if algo == "acer":
            actions, state, _ = model.step(obs, state=None, mask=dones)
        elif algo in ["a2c", "ppo2"]:
            actions, _, states, _ = model.step(obs, None, dones)
        elif algo == "deepq":
            actions = model(obs[None])[0]
        elif algo == "ddpg":
            actions = model.pi(obs, apply_noise=False, compute_Q=False)[0]
        elif algo == "ars":
            actions = [model.getAction(obs.flatten())]
        elif algo == "cma-es":
            actions = [model.getAction(obs)]
        obs, rewards, dones, _ = envs.step(actions)

        # plotting
        if load_args.plotting:
            if train_args["srl_model"] == "ground_truth":
                ajusted_obs = obs[0]
            elif train_args["srl_model"] in ["vae", "srl_priors"]:
                ajusted_obs = pca.transform([obs[0]])[0]

            # create a new line, if the episode is finished
            if sum(dones) > 0:
                old_obs.append(np.array(delta_obs))
                line.set_c(sns.color_palette()[episode % len(sns.color_palette())])
                episode += 1
                line, = ax.plot([], [], [], c=[1, 0, 0, 1], label="episode " + str(episode))
                fig.legend()
                delta_obs = [ajusted_obs]
            else:
                delta_obs.append(ajusted_obs)

            coor_plt = np.array(delta_obs)
            line._verts3d = (coor_plt[:, 0], coor_plt[:, 1], coor_plt[:, 2])
            point._offsets3d = ([coor_plt[-1, 0]], [coor_plt[-1, 1]], [coor_plt[-1, 2]])

            if i % 5 == 0:
                fig.canvas.draw()
                plt.pause(0.000001)

        if algo in ["deepq", "ddpg"]:
            if dones:
                obs = envs.reset()
            dones = [dones]
        n_done += sum(dones)
        if (n_done - last_n_done) > 1:
            last_n_done = n_done
            _, mean_reward = computeMeanReward(log_dir, n_done)
            print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

    _, mean_reward = computeMeanReward(log_dir, n_done)
    print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))


if __name__ == '__main__':
    main()
