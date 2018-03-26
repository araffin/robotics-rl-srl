import os
import time
from collections import deque
import pickle

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.ddpg.memory import Memory
from baselines.ddpg.ddpg import DDPG
from baselines import logger
import baselines.common.tf_util as tf_util

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env
from rl_baselines.utils import createTensorflowSession
from rl_baselines.deepq import CustomDummyVecEnv, WrapFrameStack
from rl_baselines.policies import DDPGActorCNN, DDPGActorMLP, DDPGCriticCNN, DDPGCriticMLP


# Copied from openai ddpg/training, in order to add callback functions
def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
          normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
          popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
          tau=0.01, eval_env=None, param_noise_adaption_interval=50, callback=None):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high

    # Mute the initialization information
    logger.set_level(logger.DISABLED)
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    logger.set_level(logger.INFO)

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with tf_util.single_threaded_session() as sess:
        # Prepare everything.
        agent.saver = tf.train.Saver()
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs, r, done, info = env.step(
                        max_action * action)
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs
                    if callback is not None:
                        callback(locals(), globals())

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                        # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(
                            max_action * eval_action)
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)

            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s' % x)

            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)


def saveDDPG(self, save_path):
    """
    implemented custom save function, as the openAI implementation lacks one
    :param save_path: (str)
    """
    # needed, otherwise tensorflow saver wont find the ckpl files
    save_path = save_path.replace("//", "/")

    # params
    data = {
        "observation_shape": tuple(self.obs0.shape[1:]),
        "action_shape": tuple(self.actions.shape[1:]),
        "param_noise": self.param_noise,
        "action_noise": self.action_noise,
        "gamma": self.gamma,
        "tau": self.tau,
        "normalize_returns": self.normalize_returns,
        "enable_popart": self.enable_popart,
        "normalize_observations": self.normalize_observations,
        "batch_size": self.batch_size,
        "observation_range": self.observation_range,
        "action_range": self.action_range,
        "return_range": self.return_range,
        "critic_l2_reg": self.critic_l2_reg,
        "actor_lr": self.actor_lr,
        "critic_lr": self.critic_lr,
        "clip_norm": self.clip_norm,
        "reward_scale": self.reward_scale
    }
    # used to reconstruct the actor and critic models
    net = {
        "actor_name": self.actor.__class__.__name__,
        "critic_name": self.critic.__class__.__name__,
        "n_actions": self.actor.n_actions,
        "layer_norm": self.actor.layer_norm
    }
    with open(save_path, "wb") as f:
        pickle.dump((data, net), f)

    self.saver.save(self.sess, save_path.split('.')[0] + ".ckpl")


def loadDDPG(self, save_path):
    """
    implemented custom load function, as the openAI implementation lacks one
    :param save_path: (str)
    """
    # needed, otherwise tensorflow saver wont find the ckpl files
    save_path = save_path.replace("//", "/")

    with open(save_path, "rb") as f:
        data, _ = pickle.load(f)
        self.__dict__.update(data)

    self.saver.restore(self.sess, save_path.split('.')[0] + ".ckpl")


def load(save_path, sess):
    """
    Create a DDPG model and load weights from a saved one.
    :param save_path: (str)
    :param sess: (Tensorflow Session)
    :return: (DDPG Object)
    """
    with open(save_path, "rb") as f:
        data, net = pickle.load(f)

    memory = Memory(limit=100, action_shape=data["action_shape"], observation_shape=data["observation_shape"])
    if net["actor_name"] == "DDPGActorMLP":
        actor = DDPGActorMLP(net["n_actions"], layer_norm=net["layer_norm"])
    elif net["actor_name"] == "DDPGActorCNN":
        actor = DDPGActorCNN(net["n_actions"], layer_norm=net["layer_norm"])
    else:
        raise NotImplemented

    if net["critic_name"] == "DDPGCriticMLP":
        critic = DDPGCriticMLP(layer_norm=net["layer_norm"])
    elif net["critic_name"] == "DDPGCriticCNN":
        critic = DDPGCriticCNN(layer_norm=net["layer_norm"])
    else:
        raise NotImplemented

    # add save and load functions to DDPG
    DDPG.save = saveDDPG
    DDPG.load = loadDDPG

    # Mute the initialization information
    logger.set_level(logger.DISABLED)
    agent = DDPG(actor=actor, critic=critic, memory=memory, **data)
    logger.set_level(logger.INFO)

    agent.saver = tf.train.Saver()
    agent.sess = sess

    return agent


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--memory-limit',
                        help='Used to define the size of the replay buffer (in number of observations)', type=int,
                        default=100)
    parser.add_argument('--noise-action',
                        help='The type of action noise added to the output, can be gaussian or OrnsteinUhlenbeck',
                        type=str, default="ou", choices=["none", "normal", "ou"])
    parser.add_argument('--noise-action-sigma', help='The variance of the action noise', type=float, default=0.2)
    parser.add_argument('--noise-param', help='Enable parameter noise', action='store_true', default=False)
    parser.add_argument('--noise-param-sigma', help='The variance of the parameter noise', type=float, default=0.2)
    parser.add_argument('--no-layer-norm', help='Disable layer normalization for the neural networks',
                        action='store_true', default=False)
    parser.add_argument('--batch-size',
                        help='The batch size used for training (use 16 for raw pixels and 64 for srl_model)', type=int,
                        default=16)
    return parser


def main(args, callback):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """

    logger.configure()
    env = CustomDummyVecEnv([make_env(args.env, args.seed, 0, args.log_dir, pytorch=False)])
    # Normalize only raw pixels
    normalize = args.srl_model == ""
    # WARNING: when using framestacking, the memory used by the replay buffer can grow quickly
    env = WrapFrameStack(env, args.num_stack, normalize=normalize)

    createTensorflowSession()
    layer_norm = not args.no_layer_norm

    # Parse noise_type
    action_noise = None
    param_noise = None
    n_actions = env.action_space.shape[-1]
    if args.noise_param:
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=args.noise_param_sigma,
                                             desired_action_stddev=args.noise_param_sigma)

    if args.noise_action == 'normal':
        action_noise = NormalActionNoise(mu=np.zeros(n_actions), sigma=args.noise_action_sigma * np.ones(n_actions))
    elif args.noise_action == 'ou':
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions),
                                                    sigma=args.noise_action_sigma * np.ones(n_actions))

    # Configure components.
    memory = Memory(limit=args.memory_limit, action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape)
    if args.srl_model != "":
        critic = DDPGCriticMLP(layer_norm=layer_norm)
        actor = DDPGActorMLP(n_actions, layer_norm=layer_norm)
    else:
        critic = DDPGCriticCNN(layer_norm=layer_norm)
        actor = DDPGActorCNN(n_actions, layer_norm=layer_norm)

    # add save and load functions to DDPG
    DDPG.save = saveDDPG
    DDPG.load = loadDDPG

    train(
        env=env,
        nb_epochs=500,
        nb_epoch_cycles=20,
        render_eval=False,
        render=False,
        reward_scale=1.,
        param_noise=param_noise,
        actor=actor,
        critic=critic,
        normalize_returns=False,
        normalize_observations=(args.srl_model == ""),
        critic_l2_reg=1e-2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        action_noise=action_noise,
        popart=False,
        gamma=0.99,
        clip_norm=None,
        nb_train_steps=50,
        nb_rollout_steps=100,
        nb_eval_steps=100,
        batch_size=args.batch_size,
        memory=memory,
        callback=callback
    )

    env.close()
