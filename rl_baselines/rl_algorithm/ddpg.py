import os
import pickle
import time
from collections import deque

import stable_baselines.common.tf_util as tf_util
import numpy as np
import tensorflow as tf
from stable_baselines import logger
from stable_baselines.ddpg.ddpg import DDPG
from stable_baselines.ddpg.memory import Memory
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from mpi4py import MPI

from rl_baselines.base_classes import BaseRLObject
from environments.utils import makeEnv
from rl_baselines.policies import DDPGActorCNN, DDPGActorMLP, DDPGCriticCNN, DDPGCriticMLP
from rl_baselines.utils import createTensorflowSession, CustomVecNormalize, CustomDummyVecEnv, WrapFrameStack, \
    loadRunningAverage, MultiprocessSRLModel


class DDPGModel(BaseRLObject):
    """
    object containing the interface between baselines.ddpg and this code base
    DDPG: Deep Deterministic Policy Gradients
    """
    def __init__(self):
        super(DDPGModel, self).__init__()
        self.model = None

    def save(self, save_path, _locals=None):
        assert self.model is not None, "Error: must train or load model before use"
        self.model.save(save_path)

    @classmethod
    def load(cls, load_path, args=None):
        sess = tf_util.make_session()
        tf.global_variables_initializer().run(session=sess)

        with open(load_path, "rb") as f:
            data, net = pickle.load(f)

        loaded_model = DDPGModel()
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
        loaded_model.model = DDPG(actor=actor, critic=critic, memory=memory, **data)
        logger.set_level(logger.INFO)

        loaded_model.model.saver = tf.train.Saver()
        loaded_model.model.sess = sess

        loaded_model.model.load(load_path)

        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--memory-limit',
                            help='Used to define the size of the replay buffer (in number of observations)', type=int,
                            default=100000)
        parser.add_argument('--noise-action',
                            help='The type of action noise added to the output, can be gaussian or OrnsteinUhlenbeck',
                            type=str, default="ou", choices=["none", "normal", "ou"])
        parser.add_argument('--noise-action-sigma', help='The variance of the action noise', type=float, default=0.2)
        parser.add_argument('--noise-param', help='Enable parameter noise', action='store_true', default=False)
        parser.add_argument('--noise-param-sigma', help='The variance of the parameter noise', type=float, default=0.2)
        parser.add_argument('--no-layer-norm', help='Disable layer normalization for the neural networks',
                            action='store_true', default=False)
        parser.add_argument('--batch-size',
                            help='The batch size used for training (use 16 for raw pixels and 64 for srl_model)',
                            type=int,
                            default=64)
        return parser

    def getActionProba(self, observation, dones=None):
        return [self.getAction(observation, dones=dones)]

    def getAction(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        return self.model.pi(observation[0], apply_noise=False, compute_Q=False)[0]

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though DeepQ is single core only, we need to use the pipe system to work
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(1, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        env = CustomDummyVecEnv([makeEnv(args.env, args.seed, 0, args.log_dir, env_kwargs=env_kwargs)])

        if args.srl_model != "raw_pixels":
            env = CustomVecNormalize(env)
            env = loadRunningAverage(env, load_path_normalise=load_path_normalise)

        # Normalize only raw pixels
        # WARNING: when using framestacking, the memory used by the replay buffer can grow quickly
        return WrapFrameStack(env, args.num_stack, normalize=args.srl_model == "raw_pixels")

    def train(self, args, callback, env_kwargs=None):
        logger.configure()
        env = self.makeEnv(args, env_kwargs)

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
        if args.srl_model != "raw_pixels":
            critic = DDPGCriticMLP(layer_norm=layer_norm)
            actor = DDPGActorMLP(n_actions, layer_norm=layer_norm)
        else:
            critic = DDPGCriticCNN(layer_norm=layer_norm)
            actor = DDPGActorCNN(n_actions, layer_norm=layer_norm)

        memory = Memory(limit=args.memory_limit, action_shape=env.action_space.shape,
                        observation_shape=env.observation_space.shape)

        # add save and load functions to DDPG
        DDPG.save = saveDDPG
        DDPG.load = loadDDPG

        # Mute the initialization information
        logger.set_level(logger.DISABLED)

        self._train_ddpg(
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
            normalize_observations=(args.srl_model == "raw_pixels"),
            critic_l2_reg=1e-2,
            actor_lr=1e-4,
            critic_lr=1e-3,
            action_noise=action_noise,
            popart=False,
            gamma=0.99,
            clip_norm=None,
            nb_train_steps=100,
            nb_rollout_steps=100,
            nb_eval_steps=50,
            batch_size=args.batch_size,
            memory=memory,
            callback=callback,
            num_max_step=args.num_timesteps
        )

        env.close()

    # Copied from openai ddpg/training, in order to add callback functions
    def _train_ddpg(self, env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor,
                    critic, normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
                    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
                    tau=0.01, eval_env=None, param_noise_adaption_interval=50, callback=None, logging_level=logger.INFO,
                    num_max_step=int(1e6 * 1.1)):
        """
        Runs the training of the Deep Deterministic Policy Gradien (DDPG) model

        DDPG: https://arxiv.org/pdf/1509.02971.pdf

        :param env: (Gym Environment) the environment
        :param nb_epochs: (int) the number of training epochs
        :param nb_epoch_cycles: (int) the number cycles within each epoch
        :param render_eval: (bool) enable rendering of the evalution environment
        :param reward_scale: (float) the value the reward should be scaled by
        :param render: (bool) enable rendering of the environment
        :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
        :param actor: (TensorFlow Tensor) the actor model
        :param critic: (TensorFlow Tensor) the critic model
        :param normalize_returns: (bool) should the critic output be normalized
        :param normalize_observations: (bool) should the observation be normalized
        :param critic_l2_reg: (float) l2 regularizer coefficient
        :param actor_lr: (float) the actor learning rate
        :param critic_lr: (float) the critic learning rate
        :param action_noise: (ActionNoise) the action noise type (can be None)
        :param popart: (bool) enable pop-art normalization of the critic output
            (https://arxiv.org/pdf/1602.07714.pdf)
        :param gamma: (float) the discount rate
        :param clip_norm: (float) clip the gradients (disabled if None)
        :param nb_train_steps: (int) the number of training steps
        :param nb_rollout_steps: (int) the number of rollout steps
        :param nb_eval_steps: (int) the number of evalutation steps
        :param batch_size: (int) the size of the batch for learning the policy
        :param memory: (Memory) the replay buffer
        :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
        :param eval_env: (Gym Environment) the evaluation environment (can be None)
        :param param_noise_adaption_interval: (int) apply param noise every N steps
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        :param logging_level: (int) the logging level (can be DEBUG=10, INFO=20, WARN=30, ERROR=40, DISABLED=50)
        :param num_max_step: (int) number of env steps to optimizer for
        """
        rank = MPI.COMM_WORLD.Get_rank()

        assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
        max_action = env.action_space.high
        logger.log('scaling actions by {} before executing in env'.format(max_action))
        self.model = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                          param_noise=param_noise, action_noise=action_noise, gamma=gamma, tau=tau,
                          normalize_returns=normalize_returns, enable_popart=popart,
                          normalize_observations=normalize_observations, batch_size=batch_size,
                          critic_l2_reg=critic_l2_reg, actor_lr=actor_lr, critic_lr=critic_lr, clip_norm=clip_norm,
                          reward_scale=reward_scale)

        logger.set_level(logger.INFO)
        logger.log('Using agent with the following configuration:')
        logger.log(str(self.model.__dict__.items()))

        # Set up logging stuff only for a single worker.
        if rank == 0:
            tf.train.Saver()

        eval_episode_rewards_history = deque(maxlen=100)
        episode_rewards_history = deque(maxlen=100)
        with tf_util.single_threaded_session() as sess:
            # Prepare everything.
            self.model.saver = tf.train.Saver()
            self.model.initialize(sess)
            sess.graph.finalize()

            self.model.reset()
            obs = env.reset()
            if eval_env is not None:
                eval_obs = eval_env.reset()
            episode_reward = 0.
            episode_step = 0
            episodes = 0
            step = 0
            total_steps = 0

            start_time = time.time()

            epoch_episode_rewards = []
            epoch_episode_steps = []
            epoch_start_time = time.time()
            epoch_actions = []
            epoch_qs = []
            epoch_episodes = 0
            for epoch in range(nb_epochs):
                for _ in range(nb_epoch_cycles):
                    # Perform rollouts.
                    for t_rollout in range(nb_rollout_steps):
                        if total_steps >= num_max_step:
                            return

                        # Predict next action.
                        action, q_value = self.model.policy(obs, apply_noise=True, compute_q=True)
                        assert action.shape == env.action_space.shape

                        # Execute next action.
                        if rank == 0 and render:
                            env.render()
                        assert max_action.shape == action.shape
                        # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        new_obs, reward, done, _ = env.step(max_action * action)
                        step += 1
                        total_steps += 1
                        if rank == 0 and render:
                            env.render()
                        episode_reward += reward
                        episode_step += 1

                        # Book-keeping.
                        epoch_actions.append(action)
                        epoch_qs.append(q_value)
                        self.model.store_transition(obs, action, reward, new_obs, done)
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

                            self.model.reset()
                            obs = env.reset()

                    # Train.
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []
                    for t_train in range(nb_train_steps):
                        # Adapt param noise, if necessary.
                        if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                            distance = self.model.adapt_param_noise()
                            epoch_adaptive_distances.append(distance)

                        critic_loss, actor_loss = self.model.train()
                        epoch_critic_losses.append(critic_loss)
                        epoch_actor_losses.append(actor_loss)
                        self.model.update_target_net()

                    # Evaluate.
                    eval_episode_rewards = []
                    eval_qs = []
                    if eval_env is not None:
                        eval_episode_reward = 0.
                        for t_rollout in range(nb_eval_steps):
                            if total_steps >= num_max_step:
                                return

                            eval_action, eval_q = self.model.policy(eval_obs, apply_noise=False, compute_q=True)
                            # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                            eval_obs, eval_r, eval_done, _ = eval_env.step(max_action * eval_action)
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
                stats = self.model.get_stats()
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
                combined_stats['total/steps_per_second'] = float(step) / float(duration)
                combined_stats['total/episodes'] = episodes
                combined_stats['rollout/episodes'] = epoch_episodes
                combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                # Evaluation statistics.
                if eval_env is not None:
                    combined_stats['eval/return'] = eval_episode_rewards
                    combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                    combined_stats['eval/Q'] = eval_qs
                    combined_stats['eval/episodes'] = len(eval_episode_rewards)

                def as_scalar(scalar):
                    """
                    check and return the input if it is a scalar, otherwise raise ValueError

                    :param scalar: (Any) the object to check
                    :return: (Number) the scalar if x is a scalar
                    """
                    if isinstance(scalar, np.ndarray):
                        assert scalar.size == 1
                        return scalar[0]
                    elif np.isscalar(scalar):
                        return scalar
                    else:
                        raise ValueError('expected scalar, got %s' % scalar)

                combined_stats_sums = MPI.COMM_WORLD.allreduce(
                    np.array([as_scalar(x) for x in combined_stats.values()]))
                combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                # Total statistics.
                combined_stats['total/epochs'] = epoch + 1
                combined_stats['total/steps'] = step

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')
                logdir = logger.get_dir()
                if rank == 0 and logdir:
                    if hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                            pickle.dump(env.get_state(), file_handler)
                    if eval_env and hasattr(eval_env, 'get_state'):
                        with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                            pickle.dump(eval_env.get_state(), file_handler)


def saveDDPG(self, save_path):
    """
    implemented custom save function, as the openAI implementation lacks one
    :param save_path: (str)
    """
    # needed, otherwise tensorflow saver wont find the ckpl files
    save_path = save_path.replace("//", "/")

    # params
    data = {
        "observation_shape": tuple([int(x) for x in self.obs0.shape[1:]]),
        "action_shape": tuple([int(x) for x in self.actions.shape[1:]]),
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
