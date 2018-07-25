import os
import pickle
import time

import tensorflow as tf
import numpy as np
from stable_baselines.common import tf_util, set_global_seeds
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.acer.acer_simple import Model, Acer, find_trainable_variables, joblib
from stable_baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy

from rl_baselines.base_classes import BaseRLObject
from rl_baselines.policies import AcerMlpPolicy
from rl_baselines.buffer_acer import Buffer
from rl_baselines.utils import createTensorflowSession


class ACERModel(BaseRLObject):
    """
    object containing the interface between baselines.acer and this code base
    ACER: Sample Efficient Actor-Critic with Experience Replay
    """

    LOG_INTERVAL = 1  # log RL model performance every 1 steps
    SAVE_INTERVAL = 20  # Save RL model every 20 steps

    def __init__(self):
        super(ACERModel, self).__init__()
        self.ob_space = None
        self.ac_space = None
        self.policy = None
        self.model = None

    def save(self, save_path, _locals=None):
        assert self.model is not None, "Error: must train or load model before use"
        self.model.save(os.path.dirname(save_path) + "/acer_weights.pkl")
        save_param = {
            "ob_space": self.ob_space,
            "ac_space": self.ac_space,
            "policy": self.policy
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_param, f)

    @classmethod
    def load(cls, load_path, args=None, tf_sess=None):
        sess = tf_util.make_session()

        with open(load_path, "rb") as f:
            save_param = pickle.load(f)
        loaded_model = ACERModel()
        loaded_model.__dict__ = {**loaded_model.__dict__, **save_param}

        policy = {'cnn': AcerCnnPolicy, 'mlp': AcerMlpPolicy}[loaded_model.policy]
        loaded_model.model = policy(sess, loaded_model.ob_space, loaded_model.ac_space, args.num_cpu, n_steps=1,
                                    n_stack=1, reuse=False)

        tf.global_variables_initializer().run(session=sess)
        loaded_params = joblib.load(os.path.dirname(load_path) + "/acer_weights.pkl")
        restores = []
        for p, loaded_p in zip(find_trainable_variables("model"), loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)

        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'mlp'], default='cnn')
        parser.add_argument('--lr-schedule', help='Learning rate schedule', choices=['constant', 'linear'],
                            default='constant')
        return parser

    def getActionProba(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        _, pi, _ = self.model.step(observation, state=None, mask=dones)
        return pi

    def getAction(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        actions, _, _ = self.model.step(observation, state=None, mask=dones)
        return actions

    def train(self, args, callback, env_kwargs=None):
        assert args.num_stack > 1, "ACER only works with '--num-stack' of 2 or more"

        envs = self.makeEnv(args, env_kwargs=env_kwargs)

        # get the associated policy for the architecture requested
        if args.srl_model != "raw_pixels" and args.policy == "cnn":
            args.policy = "mlp"

        self.ob_space = envs.observation_space
        self.ac_space = envs.action_space
        self.policy = args.policy

        self._learn(args, envs, total_timesteps=args.num_timesteps, seed=args.seed, n_stack=1,
                    lr_schedule=args.lr_schedule, callback=callback)

    def _learn(self, args, env, seed, n_steps=20, n_stack=4, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
               max_grad_norm=10, learning_rate=7e-4, lr_schedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99,
               gamma=0.99, log_interval=100, buffer_size=5000, replay_ratio=4, replay_start=1000,
               correction_term=10.0, trust_region=True, alpha=0.99, delta=1, callback=None):
        """
        Train an ACER model.

        :param args: (ArgumentParser) The arguments to learn the model
        :param env: (Gym environment) The environment to learn from
        :param seed: (int) The initial seed for training
        :param n_steps: (int) The number of steps to run for each environment
        :param n_stack: (int) The number of stacked frames
        :param total_timesteps: (int) The total number of samples
        :param q_coef: (float) Q function coefficient for the loss calculation
        :param ent_coef: (float) Entropy coefficient for the loss caculation
        :param max_grad_norm: (float) The maximum value for the gradient clipping
        :param learning_rate: (float) The learning rate
        :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                     'double_linear_con', 'middle_drop' or 'double_middle_drop')
        :param rprop_epsilon: (float) RMS prop optimizer epsilon
        :param rprop_alpha: (float) RMS prop optimizer decay
        :param gamma: (float) Discount factor
        :param log_interval: (int) The number of timesteps before logging.
        :param buffer_size: (int) The buffer size in number of steps
        :param replay_ratio: (float) The number of replay learning per on policy learning on average,
                                     using a poisson distribution
        :param replay_start: (int) The minimum number of steps in the buffer, before learning replay
        :param correction_term: (float) The correction term for the weights
        :param trust_region: (bool) Enable Trust region policy optimization loss
        :param alpha: (float) The decay rate for the Exponential moving average of the parameters
        :param delta: (float) trust region delta value
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        """
        createTensorflowSession()
        if args.policy == 'cnn':
            policy_fn = AcerCnnPolicy
        elif args.policy == 'cnnlstm':
            policy_fn = AcerLstmPolicy
        elif args.policy == 'mlp':
            policy_fn = AcerMlpPolicy
        else:
            raise ValueError("Policy {} not implemented".format(args.policy))

        n_envs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        num_procs = args.num_cpu
        self.model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, n_envs=n_envs, n_steps=n_steps,
                           n_stack=n_stack, num_procs=num_procs, ent_coef=ent_coef, q_coef=q_coef, gamma=gamma,
                           max_grad_norm=max_grad_norm, learning_rate=learning_rate, rprop_alpha=rprop_alpha,
                           rprop_epsilon=rprop_epsilon, total_timesteps=total_timesteps, lr_schedule=lr_schedule,
                           correction_term=correction_term, trust_region=trust_region, alpha=alpha, delta=delta)

        runner = _Runner(env=env, model=self.model, n_steps=n_steps, n_stack=n_stack)
        if replay_ratio > 0:
            _buffer = Buffer(env=env, n_steps=n_steps, n_stack=n_stack, size=buffer_size)
        else:
            _buffer = None
        n_batch = n_envs * n_steps
        acer = Acer(runner, self.model, _buffer, log_interval)
        acer.t_start = time.time()

        # n_batch samples, 1 on_policy call and multiple off-policy calls
        for acer.steps in range(0, total_timesteps, n_batch):
            acer.call(on_policy=True)
            if callback is not None:
                callback(locals(), globals())

            if replay_ratio > 0 and _buffer.has_atleast(replay_start):
                samples_number = np.random.poisson(replay_ratio)
                for _ in range(samples_number):
                    acer.call(on_policy=False)  # no simulation steps in this

        env.close()


class _Runner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps, n_stack):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param n_stack: (int) The number of stacked frames
        """

        super(_Runner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.env = env
        self.n_stack = n_stack
        self.model = model
        self.n_env = n_env = env.num_envs
        self.n_act = env.action_space.n
        self.n_batch = n_env * n_steps

        if len(env.observation_space.shape) > 1:
            self.raw_pixels = True
            obs_height, obs_width, obs_num_channels = env.observation_space.shape
            self.batch_ob_shape = (n_env * (n_steps + 1), obs_height, obs_width, obs_num_channels * n_stack)
            self.obs_dtype = np.uint8
            self.obs = np.zeros((n_env, obs_height, obs_width, obs_num_channels), dtype=self.obs_dtype)
            self.num_channels = obs_num_channels
        else:
            self.raw_pixels = False
            obs_dim = env.observation_space.shape[0]
            self.batch_ob_shape = (n_env * (n_steps + 1), obs_dim * n_stack)
            self.obs_dtype = np.float32
            self.obs = np.zeros((n_env, obs_dim), dtype=self.obs_dtype)
            self.obs_dim = obs_dim

        self.update_obs(self.obs)
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_env)]

    def update_obs(self, obs, dones=None):
        """
        Update the observation for rolling observation with stacking

        :param obs: ([int] or [float]) The input observation
        :param dones: ([bool])
        """
        if self.raw_pixels:
            if dones is not None:
                self.obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
            self.obs = np.roll(self.obs, shift=-self.num_channels, axis=3)
            self.obs[:, :, :, -self.num_channels:] = obs[:, :, :, :]
        else:
            if dones is not None:
                self.obs *= (1 - dones.astype(np.float32))[:, None]
            self.obs = np.roll(self.obs, shift=-self.obs_dim, axis=1)
            self.obs[:, -self.obs_dim:] = obs[:, :]

    def run(self):
        """
        Run a step leaning of the model

        :return: ([float], [float], [float], [float], [float], [bool], [float])
                 encoded observation, observations, actions, rewards, mus, dones, masks
        """
        if self.raw_pixels:
            enc_obs = np.split(self.obs, self.n_stack, axis=3)  # so now list of obs steps
        else:
            enc_obs = np.split(self.obs, self.n_stack, axis=1)  # so now list of obs steps
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        for _ in range(self.n_steps):
            actions, mus, states = self.model.step(self.obs, state=self.states, mask=self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_mus.append(mus)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            # states information for statefull models like LSTM
            self.states = states
            self.dones = dones
            self.update_obs(obs, dones)
            mb_rewards.append(rewards)
            enc_obs.append(obs)
        mb_obs.append(np.copy(self.obs))
        mb_dones.append(self.dones)

        enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones  # Used for statefull models like LSTM's to mask state when done
        mb_dones = mb_dones[:, 1:]  # Used for calculating returns. The dones array is now aligned with rewards

        # shapes are now [n_env, n_steps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks
