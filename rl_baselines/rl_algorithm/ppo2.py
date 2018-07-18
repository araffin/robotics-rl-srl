import pickle
import os
import time

import numpy as np
import joblib
from stable_baselines.common import tf_util
from stable_baselines.acer.acer_simple import find_trainable_variables
from stable_baselines.ppo2.ppo2 import Model, constfn, Runner, deque, explained_variance, safe_mean
from stable_baselines import logger
import tensorflow as tf

from rl_baselines.base_classes import BaseRLObject
from rl_baselines.policies import PPO2MLPPolicy, PPO2CNNPolicy


class PPO2Model(BaseRLObject):
    """
    object containing the interface between baselines.ppo2 and this code base
    PPO2: Proximal Policy Optimization (GPU Implementation)
    """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(PPO2Model, self).__init__()
        self.ob_space = None
        self.ac_space = None
        self.policy = None
        self.model = None
        self.continuous_actions = None
        self.states = None

    def save(self, save_path, _locals=None):
        assert self.model is not None, "Error: must train or load model before use"
        self.model.save(os.path.dirname(save_path) + "/ppo2_weights.pkl")
        save_param = {
            "ob_space": self.ob_space,
            "ac_space": self.ac_space,
            "policy": self.policy,
            "continuous_actions": self.continuous_actions
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_param, f)

    @classmethod
    def load(cls, load_path, args=None):
        sess = tf_util.make_session()

        with open(load_path, "rb") as f:
            save_param = pickle.load(f)
        loaded_model = PPO2Model()
        loaded_model.__dict__ = {**loaded_model.__dict__, **save_param}

        # MLP: multi layer perceptron
        # CNN: convolutional neural netwrok
        # LSTM: Long Short Term Memory
        # LNLSTM: Layer Normalization LSTM
        continuous = loaded_model.continuous_actions
        policy = {'cnn': PPO2CNNPolicy(continuous=continuous),
                  'cnn-lstm': PPO2CNNPolicy(continuous=continuous, reccurent=True),
                  'cnn-lnlstm': PPO2CNNPolicy(continuous=continuous, reccurent=True, normalised=True),
                  'mlp': PPO2MLPPolicy(continuous=continuous),
                  'lstm': PPO2MLPPolicy(continuous=continuous, reccurent=True),
                  'lnlstm': PPO2MLPPolicy(continuous=continuous, reccurent=True, normalised=True)}[loaded_model.policy]

        if policy is None:
            raise ValueError(loaded_model.policy + " not implemented for " + (
                "discrete" if loaded_model.continuous_actions else "continuous") + " action space.")

        loaded_model.model = policy(sess, loaded_model.ob_space, loaded_model.ac_space, args.num_cpu, nsteps=1,
                                    reuse=False)
        loaded_model.states = loaded_model.model.initial_state

        tf.global_variables_initializer().run(session=sess)
        loaded_params = joblib.load(os.path.dirname(load_path) + "/ppo2_weights.pkl")
        restores = []
        for p, loaded_p in zip(find_trainable_variables("model"), loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)

        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        parser.add_argument('--policy', help='Policy architecture', default='feedforward',
                            choices=['feedforward', 'lstm', 'lnlstm'])

        return parser

    def getActionProba(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        return self.model.probaStep(observation, self.states, dones)

    def getAction(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        actions, _, self.states, _ = self.model.step(observation, self.states, dones)
        return actions

    def train(self, args, callback, env_kwargs=None):
        envs = self.makeEnv(args, env_kwargs=env_kwargs)

        # get the associated policy for the architecture requested
        if args.srl_model == "raw_pixels":
            if args.policy == "feedforward":
                args.policy = "cnn"
            else:
                args.policy = "cnn-" + args.policy
        else:
            if args.policy == "feedforward":
                args.policy = "mlp"

        self.ob_space = envs.observation_space
        self.ac_space = envs.action_space
        self.policy = args.policy
        self.continuous_actions = args.continuous_actions

        assert not (self.policy in ['lstm', 'lnlstm'] and args.num_cpu % 4 != 0), \
            "Error: Reccurent policies must have num cpu at a multiple of 4."

        logger.configure()
        self._learn(args, envs, nsteps=128, ent_coef=.01, lr=lambda f: f * 2.5e-4, total_timesteps=args.num_timesteps,
                    callback=callback)

    # Modified version of OpenAI to work with SRL models
    def learn(self, *, args, env, n_steps, total_timesteps, ent_coef, learning_rate, vf_coef=0.5, max_grad_norm=0.5,
              gamma=0.99, lam=0.95, log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, save_interval=0,
              load_path=None, callback=None):
        """
        Return a trained PPO2 model.

        :param args: (Arguments object)
        :param env: (Gym environment) The environment to learn from
        :param n_steps: (int) The number of steps to run for each environment
        :param total_timesteps: (int) The total number of samples
        :param ent_coef: (float) Entropy coefficient for the loss caculation
        :param learning_rate: (float or callable) The learning rate, it can be a function
        :param vf_coef: (float) Value function coefficient for the loss calculation
        :param max_grad_norm: (float) The maximum value for the gradient clipping
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param nminibatches: (int) Number of minibatches for the policies
        :param noptepochs: (int) Number of epoch when optimizing the surrogate
        :param cliprange: (float or callable) Clipping parameter, it can be a function
        :param log_interval: (int) The number of timesteps before logging.
        :param save_interval: (int) The number of timesteps before saving.
        :param load_path: (str) Path to a trained ppo2 model, set to None, it will learn from scratch
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        :return: (Model) PPO2 model
        """
        # MLP: multi layer perceptron
        # CNN: convolutional neural netwrok
        # LSTM: Long Short Term Memory
        # LNLSTM: Layer Normalization LSTM
        continuous = args.continuous_actions
        policy = {'cnn': PPO2CNNPolicy(continuous=continuous),
                  'cnn-lstm': PPO2CNNPolicy(continuous=continuous, reccurent=True),
                  'cnn-lnlstm': PPO2CNNPolicy(continuous=continuous, reccurent=True, normalised=True),
                  'mlp': PPO2MLPPolicy(continuous=continuous),
                  'lstm': PPO2MLPPolicy(continuous=continuous, reccurent=True),
                  'lnlstm': PPO2MLPPolicy(continuous=continuous, reccurent=True, normalised=True)}[args.policy]

        if policy is None:
            raise ValueError(args.policy + " not implemented for " + (
                "discrete" if args.continuous_actions else "continuous") + " action space.")

        if isinstance(learning_rate, float):
            learning_rate = constfn(learning_rate)
        else:
            assert callable(learning_rate)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)
        total_timesteps = int(total_timesteps)

        n_envs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        nbatch = n_envs * n_steps
        nbatch_train = nbatch // nminibatches

        make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=n_envs,
                                   nbatch_train=nbatch_train, nsteps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef,
                                   max_grad_norm=max_grad_norm)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(os.path.join(logger.get_dir(), 'make_model.pkl'), 'wb') as file_handler:
                file_handler.write(cloudpickle.dumps(make_model))
        self.model = make_model()
        self.states = self.model.initial_state
        if load_path is not None:
            self.model.load(load_path)
        runner = Runner(env=env, model=self.model, nsteps=n_steps, gamma=gamma, lam=lam)

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        nupdates = total_timesteps // nbatch
        for update in range(1, nupdates + 1):
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lr_now = learning_rate(frac)
            cliprangenow = cliprange(frac)
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
            epinfobuf.extend(epinfos)
            mblossvals = []
            if states is None:  # nonrecurrent version
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(self.model.train(lr_now, cliprangenow, *slices))
            else:  # recurrent version
                assert n_envs % nminibatches == 0
                envinds = np.arange(n_envs)
                flatinds = np.arange(n_envs * n_steps).reshape(n_envs, n_steps)
                envsperbatch = nbatch_train // n_steps
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, n_envs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(self.model.train(lr_now, cliprangenow, *slices, mbstates))

            lossvals = np.mean(mblossvals, axis=0)

            if callback is not None:
                callback(locals(), globals())

            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                explained_var = explained_variance(values, returns)
                logger.logkv("serial_timesteps", update * n_steps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update * nbatch)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(explained_var))
                logger.logkv('eprewmean', safe_mean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safe_mean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
                checkdir = os.path.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = os.path.join(checkdir, '%.5i' % update)
                print('Saving to', savepath)
                self.model.save(savepath)
        env.close()
        return self.model
