import pickle
import os
import time

import numpy as np
import joblib
from baselines.common import tf_util
from baselines.acer.acer_simple import find_trainable_variables
from baselines.ppo2.ppo2 import Model, constfn, Runner, deque, explained_variance, safemean, osp
from baselines import logger
import tensorflow as tf

from rl_baselines.base_classes import BaseRLObject
from rl_baselines.policies import MlpPolicyDiscrete, CNNPolicyContinuous, MlpPolicyContinuous, CnnPolicy, LstmPolicy, \
    LnLstmPolicy


class PPO2Model(BaseRLObject):
    """
    object containing the interface between baselines.ppo2 and this code base
    PPO2: Proximal Policy Optimization (GPU Implementation)
    """
    def __init__(self):
        super(PPO2Model, self).__init__()
        self.ob_space = None
        self.ac_space = None
        self.policy = None
        self.model = None
        self.continuous_actions = None

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

        if loaded_model.continuous_actions:
            policy = {'cnn': CNNPolicyContinuous,
                      'lstm': None,
                      'lnlstm': None,
                      'mlp': MlpPolicyContinuous}[loaded_model.policy]
        else:
            # LN-LSTM: Layer Normalization LSTM
            policy = {'cnn': CnnPolicy,
                      'lstm': LstmPolicy,
                      'lnlstm': LnLstmPolicy,
                      'mlp': MlpPolicyDiscrete}[loaded_model.policy]

        if policy is None:
            raise ValueError(loaded_model.policy + " not implemented for " + (
                "discrete" if loaded_model.continuous_actions else "continuous") + " action space.")

        loaded_model.model = policy(sess, loaded_model.ob_space, loaded_model.ac_space, args.num_cpu, nsteps=1,
                                    reuse=False)

        tf.global_variables_initializer().run(session=sess)
        loaded_params = joblib.load(os.path.dirname(load_path) + "/ppo2_weights.pkl")
        restores = []
        for p, loaded_p in zip(find_trainable_variables("model"), loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)

        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'],
                            default='cnn')

        return parser

    def getActionProba(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        return self.model.probaStep(observation, None, dones)

    def getAction(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        actions, _, _, _ = self.model.step(observation, None, dones)
        return actions

    def train(self, args, callback, env_kwargs=None):
        envs = self.makeEnv(args, env_kwargs=env_kwargs)

        self.ob_space = envs.observation_space
        self.ac_space = envs.action_space
        self.policy = args.policy
        self.continuous_actions = args.continuous_actions

        logger.configure()
        self._learn(args, envs, nsteps=128, ent_coef=.01, lr=lambda f: f * 2.5e-4, total_timesteps=args.num_timesteps,
                    callback=callback)

    # Modified version of OpenAI to work with SRL models
    def _learn(self, args, env, nsteps, total_timesteps, ent_coef, lr, vf_coef=0.5, max_grad_norm=0.5, gamma=0.99,
               lam=0.95, log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, save_interval=0, callback=None):
        """
        :param args: (Arguments object)
        :param env: (gym VecEnv)
        :param nsteps: (int)
        :param total_timesteps: (int)
        :param ent_coef: (float) entropy coefficient
        :param lr: (float or function) learning rate
        :param vf_coef: (float)
        :param gamma: (float) discount factor
        :param lam: (float) lambda ?
        :param log_interval: (int)
        :param nminibatches: (int)
        :param noptepochs: (int)
        :param cliprange: (float or function)
        :param save_interval: (int)
        :param max_grad_norm: (float)
        :param callback: (function)
        """
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=args.num_cpu,
                                inter_op_parallelism_threads=args.num_cpu)
        config.gpu_options.allow_growth = True
        tf.Session(config=config).__enter__()

        if args.continuous_actions:
            policy = {'cnn': CNNPolicyContinuous,
                      'lstm': None,
                      'lnlstm': None,
                      'mlp': MlpPolicyContinuous}[args.policy]
        else:
            # LN-LSTM: Layer Normalization LSTM
            policy = {'cnn': CnnPolicy,
                      'lstm': LstmPolicy,
                      'lnlstm': LnLstmPolicy,
                      'mlp': MlpPolicyDiscrete}[args.policy]

        if policy is None:
            raise ValueError(args.policy + " not implemented for " + (
                "discrete" if args.continuous_actions else "continuous") + " action space.")

        if isinstance(lr, float):
            lr = constfn(lr)
        else:
            assert callable(lr)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)
        total_timesteps = int(total_timesteps)

        nenvs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        nbatch = nenvs * nsteps
        nbatch_train = nbatch // nminibatches

        make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                   nbatch_train=nbatch_train,
                                   nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                   max_grad_norm=max_grad_norm)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_model))
        self.model = make_model()
        runner = Runner(env=env, model=self.model, nsteps=nsteps, gamma=gamma, lam=lam)

        epinfobuf = deque(maxlen=100)
        tfirststart = time.time()

        nupdates = total_timesteps // nbatch
        for update in range(1, nupdates + 1):
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
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
                        mblossvals.append(self.model.train(lrnow, cliprangenow, *slices))
            else:  # recurrent version
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                envsperbatch = nbatch_train // nsteps
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(self.model.train(lrnow, cliprangenow, *slices, mbstates))

            lossvals = np.mean(mblossvals, axis=0)

            if callback is not None:
                callback(locals(), globals())

            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, returns)
                logger.logkv("serial_timesteps", update * nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update * nbatch)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, self.model.loss_names):
                    logger.logkv(lossname, lossval)
                logger.dumpkvs()
        env.close()
