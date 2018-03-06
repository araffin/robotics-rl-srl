from baselines import logger
from baselines.a2c.a2c import *
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env
from rl_baselines.policies import MlpPolicyDicrete
from rl_baselines.utils import createTensorflowSession
from srl_priors.utils import printYellow


# Redefine runner to add support for srl models
class Runner(object):
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nenv = env.num_envs
        if len(env.observation_space.shape) > 1:
            nh, nw, nc = env.observation_space.shape
            self.batch_ob_shape = (nenv * nsteps, nh, nw, nc)
            self.dtype = np.uint8
            self.obs = np.zeros((nenv, nh, nw, nc), dtype=self.dtype)
            self.nc = nc
        else:
            obs_dim = env.observation_space.shape[0]
            self.batch_ob_shape = (nenv * nsteps, obs_dim)
            self.dtype = np.float32
            self.obs = np.zeros((nenv, obs_dim), dtype=self.dtype)

        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for k, done in enumerate(dones):
                if done:
                    self.obs[k] = self.obs[k] * 0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def learn(policy, env, seed=0, nsteps=5, total_timesteps=int(1e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, callback=None):
    tf.reset_default_graph()
    createTensorflowSession()

    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'mlp':
        policy_fn = MlpPolicyDicrete
    else:
        raise ValueError("Policy {} not implemented".format(policy))

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef,
                  vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)

        if callback is not None:
            callback(locals(), globals())

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    return parser


def main(args, callback):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """

    if args.srl_model != "":
        printYellow("Using MLP policy because working on state representation")
        args.policy = "mlp"

    envs = [make_env(args.env, args.seed, i, args.log_dir, pytorch=False)
            for i in range(args.num_cpu)]

    envs = SubprocVecEnv(envs)
    # if args.srl_model == "ground_truth":
    #     # TODO: save running average
    #     envs = VecNormalize(envs)

    envs = VecFrameStack(envs, args.num_stack)
    logger.configure()
    learn(args.policy, envs, total_timesteps=args.num_timesteps, seed=args.seed,
          lrschedule=args.lrschedule, callback=callback)
    envs.close()
