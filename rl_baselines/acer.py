from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy

import environments.kuka_button_gym_env as kuka_env
from rl_baselines.utils import createTensorflowSession, createEnvs
from rl_baselines.policies import AcerMlpPolicy
from rl_baselines.buffer_acer import Buffer


class Runner(object):
    def __init__(self, env, model, nsteps, nstack):
        self.env = env
        self.nstack = nstack
        self.model = model
        self.nenv = nenv = env.num_envs
        self.nact = env.action_space.n
        self.nbatch = nenv * nsteps

        if len(env.observation_space.shape) > 1:
            self.raw_pixels = True
            nh, nw, nc = env.observation_space.shape
            self.batch_ob_shape = (nenv * (nsteps + 1), nh, nw, nc * nstack)
            self.obs_dtype = np.uint8
            self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=self.obs_dtype)
            self.nc = nc
        else:
            self.raw_pixels = False
            obs_dim = env.observation_space.shape[0]
            self.batch_ob_shape = (nenv * (nsteps + 1), obs_dim * nstack)
            self.obs_dtype = np.float32
            self.obs = np.zeros((nenv, obs_dim * nstack), dtype=self.obs_dtype)
            self.obs_dim = obs_dim

        obs = env.reset()
        self.update_obs(obs)
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs, dones=None):
        if self.raw_pixels:
            if dones is not None:
                self.obs *= (1 - dones.astype(np.uint8))[:, None, None, None]
            self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
            self.obs[:, :, :, -self.nc:] = obs[:, :, :, :]
        else:
            if dones is not None:
                self.obs *= (1 - dones.astype(np.float32))[:, None]
            self.obs = np.roll(self.obs, shift=-self.obs_dim, axis=1)
            self.obs[:, -self.obs_dim:] = obs[:, :]

    def run(self):
        if self.raw_pixels:
            enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        else:
            enc_obs = np.split(self.obs, self.nstack, axis=1)  # so now list of obs steps
        mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        for _ in range(self.nsteps):
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

        # shapes are now [nenv, nsteps, []]
        # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks


def learn(policy, env, seed, nsteps=20, nstack=4, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=5000, replay_ratio=4, replay_start=1000, c=10.0,
          trust_region=True, alpha=0.99, delta=1, callback=None):
    tf.reset_default_graph()
    createTensorflowSession()

    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    elif policy == 'mlp':
        policy_fn = AcerMlpPolicy
    else:
        raise ValueError("Policy {} not implemented".format(policy))

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = nenvs
    model = Model(policy=policy_fn, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                  num_procs=num_procs, ent_coef=ent_coef, q_coef=q_coef, gamma=gamma,
                  max_grad_norm=max_grad_norm, lr=lr, rprop_alpha=rprop_alpha, rprop_epsilon=rprop_epsilon,
                  total_timesteps=total_timesteps, lrschedule=lrschedule, c=c,
                  trust_region=trust_region, alpha=alpha, delta=delta)

    runner = Runner(env=env, model=model, nsteps=nsteps, nstack=nstack)
    if replay_ratio > 0:
        _buffer = Buffer(env=env, nsteps=nsteps, nstack=nstack, size=buffer_size)
    else:
        _buffer = None
    nbatch = nenvs * nsteps
    acer = Acer(runner, model, _buffer, log_interval)
    acer.tstart = time.time()

    # nbatch samples, 1 on_policy call and multiple off-policy calls
    for acer.steps in range(0, total_timesteps, nbatch):
        acer.call(on_policy=True)
        if callback is not None:
            callback(locals(), globals())

        if replay_ratio > 0 and _buffer.has_atleast(replay_start):
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                acer.call(on_policy=False)  # no simulation steps in this


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'mlp'], default='cnn')
    parser.add_argument('--lr-schedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    return parser


def main(args, callback):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """
    envs = createEnvs(args)

    learn(args.policy, envs, total_timesteps=args.num_timesteps, seed=args.seed, nstack=args.num_stack,
          lrschedule=args.lr_schedule, callback=callback)
