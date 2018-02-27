from baselines import logger
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import *
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env


def learn(policy, env, seed=0, nsteps=5, total_timesteps=int(1e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr=7e-4,
          lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, callback=None):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'mlp':
        policy_fn = MlpPolicy
    else:
        raise ValueError("Policy {} not implemented".format(policy))

    tf.reset_default_graph()
    set_global_seeds(seed)

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
    parser.add_argument('--num_cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    return parser


def main(args, callback):
    envs = [make_env(args.env, 0, i, args.log_dir, pytorch=False)
            for i in range(args.num_cpu)]

    envs = SubprocVecEnv(envs)
    envs = VecFrameStack(envs, args.num_stack)
    logger.configure()
    learn(args.policy, envs, total_timesteps=args.num_timesteps, seed=args.seed,
          lrschedule=args.lrschedule, callback=callback)
    envs.close()
