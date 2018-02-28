from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import environments.kuka_button_gym_env as kuka_env
from pytorch_agents.envs import make_env


# Original code from https://github.com/openai/baselines/blob/master/baselines/acer/acer_simple.py
# Fixes https://github.com/openai/baselines/issues/301
class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
                 ent_coef, q_coef, gamma, max_grad_norm, lr,
                 rprop_alpha, rprop_epsilon, total_timesteps, lrschedule,
                 c, trust_region, alpha, delta):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])  # actions
        D = tf.placeholder(tf.float32, [nbatch])  # dones
        R = tf.placeholder(tf.float32, [nbatch])  # rewards, not returns
        MU = tf.placeholder(tf.float32, [nbatch, nact])  # mu's
        LR = tf.placeholder(tf.float32, [])
        eps = 1e-6

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)

        params = find_trainable_variables("model")
        print("Params {}".format(len(params)))
        for var in params:
            print(var)

        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)
        ema_apply_op = ema.apply(params)

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            print(v.name)
            return v

        with tf.variable_scope("", custom_getter=custom_getter, reuse=True):
            polyak_model = policy(sess, ob_space, ac_space, nenvs, nsteps + 1, nstack, reuse=True)

        # Notation: (var) = batch variable, (var)s = seqeuence variable, (var)_i = variable index by action at step i
        v = tf.reduce_sum(train_model.pi * train_model.q, axis=-1)  # shape is [nenvs * (nsteps + 1)]

        # strip off last step
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [train_model.pi, polyak_model.pi, train_model.q])
        # Get pi and q values for actions taken
        f_i = get_by_index(f, A)
        q_i = get_by_index(q, A)

        # Compute ratios for importance truncation
        rho = f / (MU + eps)
        rho_i = get_by_index(rho, A)

        # Calculate Q_retrace targets
        qret = q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma)

        # Calculate losses
        # Entropy
        entropy = tf.reduce_mean(cat_entropy_softmax(f))

        # Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)
        check_shape([qret, v, rho_i, f_i], [[nenvs * nsteps]] * 4)
        check_shape([rho, f, q], [[nenvs * nsteps, nact]] * 2)

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps)  # / (f_old + eps)
        check_shape([adv_bc, logf_bc], [[nenvs * nsteps, nact]] * 2)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f),
                                axis=1)  # IMP: This is sum, as expectation wrt f
        loss_bc = -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        check_shape([qret, q_i], [[nenvs * nsteps]] * 2)
        ev = q_explained_variance(tf.reshape(q_i, [nenvs, nsteps]), tf.reshape(qret, [nenvs, nsteps]))
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i) * 0.5)

        # Net loss
        check_shape([loss_policy, loss_q, entropy], [[]] * 3)
        loss = loss_policy + q_coef * loss_q - ent_coef * entropy

        if trust_region:
            g = tf.gradients(- (loss_policy - ent_coef * entropy) * nsteps * nenvs, f)  # [nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps)  # [nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) / (
            tf.reduce_sum(tf.square(k), axis=-1) + eps))  # [nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = avg_norm(k)
            avg_norm_g = avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = g - tf.reshape(adj, [nenvs * nsteps, 1]) * k
            grads_f = -g / (
            nenvs * nsteps)  # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(loss_q * q_coef, params)
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]

            avg_norm_grads_f = avg_norm(grads_f) * (nsteps * nenvs)
            norm_grads_q = tf.global_norm(grads_q)
            norm_grads_policy = tf.global_norm(grads_policy)
        else:
            grads = tf.gradients(loss, params)

        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=rprop_alpha, epsilon=rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_opt_op]):
            _train = tf.group(ema_apply_op)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Ops/Summaries to run, and their names for logging
        run_ops = [_train, loss, loss_q, entropy, loss_policy, loss_f, loss_bc, ev, norm_grads]
        names_ops = ['loss', 'loss_q', 'entropy', 'loss_policy', 'loss_f', 'loss_bc', 'explained_variance',
                     'norm_grads']
        if trust_region:
            run_ops = run_ops + [norm_grads_q, norm_grads_policy, avg_norm_grads_f, avg_norm_k, avg_norm_g,
                                 avg_norm_k_dot_g,
                                 avg_norm_adj]
            names_ops = names_ops + ['norm_grads_q', 'norm_grads_policy', 'avg_norm_grads_f', 'avg_norm_k',
                                     'avg_norm_g',
                                     'avg_norm_k_dot_g', 'avg_norm_adj']

        def train(obs, actions, rewards, dones, mus, states, masks, steps):
            cur_lr = lr.value_steps(steps)
            td_map = {train_model.X: obs, polyak_model.X: obs, A: actions, R: rewards, D: dones, MU: mus, LR: cur_lr}
            if states:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
                td_map[polyak_model.S] = states
                td_map[polyak_model.M] = masks
            return names_ops, sess.run(run_ops, td_map)[1:]  # strip off _train

        def save(save_path):
            ps = sess.run(params)
            # make_path(save_path)
            joblib.dump(ps, save_path)

        self.train = train
        self.save = save
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)


# TODO: save the learned model
def learn(policy, env, seed, nsteps=20, nstack=4, total_timesteps=int(80e6), q_coef=0.5, ent_coef=0.01,
          max_grad_norm=10, lr=7e-4, lrschedule='linear', rprop_epsilon=1e-5, rprop_alpha=0.99, gamma=0.99,
          log_interval=100, buffer_size=5000, replay_ratio=4, replay_start=1000, c=10.0,
          trust_region=True, alpha=0.99, delta=1, callback=None):
    win, win_smooth, win_episodes = None, None, None
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)  # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
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


def train(envs, num_timesteps, seed, policy, lrschedule, callback=None, num_stack=1):
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        raise ValueError("Policy {} not implemented".format(policy))
    learn(policy_fn, envs, seed, total_timesteps=num_timesteps, lrschedule=lrschedule,
          callback=callback, nstack=num_stack)


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num_cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'],
                        default='constant')
    return parser


def main(args, callback):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """

    if args.srl_model != "":
        raise NotImplementedError("RL on SRL not supported for acer")

    envs = [make_env(args.env, 0, i, args.log_dir, pytorch=False)
            for i in range(args.num_cpu)]

    envs = SubprocVecEnv(envs)
    train(envs, num_timesteps=args.num_timesteps, seed=args.seed, num_stack=args.num_stack,
          policy=args.policy, lrschedule=args.lrschedule, callback=callback)
