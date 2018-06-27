import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines.a2c.utils import fc, sample, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.ddpg.models import Model
from baselines.ppo2.policies import nature_cnn


def PPO2MLPPolicy(continuous=False, reccurent=False, normalised=False, nlstm=64):
    class Policy(object):
        def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
            """
            Modified version of OpenAI PPO2 policies, to support continous actions and returning pi.
            :param sess: (tf Session)
            :param ob_space: (tuple)
            :param ac_space: (gym action space)
            :param nbatch: (int)
            :param nsteps: (int)
            :param reuse: (bool) for tensorflow
            """
            assert reccurent or not normalised, "Must be reccurent policy to be normalised."

            nenv = nbatch // nsteps
            if continuous:
                actdim = ac_space.shape[0]
            else:
                actdim = ac_space.n
            ob_shape = (nbatch,) + ob_space.shape
            X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
            M = None
            S = None
            if reccurent:
                M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
                S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states

            # Layers
            with tf.variable_scope("model", reuse=reuse):
                activ = tf.tanh
                # input layers
                decoder = X
                if reccurent:
                    h1 = activ(fc(X, 'lstm_fc1', nh=64, init_scale=np.sqrt(2)))
                    decoder = activ(fc(h1, 'lstm_fc2', nh=64, init_scale=np.sqrt(2)))

                # Reccurent layer
                if reccurent:
                    xs = batch_to_seq(decoder, nenv, nsteps)
                    ms = batch_to_seq(M, nenv, nsteps)
                    if normalised:
                        h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
                    else:
                        h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
                    decoder = seq_to_batch(h5)
                # output layer
                h_pi = activ(fc(decoder, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
                h_pi = activ(fc(h_pi, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
                h_vf = activ(fc(decoder, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
                h_vf = activ(fc(h_vf, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))

                pi = fc(h_pi, 'pi', actdim, init_scale=0.01)
                vf = fc(h_vf, 'vf', 1)[:, 0]
                if continuous:
                    logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

            # parameters
            self.pdtype = make_pdtype(ac_space)
            if continuous:
                pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                self.pd = self.pdtype.pdfromflat(pdparam)
            else:
                self.pd = self.pdtype.pdfromflat(pi)
            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None
            if reccurent:
                self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

            # functions
            def step(ob, state, mask):
                if reccurent:
                    return sess.run([a0, vf, snew, neglogp0], {X: ob, S: state, M: mask})
                else:
                    a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
                    return a, v, self.initial_state, neglogp

            def probaStep(ob, state, mask):
                if reccurent:
                    return sess.run(pi, {X: ob, S: state, M: mask})
                else:
                    return sess.run(pi, {X: ob})

            def value(ob, state, mask):
                if reccurent:
                    return sess.run(vf, {X: ob, S: state, M: mask})
                else:
                    return sess.run(vf, {X: ob})

            self.X = X
            self.M = M
            self.S = S
            self.pi = pi
            self.vf = vf
            self.step = step
            self.probaStep = probaStep
            self.value = value

    return Policy


def PPO2CNNPolicy(continuous=False, reccurent=False, normalised=False, nlstm=64):
    class Policy(object):
        def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
            """
            Modified version of OpenAI PPO2 policies, to support continous actions and returning pi.
            :param sess: (tf Session)
            :param ob_space: (tuple)
            :param ac_space: (gym action space)
            :param nbatch: (int)
            :param nsteps: (int)
            :param reuse: (bool) for tensorflow
            """
            assert reccurent or not normalised, "Must be reccurent policy to be normalised."

            nenv = nbatch // nsteps
            if continuous:
                actdim = ac_space.shape[0]
            else:
                actdim = ac_space.n
            nh, nw, nc = ob_space.shape
            ob_shape = (nbatch, nh, nw, nc)
            X = tf.placeholder(tf.uint8, ob_shape)
            M = None
            S = None
            if reccurent:
                M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
                S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states

            # Layers
            with tf.variable_scope("model", reuse=reuse):
                activ = tf.tanh
                # input layers
                decoder = nature_cnn(X)
                # Reccurent layer
                if reccurent:
                    xs = batch_to_seq(decoder, nenv, nsteps)
                    ms = batch_to_seq(M, nenv, nsteps)
                    if normalised:
                        h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
                    else:
                        h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
                    decoder = seq_to_batch(h5)
                # output layer
                pi = fc(decoder, 'pi', actdim, init_scale=0.01)
                vf = fc(decoder, 'vf', 1)[:, 0]
                if continuous:
                    logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

            # parameters
            self.pdtype = make_pdtype(ac_space)
            if continuous:
                pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                self.pd = self.pdtype.pdfromflat(pdparam)
            else:
                self.pd = self.pdtype.pdfromflat(pi)
            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None
            if reccurent:
                self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

            # functions
            def step(ob, state, mask):
                if reccurent:
                    return sess.run([a0, vf, snew, neglogp0], {X: ob, S: state, M: mask})
                else:
                    a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
                    return a, v, self.initial_state, neglogp

            def probaStep(ob, state, mask):
                if reccurent:
                    return sess.run(pi, {X: ob, S: state, M: mask})
                else:
                    return sess.run(pi, {X: ob})

            def value(ob, state, mask):
                if reccurent:
                    return sess.run(vf, {X: ob, S: state, M: mask})
                else:
                    return sess.run(vf, {X: ob})

            self.X = X
            self.M = M
            self.S = S
            self.pi = pi
            self.vf = vf
            self.step = step
            self.probaStep = probaStep
            self.value = value

    return Policy


class AcerMlpPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        """
        :param sess: (tf Session)
        :param ob_space: (tuple)
        :param ac_space: (gym action space)
        :param nsteps: (int)
        :param nstack: (int)
        :param reuse: (bool) for tensorflow
        """
        nbatch = nenv * nsteps
        obs_dim = ob_space.shape[0]
        ob_shape = (nbatch, obs_dim * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi_logits = fc(h2, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            h1 = activ(fc(X, 'q_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'q_fc2', nh=64, init_scale=np.sqrt(2)))
            q = fc(h2, 'q', nact)

        a = sample(pi_logits)  # could change this to use self.pi instead
        self.initial_state = []  # not stateful
        self.X = X
        self.pi = pi  # actual policy params now
        self.q = q

        def step(ob, *args, **kwargs):
            # returns actions, mus, states
            a0, pi0 = sess.run([a, pi], {X: ob})
            return a0, pi0, []  # dummy state

        def out(ob, *args, **kwargs):
            pi0, q0 = sess.run([pi, q], {X: ob})
            return pi0, q0

        def act(ob, *args, **kwargs):
            return sess.run(a, {X: ob})

        self.step = step
        self.out = out
        self.act = act


class DDPGActorCNN(Model):
    """
    Adapted from openAI baseline, used for DDPG
    """

    def __init__(self, n_actions, name='DDPGActorCNN', layer_norm=True):
        """
        :param n_actions: (int)
        :param name: (str)
        :param layer_norm: (bool)
        """
        super(DDPGActorCNN, self).__init__(name=name)
        self.n_actions = n_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        """
        :param obs: (Tensor)
        :param reuse: (bool)
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            x = tf.layers.conv2d(x, 32, (8, 8), (4, 4))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, 64, (4, 4), (2, 2))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, 64, (3, 3))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tc.layers.flatten(x)

            x = tf.layers.dense(x, 256)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 256)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.n_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)

        return x


class DDPGCriticCNN(Model):
    """
    Adapted from openAI baseline, used for DDPG
    """

    def __init__(self, name='DDPGCriticCNN', layer_norm=True):
        """
        :param name: (str)
        :param layer_norm: (bool)
        """
        super(DDPGCriticCNN, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        """
        :param obs: (Tensor)
        :param action: (Tensor)
        :param reuse: (bool)
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            x = tf.layers.conv2d(x, 32, (8, 8), (4, 4))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, 64, (4, 4), (2, 2))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, 64, (3, 3))
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tc.layers.flatten(x)

            x = tf.layers.dense(x, 256)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 256)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class DDPGActorMLP(Model):
    """
    Adapted from openAI baseline, used for DDPG
    """

    def __init__(self, n_actions, name='DDPGActorMLP', layer_norm=True):
        """
        :param n_actions: (int)
        :param name: (str)
        :param layer_norm: (bool)
        """
        super(DDPGActorMLP, self).__init__(name=name)
        self.n_actions = n_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        """
        :param obs: (Tensor)
        :param reuse: (bool)
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            x = tf.layers.dense(x, 400)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 300)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.n_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)

        return x


class DDPGCriticMLP(Model):
    """
    Adapted from openAI baseline, used for DDPG
    """

    def __init__(self, name='DDPGCriticMLP', layer_norm=True):
        """
        :param name: (str)
        :param layer_norm: (bool)
        """
        super(DDPGCriticMLP, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        """
        :param obs: (Tensor)
        :param action: (Tensor)
        :param reuse: (bool)
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            x = tf.layers.dense(x, 400)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 300)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
