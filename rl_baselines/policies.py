import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines.a2c.utils import fc, sample
from baselines.common.distributions import make_pdtype
from baselines.ddpg.models import Model
from baselines.ppo2.policies import nature_cnn


class MlpPolicyDiscrete(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        """
        Modified version of OpenAI PPO2 MLP so it can support discrete actions
        :param sess: (tf Session)
        :param ob_space: (tuple)
        :param ac_space: (gym action space)
        :param nbatch: (int)
        :param nsteps: (int)
        :param reuse: (bool) for tensorflow
        """
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, pi_val = sess.run([a0, vf, neglogp0, pi], {X:ob})
            return a, v, self.initial_state, neglogp, pi_val

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


# Modified version of OpenAI to retrun PI
class MlpPolicyContinuous(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, pi_val = sess.run([a0, vf, neglogp0, pi], {X:ob})
            return a, v, self.initial_state, neglogp, pi_val

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CNNPolicyContinuous(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        """
        Modified version of OpenAI PPO2 CNN so it can support continuous actions
        :param sess: (tf Session)
        :param ob_space: (tuple)
        :param ac_space: (gym action space)
        :param nbatch: (int)
        :param nsteps: (int)
        :param reuse: (bool) for tensorflow
        """
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', actdim, init_scale=0.01)
            vf = fc(h, 'v', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, pi_val = sess.run([a0, vf, neglogp0, pi], {X:ob})
            return a, v, self.initial_state, neglogp, pi_val

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

# Modified version of OpenAI to retrun PI
class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, pi_val = sess.run([a0, vf, neglogp0, self.pi], {X:ob})
            return a, v, self.initial_state, neglogp, pi_val

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

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
