import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from stable_baselines.a2c.utils import linear, sample, batch_to_seq, seq_to_batch, lstm
from stable_baselines.ddpg.models import Model
from stable_baselines.a2c.policies import nature_cnn
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common.input import observation_input


def PPO2MLPPolicy(continuous=False, reccurent=False, normalised=False, nlstm=64):
    """
    Generates an MLP policy for PPO2 and A2C
    :param continuous: (bool) If the output of the policy is continuous actions
    :param reccurent: (bool) If the policy uses a reccurent neural network
    :param normalised: (bool) If the policy uses layer normalisation for a reccurent policy
    :param nlstm: (int) Number of lstm cells to use
    :return: (Policy)
    """
    super_policy = FeedForwardPolicy if not reccurent else LstmPolicy

    class MlpPolicy(super_policy):
        def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **_kwargs):
            super(MlpPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, _type="mlp",
                                            layer_norm=normalised, continuous=continuous)

    return MlpPolicy


def PPO2CNNPolicy(continuous=False, reccurent=False, normalised=False, nlstm=64):
    """
    Generates an CNN policy for PPO2 and A2C
    :param continuous: (bool) If the output of the policy is continuous actions
    :param reccurent: (bool) If the policy uses a reccurent neural network
    :param normalised: (bool) If the policy uses layer normalisation for a reccurent policy
    :param nlstm: (int) Number of lstm cells to use
    :return: (Policy)
    """
    super_policy = FeedForwardPolicy if not reccurent else LstmPolicy

    class CnnPolicy(super_policy):
        def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **_kwargs):
            super(CnnPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, _type="cnn",
                                            layer_norm=normalised, continuous=continuous)

    return CnnPolicy


class A2CPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, continuous=False):
        """
        Policy object for A2C

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param nbatch: (int) The number of batch to run (nenvs * nsteps)
        :param nsteps: (int) The number of steps to run for each environment
        :param nlstm: (int) The number of LSTM cells (for reccurent policies)
        :param reuse: (bool) If the policy is reusable or not
        :param continuous: (bool) enable continuous action
        """
        if continuous:
            self.actdim = ac_space.shape[0]
        else:
            self.actdim = ac_space.n
        self.nenv = nbatch // nsteps
        self.obs_ph, self.processed_x = observation_input(ob_space, nbatch)
        self.masks_ph = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        self.states_ph = tf.placeholder(tf.float32, [self.nenv, nlstm * 2])  # states
        self.pdtype = make_proba_dist_type(self.actdim)
        self.sess = sess
        self.reuse = reuse

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp0
        """
        raise NotImplementedError

    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in reccurent policies)
        :param mask: ([float]) The last masks (used in reccurent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class FeedForwardPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, _type="cnn", continuous=False,
                 **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse,
                                                continuous=continuous)
        with tf.variable_scope("model", reuse=reuse):
            if _type == "cnn":
                extracted_features = nature_cnn(self.processed_x, **kwargs)
                value_fn = linear(extracted_features, 'v', 1)[:, 0]
            else:
                activ = tf.tanh
                processed_x = tf.layers.flatten(self.processed_x)
                pi_h1 = activ(linear(processed_x, 'pi_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                pi_h2 = activ(linear(pi_h1, 'pi_fc2', n_hidden=64, init_scale=np.sqrt(2)))
                vf_h1 = activ(linear(processed_x, 'vf_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                vf_h2 = activ(linear(vf_h1, 'vf_fc2', n_hidden=64, init_scale=np.sqrt(2)))
                value_fn = linear(vf_h2, 'vf', 1)[:, 0]
                extracted_features = pi_h2

            if continuous:
                logstd = tf.get_variable(name="logstd", shape=[1, self.actdim], initializer=tf.zeros_initializer())
                extracted_features = tf.concat([extracted_features, extracted_features * 0.0 + logstd], axis=1)

            self.proba_distribution, self.policy = self.pdtype.proba_distribution_from_latent(extracted_features,
                                                                                              init_scale=0.01)
        self.action_0 = self.proba_distribution.sample()
        self.neglogp0 = self.proba_distribution.neglogp(self.action_0)
        self.initial_state = None
        self.value_fn = value_fn

    def step(self, obs, state=None, mask=None):
        action, value, neglogp = self.sess.run([self.action_0, self.value_fn, self.neglogp0], {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_fn, {self.obs_ph: obs})


class LstmPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, layer_norm=False, _type="cnn",
                 continuous=False, **kwargs):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, continuous=continuous)
        with tf.variable_scope("model", reuse=reuse):
            if _type == "cnn":
                extracted_features = nature_cnn(self.obs_ph, **kwargs)
            else:
                activ = tf.tanh
                extracted_features = tf.layers.flatten(self.obs_ph)
                extracted_features = activ(linear(extracted_features, 'pi_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                extracted_features = activ(linear(extracted_features, 'pi_fc2', n_hidden=64, init_scale=np.sqrt(2)))
            input_sequence = batch_to_seq(extracted_features, self.nenv, nsteps)
            masks = batch_to_seq(self.masks_ph, self.nenv, nsteps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=nlstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'v', 1)

            if continuous:
                logstd = tf.get_variable(name="logstd", shape=[1, self.actdim], initializer=tf.zeros_initializer())
                rnn_output = tf.concat([rnn_output, rnn_output * 0.0 + logstd], axis=1)

            self.proba_distribution, self.policy = self.pdtype.proba_distribution_from_latent(rnn_output)

        self.value_0 = value_fn[:, 0]
        self.action_0 = self.proba_distribution.sample()
        self.neglogp0 = self.proba_distribution.neglogp(self.action_0)
        self.initial_state = np.zeros((self.nenv, nlstm * 2), dtype=np.float32)
        self.value_fn = value_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run([self.action_0, self.value_0, self.snew, self.neglogp0],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_0, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class LnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, continuous=False, **_):
        super(LnLstmPolicy, self).__init__(sess, ob_space, ac_space, nbatch, nsteps, nlstm, reuse, layer_norm=True,
                                           continuous=continuous)


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
            h1 = activ(linear(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(linear(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi_logits = linear(h2, 'pi', nact, init_scale=0.01)
            pi = tf.nn.softmax(pi_logits)
            h1 = activ(linear(X, 'q_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(linear(h1, 'q_fc2', nh=64, init_scale=np.sqrt(2)))
            q = linear(h2, 'q', nact)

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
