import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from stable_baselines.a2c.utils import linear, sample, batch_to_seq, seq_to_batch, lstm
from stable_baselines.ddpg.models import Model
from stable_baselines.a2c.policies import nature_cnn, A2CPolicy
from stable_baselines.acer.policies import AcerPolicy


class LstmPolicy(A2CPolicy):
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps, nlstm=256, reuse=False, layer_norm=False, _type="cnn",
                 **kwargs):
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_batch, n_steps, nlstm, reuse)
        with tf.variable_scope("model", reuse=reuse):
            if _type == "cnn":
                extracted_features = nature_cnn(self.obs_ph, **kwargs)
            else:
                activ = tf.tanh
                extracted_features = tf.layers.flatten(self.obs_ph)
                extracted_features = activ(linear(extracted_features, 'pi_fc1', n_hidden=64, init_scale=np.sqrt(2)))
                extracted_features = activ(linear(extracted_features, 'pi_fc2', n_hidden=64, init_scale=np.sqrt(2)))
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=nlstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'v', 1)

            self.proba_distribution, self.policy = self.pdtype.proba_distribution_from_latent(rnn_output)

        self.value_0 = value_fn[:, 0]
        self.action_0 = self.proba_distribution.sample()
        self.neglogp0 = self.proba_distribution.neglogp(self.action_0)
        self.initial_state = np.zeros((self.n_env, nlstm * 2), dtype=np.float32)
        self.value_fn = value_fn

    def step(self, obs, state=None, mask=None):
        return self.sess.run([self.action_0, self.value_0, self.snew, self.neglogp0],
                             {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_0, {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})


class CnnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_batch, n_steps, n_lstm, reuse, layer_norm=False,
                                            _type="cnn")


class CnnLnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps, n_lstm=256, reuse=False, **_kwargs):
        super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_batch, n_steps, n_lstm, reuse,
                                              layer_norm=True,
                                              _type="cnn")


class MlpLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLstmPolicy, self).__init__(sess, ob_space, ac_space, n_batch, n_steps, n_lstm, reuse, layer_norm=False,
                                            _type="mlp")


class MlpLnLstmPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_batch, n_steps, n_lstm=256, reuse=False, **_kwargs):
        super(MlpLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_batch, n_steps, n_lstm, reuse,
                                              layer_norm=True,
                                              _type="mlp")


class AcerMlpPolicy(object):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse=False):
        """
        :param sess: (tf Session)
        :param ob_space: (tuple)
        :param ac_space: (gym action space)
        :param n_steps: (int)
        :param n_stack: (int)
        :param reuse: (bool) for tensorflow
        """
        # MLP not supported yet
        #super(AcerMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_stack, reuse)
        self.n_batch = n_env * n_steps
        self.n_act = ac_space.n
        obs_dim = ob_space.shape[0]
        self.ob_shape = (self.n_batch, obs_dim * n_stack)
        self.obs_ph = tf.placeholder(tf.float32, self.ob_shape)  # obs
        self.sess = sess
        self.reuse = reuse

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(linear(self.obs_ph, 'pi_fc1', n_hidden=64, init_scale=np.sqrt(2)))
            h2 = activ(linear(h1, 'pi_fc2', n_hidden=64, init_scale=np.sqrt(2)))
            pi_logits = linear(h2, 'pi', self.n_act, init_scale=0.01)
            policy = tf.nn.softmax(pi_logits)
            h1 = activ(linear(self.obs_ph, 'q_fc1', n_hidden=64, init_scale=np.sqrt(2)))
            h2 = activ(linear(h1, 'q_fc2', n_hidden=64, init_scale=np.sqrt(2)))
            q_value = linear(h2, 'q', self.n_act)

        self.action = sample(pi_logits)  # could change this to use self.policy instead
        self.initial_state = []  # not stateful
        self.policy = policy  # actual policy params now
        self.q_value = q_value

    def step(self, obs, state=None, mask=None):
        # returns actions, mus, states
        action_0, policy_0 = self.sess.run([self.action, self.policy], {self.obs_ph: obs})
        return action_0, policy_0, []  # dummy state

    def out(self, obs, state=None, mask=None):
        policy_0, q_value_0 = self.sess.run([self.policy, self.q_value], {self.obs_ph: obs})
        return policy_0, q_value_0

    def act(self, obs, state=None, mask=None):
        return self.sess.run(self.action, {self.obs_ph: obs})


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
