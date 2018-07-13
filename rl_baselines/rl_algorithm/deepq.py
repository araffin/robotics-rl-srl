from baselines import deepq
from baselines import logger
import tensorflow as tf

from rl_baselines.base_classes import BaseRLObject
from environments.utils import makeEnv
from rl_baselines.utils import createTensorflowSession, CustomVecNormalize, CustomDummyVecEnv, \
    WrapFrameStack, loadRunningAverage, MultiprocessSRLModel, softmax


class DeepQModel(BaseRLObject):
    """
    object containing the interface between baselines.deepq and this code base
    DeepQ: https://arxiv.org/pdf/1312.5602v1.pdf
    """
    def __init__(self):
        super(DeepQModel, self).__init__()
        self.model = None

    def save(self, save_path, _locals=None):
        assert self.model is not None or locals is not None, "Error: must train or load model before use"
        if self.model is None:
            self.model = _locals["act"]
        self.model.save(save_path)

    @classmethod
    def load(cls, load_path, args=None):
        loaded_model = DeepQModel()
        loaded_model.model = deepq.load(load_path)
        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--prioritized', type=int, default=1)
        parser.add_argument('--dueling', type=int, default=1)
        parser.add_argument('--buffer-size', type=int, default=int(1e3), help="Replay buffer size")
        return parser

    def getActionProba(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        # Get the tensor just before the softmax function in the TensorFlow graph,
        # then execute the graph from the input observation to this tensor.
        tensor = tf.get_default_graph().get_tensor_by_name('deepq/q_func/fully_connected_2/BiasAdd:0')
        sess = tf.get_default_session()
        return softmax(sess.run(tensor, feed_dict={'deepq/observation:0': observation}))

    def getAction(self, observation, dones=None):
        assert self.model is not None, "Error: must train or load model before use"
        return self.model(observation)[0]

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though DeepQ is single core only, we need to use the pipe system to work
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(1, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        env = CustomDummyVecEnv([makeEnv(args.env, args.seed, 0, args.log_dir, env_kwargs=env_kwargs)])

        if args.srl_model != "raw_pixels":
            env = CustomVecNormalize(env)
            env = loadRunningAverage(env, load_path_normalise=load_path_normalise)

        # Normalize only raw pixels
        # WARNING: when using framestacking, the memory used by the replay buffer can grow quickly
        return WrapFrameStack(env, args.num_stack, normalize=args.srl_model == "raw_pixels")

    @classmethod
    def getOptParam(cls):
        return {
            "lr": (float, (0, 0.1)),
            "exploration_fraction": (float, (0, 1)),
            "exploration_final_eps": (float, (0, 1)),
            "train_freq": (int, (1, 10)),
            "learning_starts": (int, (10, 10000)),
            "target_network_update_freq": (int, (10, 10000)),
            "gamma": (float, (0, 1)),
            "batch_size": (int, (8, 128)),
        }

    def train(self, args, callback, env_kwargs=None, hyperparam=None):
        logger.configure()

        env = self.makeEnv(args, env_kwargs)
        if hyperparam is None:
            hyperparam = {}

        createTensorflowSession()

        if args.srl_model != "raw_pixels":
            model = deepq.models.mlp([64, 64])
        else:
            # Atari CNN
            model = deepq.models.cnn_to_mlp(
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                hiddens=[256],
                dueling=bool(args.dueling),
            )

        deepq_param = {
            "lr": 1e-4,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
            "train_freq": 4,
            "learning_starts": 500,
            "target_network_update_freq": 500,
            "gamma": 0.99,
            "batch_size": 32,
            **hyperparam
        }

        self.model = deepq.learn(
            env,
            q_func=model,
            max_timesteps=args.num_timesteps,
            buffer_size=args.buffer_size,
            prioritized_replay=bool(args.prioritized),
            print_freq=10,  # Print every 10 episodes
            callback=callback,
            **deepq_param
        )
        self.model.save(args.log_dir + "deepq_model_end.pkl")
        env.close()
