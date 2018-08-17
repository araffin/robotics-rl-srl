from stable_baselines.acer import ACER

from rl_baselines.base_classes import StableBaselinesRLObject


class ACERModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.acer and this code base
    ACER: Sample Efficient Actor-Critic with Experience Replay
    """

    LOG_INTERVAL = 1  # log RL model performance every 1 steps
    SAVE_INTERVAL = 20  # Save RL model every 20 steps

    def __init__(self):
        super(ACERModel, self).__init__(name="acer", model_class=ACER)

    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        parser.add_argument('--lr-schedule', help='Learning rate schedule', choices=['constant', 'linear'],
                            default='constant')
        return parser

    @classmethod
    def getOptParam(cls):
        return {
            "n_steps": (int, (1, 100)),
            "q_coef": (float, (0, 1)),
            "ent_coef": (float, (0, 1)),
            "max_grad_norm": (float, (0.1, 5)),
            "learning_rate": (float, (0, 0.1)),
            "rprop_epsilon": (float, (0, 0.01)),
            "rprop_alpha": (float, (0.5, 1)),
            "gamma": (float, (0.5, 1)),
            "alpha": (float, (0.5, 1)),
            "replay_ratio": (int, (0, 10)),
            "correction_term": (float, (1, 10)),
            "delta": (float, (0.1, 10)),
            "lr_schedule": ((list, str),
                            ['linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop'])
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}

        assert args.num_stack > 1, "ACER only works with '--num-stack' of 2 or more"

        param_kwargs = {
            "verbose": 1,
            "n_steps": 20,
            "n_stack": 1,
            "q_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 10,
            "learning_rate": 7e-4,
            "rprop_epsilon": 1e-5,
            "rprop_alpha": 0.99,
            "gamma": 0.99,
            "buffer_size": 5000,
            "replay_ratio": 4,
            "replay_start": 1000,
            "correction_term": 10.0,
            "trust_region": True,
            "alpha": 0.99,
            "delta": 1,
            "lr_schedule": args.lr_schedule
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
