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

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}
        train_kwargs["lr_schedule"] = args.lr_schedule

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
            "log_interval": 100,
            "buffer_size": 5000,
            "replay_ratio": 4,
            "replay_start": 1000,
            "correction_term": 10.0,
            "trust_region": True,
            "alpha": 0.99,
            "delta": 1
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
