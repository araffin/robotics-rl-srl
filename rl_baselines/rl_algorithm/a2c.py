from stable_baselines import A2C

from rl_baselines.base_classes import StableBaselinesRLObject


class A2CModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.a2c and this code base
    A2C: A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C)
    """

    SAVE_INTERVAL = 10  # Save RL model every 10 steps

    def __init__(self):
        super(A2CModel, self).__init__(name="a2c", model_class=A2C)

    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        parser.add_argument('--lr-schedule', help='Learning rate schedule', default='constant',
                            choices=['linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop'])
        return parser

    @classmethod
    def getOptParam(cls):
        return {
            "n_steps": (int, (1, 100)),
            "vf_coef": (float, (0, 1)),
            "ent_coef": (float, (0, 1)),
            "max_grad_norm": (float, (0.1, 5)),
            "learning_rate": (float, (0, 0.1)),
            "epsilon": (float, (0, 0.01)),
            "alpha": (float, (0.5, 1)),
            "gamma": (float, (0.5, 1)),
            "lr_schedule": ((list, str),
                            ['linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop'])
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}

        param_kwargs = {
            "verbose": 1,
            "n_steps": 5,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "learning_rate": 7e-4,
            "epsilon": 1e-5,
            "alpha": 0.99,
            "gamma": 0.99,
            "lr_schedule": args.lr_schedule
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
