from stable_baselines.acktr import ACKTR

from rl_baselines.base_classes import StableBaselinesRLObject


class ACKTRModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.acktr and this code base
    ACKTR: Actor Critic using Kronecker-Factored Trust Region
    """

    def __init__(self):
        super(ACKTRModel, self).__init__(name="acktr", model_class=ACKTR)

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
            "vf_fisher_coef": (float, (0, 1)),
            "gamma": (float, (0.5, 1)),
            "kfac_clip": (float, (0, 1)),
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
            "vf_fisher_coef": 1.0,
            "gamma": 0.99,
            "lr_schedule": args.lr_schedule
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
