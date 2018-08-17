from stable_baselines.ppo2 import PPO2

from rl_baselines.base_classes import StableBaselinesRLObject


class PPO2Model(StableBaselinesRLObject):
    """
    object containing the interface between baselines.ppo2 and this code base
    PPO2: Proximal Policy Optimization (GPU Implementation)
    """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(PPO2Model, self).__init__(name="ppo2", model_class=PPO2)

    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)

        return parser

    @classmethod
    def getOptParam(cls):
        return {
            "lam": (float, (0, 1)),
            "gamma": (float, (0, 1)),
            "max_grad_norm": (float, (0, 1)),
            "vf_coef": (float, (0, 1)),
            "learning_rate": (float, (1e-2, 1e-5)),
            "ent_coef": (float, (0, 1)),
            "cliprange": (float, (0, 1)),
            "noptepochs": (int, (1, 10)),
            "n_steps": (int, (32, 2048))
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}

        assert not (self.policy in ['lstm', 'lnlstm', 'cnnlstm', 'cnnlnlstm'] and args.num_cpu % 4 != 0), \
            "Error: Reccurent policies must have num cpu at a multiple of 4."

        if "lstm" in args.policy:
            param_kwargs = {
                "verbose": 1,
                "n_steps": 609,
                "ent_coef": 0.06415865069774951,
                "learning_rate": 0.004923676735761618,
                "vf_coef": 0.056219345567007695,
                "max_grad_norm": 0.19232704980689763,
                "gamma": 0.9752388470759489,
                "lam": 0.3987544314875193,
                "nminibatches": 4,
                "noptepochs": 8
            }
        else:
            param_kwargs = {
                "verbose": 1,
                "n_steps": 128,
                "ent_coef": 0.01,
                "learning_rate": lambda f: f * 2.5e-4,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gamma": 0.99,
                "lam": 0.95,
                "nminibatches": 4,
                "noptepochs": 4,
                "cliprange": 0.2
            }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
