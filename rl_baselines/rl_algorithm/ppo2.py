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

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}
        train_kwargs["lr_schedule"] = args.lr_schedule

        assert not (self.policy in ['lstm', 'lnlstm'] and args.num_cpu % 4 != 0), \
            "Error: Reccurent policies must have num cpu at a multiple of 4."

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
