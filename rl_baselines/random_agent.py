"""
Random agent: randomly sample actions from the action space
"""
import time

from rl_baselines.base_classes import BaseRLObject


class RandomAgentModel(BaseRLObject):
    def __init__(self):
        super(RandomAgentModel, self).__init__()

    def save(self, save_path, _locals=None):
        pass

    @classmethod
    def load(cls, load_path, args=None):
        raise ValueError("Error: loading a saved random agent is not useful")

    def customArguments(self, parser):
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        return parser

    def getAction(self, observation, dones=None):
        # Action space is not available here, so we do not support it.
        raise ValueError("Error: getAction is not supported for random agent.")

    def train(self, args, callback, env_kwargs=None, hyperparam=None):
        env = self.makeEnv(args, env_kwargs=env_kwargs)

        obs = env.reset()
        num_updates = int(args.num_timesteps) // args.num_cpu
        start_time = time.time()

        for step in range(num_updates):
            actions = [env.action_space.sample() for _ in range(args.num_cpu)]
            obs, reward, done, info = env.step(actions)
            if callback is not None:
                callback(locals(), globals())
            if (step + 1) % 500 == 0:
                total_steps = step * args.num_cpu
                print("{} steps - {:.2f} FPS".format(total_steps, total_steps / (time.time() - start_time)))
