import gym
from gym.utils import seeding


class SRLGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, *, srl_model, relative_pos, env_rank, srl_pipe):
        """
        Gym wrapper for SRL environments

        :param srl_model: (str) The SRL_model used
        :param relative_pos: (bool) position for ground truth
        :param env_rank: (int) the number ID of the environment
        :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
        """
        # the * here, means that the rest of the args need to be called as kwargs.
        # This is done to avoid unwanted situations where we might add a parameter
        #  later and not realise that srl_pipe was not set by an unchanged subclass.
        self.env_rank = env_rank
        self.srl_pipe = srl_pipe
        self.srl_model = srl_model
        self.relative_pos = relative_pos
        self.np_random = None

        # Create numpy random generator
        # This seed can be changed later
        self.seed(0)

    def getSRLState(self, observation):
        """
        get the SRL state for this environement with a given observation
        :param observation: (numpy float) image
        :return: (numpy float)
        """
        if self.srl_model == "ground_truth":
            if self.relative_pos:
                return self.getGroundTruth() - self.getTargetPos()
            return self.getGroundTruth()
        else:
            # srl_pipe is a tuple that contains:
            #  Queue: input to the SRL model, sends origin (where does the message comes from, here the rank of the environment)
            #  and observation that needs to be transformed into a state
            #  [Queue]: input for all the envs, sends state associated to the observation
            self.srl_pipe[0].put((self.env_rank, observation))
            return self.srl_pipe[1][self.env_rank].get()

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        raise NotImplementedError()

    @staticmethod
    def getGroundTruthDim():
        """
        :return: (int)
        """
        raise NotImplementedError()

    def getGroundTruth(self):
        """
        Alias for getArmPos for compatibility between envs
        :return: (numpy array)
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """
        Seed random generator
        :param seed: (int)
        :return: ([int])
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        # TODO: implement close function to close GUI
        pass

    def step(self, action, generated_observation=None, action_proba=None, action_grid_walker=None):
        """
        :param action: (int or [float])
        """
        raise NotImplementedError()

    def reset(self, generated_observation=None, state_override=None):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        raise NotImplementedError()

    def render(self, mode='human'):
        """
        :param mode: (str)
        :return: (numpy array)
        """
        raise NotImplementedError()
