import gym

class SRLGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    """
    Gym wrapper for SRL environments
    :param use_ground_truth: (bool) Set to true, the observation will be the ground truth (arm position)
    :param relative_pos: (bool) position for ground truth
    :param env_rank: (int) the number ID of the environment
    :param pipe: (tuple) contains the input and output of the SRL model
    """

    def __init__(self, *, use_ground_truth, relative_pos, env_rank, srl_pipe):
        # the * here, means that the rest of the args need to be called as kwargs.
        # This is done to avoid unwanted situations where we might add a parameter
        #  later and not realise that srl_pipe was not set by an unchanged subclass.
        self.env_rank = env_rank
        self.srl_pipe = srl_pipe
        self.use_ground_truth = use_ground_truth
        self.relative_pos = relative_pos


    def getSRLState(self, observation):
        """
        get the SRL state for this environement with a given observation
        :param observation: (numpy float) image
        :return: (numpy float)
        """
        if self.use_ground_truth:
            if self.relative_pos:
                return self.getGroundTruth() - self.getTargetPos()
            return self.getGroundTruth()
        else:
            # srl_pipe is a tuple that containes:
            #  Queue: input to the SRL model, send origin and observation
            #  [Queue]: input for all the envs, send observation
            self.srl_pipe[0].put((self.env_rank, observation))
            return self.srl_pipe[1][self.env_rank].get()

    def getTargetPos(self):
        """
        :return (numpy array): Position of the target (button)
        """
        # Return only the [x, y] coordinates
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
        # Return only the [x, y] coordinates
        raise NotImplementedError()