from __future__ import division, print_function, absolute_import

from gym import spaces # these import cannot put at the top, because python2 load OmnirobotManagerBase too
from gym import logger
import gym
import numpy as np
class RingBox(gym.Space):
    """
    A ring box in R^n.
    I.e., each coordinate is bounded.
    there are minimum constrains (absolute) on all of the coordinates 
    """
    def __init__(self, positive_low=None, positive_high=None, negative_low=None, negative_high=None, shape=None, dtype=None):
        """
        for each coordinate
        the value will be sampled from [positive_low, positive_hight] or [negative_low, negative_high]        
        """



        if shape is None:
            assert positive_low.shape == positive_high.shape == negative_low.shape == negative_high.shape
            shape = positive_low.shape
        else:
            assert np.isscalar(positive_low) and np.isscalar(positive_high) and np.isscalar(negative_low) and np.isscalar(negative_high)
            positive_low = positive_low + np.zeros(shape)
            positive_high = positive_high + np.zeros(shape)
            negative_low = negative_low + np.zeros(shape)
            negative_high = negative_high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (positive_high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            logger.warn("Ring Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))
        self.positive_low = positive_low.astype(dtype)
        self.positive_high = positive_high.astype(dtype)
        self.negative_low = negative_low.astype(dtype)
        self.negative_high = negative_high.astype(dtype)
        self.length_positive = self.positive_high - self.positive_low 
        self.length_negative = self.negative_high - self.negative_low
        super(RingBox, self).__init__(shape, dtype)
        self.np_random = np.random.RandomState()
    def seed(self, seed):
        self.np_random.seed(seed)

    def sample(self):
        length_positive = self.length_positive if self.dtype.kind == 'f' else self.length_positive.astype('int64') + 1
        origin_sample = self.np_random.uniform(low=-self.length_negative, high=length_positive, size=self.negative_high.shape).astype(self.dtype)
        origin_sample = origin_sample + self.positive_low * (origin_sample >= 0) + self.negative_high * (origin_sample <= 0)
        return origin_sample

    def contains(self, x):
        return x.shape == self.shape and np.logical_or(np.logical_and(x >= self.positive_low, x <= self.positive_high),
                np.logical_and(x <= self.negative_high,  x >= self.negative_low)).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.positive_low, other.positive_low) and np.allclose(self.positive_high, other.positive_high) \
            and np.allclose(self.negative_low, other.negative_low) and np.allclose(self.negative_high, other.negative_high)
