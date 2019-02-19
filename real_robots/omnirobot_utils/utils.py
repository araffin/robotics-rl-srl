from __future__ import division, print_function, absolute_import
from gym import spaces
from gym import logger
import gym
import numpy as np
import cv2

class PosTransformer(object):
    def __init__(self, camera_mat: np.ndarray, dist_coeffs: np.ndarray,
                 pos_camera_coord_ground: np.ndarray, rot_mat_camera_coord_ground: np.ndarray):
        """
        Transform the position among physical position in camera coordinate,
                                     physical position in ground coordinate,
                                     pixel position of image
        """
        super(PosTransformer, self).__init__()
        self.camera_mat = camera_mat

        self.dist_coeffs = dist_coeffs

        self.camera_2_ground_trans = np.zeros((4, 4), np.float)
        self.camera_2_ground_trans[0:3, 0:3] = rot_mat_camera_coord_ground
        self.camera_2_ground_trans[0:3, 3] = pos_camera_coord_ground
        self.camera_2_ground_trans[3, 3] = 1.0

        self.ground_2_camera_trans = np.linalg.inv(self.camera_2_ground_trans)

    def phyPosCam2PhyPosGround(self, pos_coord_cam):
        """
        Transform physical position in camera coordinate to physical position in ground coordinate
        """
        assert pos_coord_cam.shape == (3, 1)
        homo_pos = np.ones((4, 1), np.float32)
        homo_pos[0:3, :] = pos_coord_cam
        return (np.matmul(self.camera_2_ground_trans, homo_pos))[0:3, :]

    def phyPosGround2PixelPos(self, pos_coord_ground, return_distort_image_pos=False):
        """
        Transform the physical position in ground coordinate to pixel position
        """
        pos_coord_ground = np.array(pos_coord_ground)
        if len(pos_coord_ground.shape) == 1:
            pos_coord_ground = pos_coord_ground.reshape(-1,1)

        assert pos_coord_ground.shape == (
            3, 1) or pos_coord_ground.shape == (2, 1)

        homo_pos = np.ones((4, 1), np.float32)
        if pos_coord_ground.shape == (2, 1):
            # by default, z =0 since it's on the ground
            homo_pos[0:2, :] = pos_coord_ground
            
            # (np.random.randn() - 0.5) * 0.05 # add noise to the z-axis
            homo_pos[2, :] = 0
        else:
            homo_pos[0:3, :] = pos_coord_ground
        homo_pos = np.matmul(self.ground_2_camera_trans, homo_pos)
        
        pixel_points, _ = cv2.projectPoints(homo_pos[0:3, :].reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                            self.camera_mat, self.dist_coeffs if return_distort_image_pos else None)
        return pixel_points.reshape((2, 1))

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
            assert np.isscalar(positive_low) and np.isscalar(
                positive_high) and np.isscalar(negative_low) and np.isscalar(negative_high)
            positive_low = positive_low + np.zeros(shape)
            positive_high = positive_high + np.zeros(shape)
            negative_low = negative_low + np.zeros(shape)
            negative_high = negative_high + np.zeros(shape)
        if dtype is None:  # Autodetect type
            if (positive_high == 255).all():
                dtype = np.uint8
            else:
                dtype = np.float32
            logger.warn(
                "Ring Box autodetected dtype as {}. Please provide explicit dtype.".format(dtype))
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
        length_positive = self.length_positive if self.dtype.kind == 'f' else self.length_positive.astype(
            'int64') + 1
        origin_sample = self.np_random.uniform(
            low=-self.length_negative, high=length_positive, size=self.negative_high.shape).astype(self.dtype)
        origin_sample = origin_sample + self.positive_low * \
            (origin_sample >= 0) + self.negative_high * (origin_sample <= 0)
        return origin_sample

    def contains(self, x):
        return x.shape == self.shape and np.logical_or(np.logical_and(x >= self.positive_low, x <= self.positive_high),
                                                       np.logical_and(x <= self.negative_high,  x >= self.negative_low)).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "RingBox" + str(self.shape)

    def __eq__(self, other):
        return np.allclose(self.positive_low, other.positive_low) and np.allclose(self.positive_high, other.positive_high) \
            and np.allclose(self.negative_low, other.negative_low) and np.allclose(self.negative_high, other.negative_high)
