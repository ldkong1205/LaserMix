import abc
import numpy as np
from pyquaternion import Quaternion
from data.nuscenes.utils import DataSample


class LidarAugmentInterface(abc.ABC):

    @abc.abstractmethod
    def augment(self, data_sample: DataSample) -> DataSample:
        pass


class NoAugment(LidarAugmentInterface):

    def augment(self, data_sample: DataSample) -> DataSample:
        return data_sample


class GlobalAugment(LidarAugmentInterface):

    def __init__(
        self,
        yaw: float = 0,
        jitter: float = 0,
        scale: float = 1,
        xflip: bool = False,
        yflip: bool = False,
    ):
        assert scale >= 1.0, "This must be greater than or equal to 1, see docstring for details."

        self.yaw = yaw
        self.jitter = jitter
        self.scale = scale
        self.xflip = xflip
        self.yflip = yflip

        self.rand_yaw = None
        self.rand_jitter = None
        self.rand_scale = None
        self.rand_xflip = None
        self.rand_yflip = None

    def augment(self, data_sample: DataSample) -> DataSample:
        if self.xflip:
            self.rand_xflip = (np.random.random() < 0.5)
            if self.rand_xflip:
                data_sample = AugmentUtils.global_xflip(data_sample)

        if self.yflip:
            self.rand_yflip = (np.random.random() < 0.5)
            if self.rand_yflip:
                data_sample = AugmentUtils.global_yflip(data_sample)

        if self.scale > 1:
            self.rand_scale = AugmentUtils.generate_scale(self.scale)
            data_sample = AugmentUtils.global_scale(data_sample, self.rand_scale)

        if self.yaw > 0:
            self.rand_yaw = self.yaw * (2 * np.random.random() - 1)
            rand_rotation = Quaternion(axis=(0, 0, 1), angle=self.rand_yaw)
            data_sample = AugmentUtils.global_rotation(data_sample, rand_rotation)

        if self.jitter > 0:
            self.rand_jitter = np.clip(np.random.normal(0, self.jitter, 3), -3 * self.jitter, 3 * self.jitter)
            data_sample = AugmentUtils.global_jitter(data_sample, self.rand_jitter)

        return data_sample


class AugmentUtils:

    @staticmethod
    def generate_scale(scale: float) -> float:
        scale = np.random.uniform(1, scale)

        if np.random.random() < 0.5:
            scale = 1 / scale

        return scale

    @staticmethod
    def global_rotation(data_sample: DataSample, rotation: Quaternion) -> DataSample:
        pointcloud = data_sample[0]
        pointcloud.rotate(rotation)

        return DataSample(pointcloud)

    @staticmethod
    def global_jitter(data_sample: DataSample, jitter: np.ndarray) -> DataSample:
        pointcloud = data_sample[0]
        pointcloud.translate(jitter)

        return DataSample(pointcloud)

    @staticmethod
    def global_scale(data_sample: DataSample, scale: float) -> DataSample:
        pointcloud = data_sample[0]

        assert scale > 0, "The scale must be greater than 0."
        pointcloud.points[:3] *= scale

        return DataSample(pointcloud)

    @staticmethod
    def global_xflip(data_sample: DataSample) -> DataSample:
        pointcloud = data_sample[0]
        pointcloud.points[0] *= -1

        return DataSample(pointcloud)

    @staticmethod
    def global_yflip(data_sample: DataSample) -> DataSample:
        pointcloud = data_sample[0]
        pointcloud.points[1] *= -1

        return DataSample(pointcloud)
