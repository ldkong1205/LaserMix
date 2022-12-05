import abc
from typing import Any, IO, ByteString, Dict, List, Tuple, Union
import numpy as np
from pyquaternion import Quaternion


Color = Tuple[int, int, int, int]

class Label:

    def __init__(self, name: str, color: Color):
        self.name = name
        self.color = color

        for c in self.color:
            assert 0 <= c <= 255

    def __repr__(self):
        return "Label(name='{}', color={})".format(self.name, self.color)

    def __eq__(self, other):
        return self.name == other.name and self.color == other.color

    @property
    def normalized_color(self) -> Tuple[float, float, float, float]:
        return tuple([c / 255.0 for c in self.color])

    def serialize(self) -> Dict[str, Any]:
        return {'name': self.name, 'color': self.color}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        return Label(name=data['name'], color=tuple([int(channel) for channel in data['color']]))


class LidarPointCloud:

    def __init__(self, points):
        if points.ndim == 1:
            points = np.atleast_2d(points).T

        self.points = points

    @staticmethod
    def load_pcd_bin(pcd_bin: str) -> np.ndarray:
        scan = np.fromfile(pcd_bin, dtype=np.float32)
        points = scan.reshape((-1, 5))  # [m, 5]
        points = np.hstack((points, -1 * np.ones((points.shape[0], 1), dtype=np.float32)))  # [m, 6]

        return points.T

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        if file_name.endswith('.bin'):
            points = cls.load_pcd_bin(file_name)
        else:
            raise ValueError('Unsupported filetype {}.'.format(file_name))

        return cls(points)

    @classmethod
    def from_buffer(cls, pcd_data: Union[IO, ByteString], content_type: str = 'bin') -> 'LidarPointCloud':
        if content_type == 'bin':
            return cls(cls.load_pcd_bin(pcd_data, 1))
        elif content_type == 'bin2':
            return cls(cls.load_pcd_bin(pcd_data, 2))
        elif content_type == 'pcd':
            return cls(cls.load_pcd(pcd_data))
        else:
            raise NotImplementedError('Not implemented content type: %s' % content_type)

    @classmethod
    def make_random(cls):
        return LidarPointCloud(points=np.random.normal(0, 100, size=(4, 100)))

    def __eq__(self, other):
        return np.allclose(self.points, other.points, atol=1e-06)

    def copy(self):
        return LidarPointCloud(points=self.points.copy())

    def nbr_points(self):
        return self.points.shape[1]

    def subsample(self, ratio):
        assert 0 < ratio < 1

        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, min_dist: float):
        dist_from_orig = np.linalg.norm(self.points[:2, :], axis=0)
        self.points = self.points[:, dist_from_orig >= min_dist]

    def radius_filter(self, radius: float):
        keep = np.sqrt(self.points[0]**2 + self.points[1]**2) <= radius
        self.points = self.points[:, keep]

    def range_filter(self, xrange=(-np.inf, np.inf), yrange=(-np.inf, np.inf), zrange=(-np.inf, np.inf)):
        keep_x = np.logical_and(xrange[0] <= self.points[0], self.points[0] <= xrange[1])
        keep_y = np.logical_and(yrange[0] <= self.points[1], self.points[1] <= yrange[1])
        keep_z = np.logical_and(zrange[0] <= self.points[2], self.points[2] <= zrange[1])
        keep = np.logical_and(keep_x, np.logical_and(keep_y, keep_z))
        self.points = self.points[:, keep]

    def translate(self, x):
        self.points[:3] += x.reshape((-1, 1))

    def rotate(self, quaternion: Quaternion):
        self.points[:3] = np.dot(quaternion.rotation_matrix.astype(np.float32), self.points[:3])

    def transform(self, transf_matrix):
        transf_matrix = transf_matrix.astype(np.float32)
        self.points[:3, :] = transf_matrix[:3, :3] @ self.points[:3] + transf_matrix[:3, 3].reshape((-1, 1))

    def scale(self, scale: Tuple[float, float, float]) -> None:
        scale = np.array(scale)
        scale.shape = (3, 1)  # make sure it is a column vector
        self.points[:3, :] *= np.tile(scale, (1, self.nbr_points()))


class LidarSegDatabaseInterface(abc.ABC):

    @property
    @abc.abstractmethod
    def db_version(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def tokens(self) -> List[str]:
        pass

    @abc.abstractmethod
    def load_from_db(self, token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @property
    @abc.abstractmethod
    def caching_mode(self) -> bool:
        pass

    @abc.abstractmethod
    def load_range_view_coordinates(self, points: np.ndarray, proj_type: str, ring: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, int, int]:
        pass

    @property
    @abc.abstractmethod
    def global2local(self) -> Dict[str, str]:
        pass

    @property
    @abc.abstractmethod
    def local2id(self) -> Dict[str, int]:
        pass

    @property
    def labelmap(self) -> Dict[int, Tuple[str, Tuple]]:
        id2label = {v: k for k, v in self.raw_mapping['local2id'].items()}
        id2color = self.raw_mapping['id2color']
        assert sorted(list(id2color.keys())) == sorted(list(id2label.keys())), \
            'Error: Must map from the same label ids.'
        label_map = {int(id_): (id2label[id_], id2color[id_]) for id_ in id2color.keys()}
        return label_map

    @property
    def labelmap_for_evaluator(self) -> Dict[int, Label]:
        labelmap_for_eval = {}
        for key, value in self.labelmap.items():
            labelmap_for_eval[int(key)] = Label(name=value[0], color=value[1])
        return labelmap_for_eval

    @property
    @abc.abstractmethod
    def raw_mapping(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def xrange(self) -> Tuple[float, float]:
        pass

    @property
    @abc.abstractmethod
    def yrange(self) -> Tuple[float, float]:
        pass
