import os
from typing import Dict, List, Optional, Tuple
from joblib import Memory
from itertools import accumulate

import numpy as np
import torch
import torch.utils.data as data

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from data.nuscenes.lidar import LidarPointCloud, LidarSegDatabaseInterface
from data.nuscenes.utils import DataSample
from data.nuscenes.augment import NoAugment, GlobalAugment


class LidarSegVoxelDataset(data.Dataset):

    def __init__(
        self,
        db: LidarSegDatabaseInterface,
        channel_mean: Optional[Tuple[float, ...]] = None,
        channel_scale: Optional[Tuple[float, ...]] = None,
        log_intensity: bool = False,
        augment: str = "NoAugment",
        voxel_grid_size: float = 0.1,
    ):
        self.db = db
        self.tokens = self.db.tokens
        self._min_distance = self.db.min_distance  # 0.9
        self.log_intensity = log_intensity  # False
        self.mean = channel_mean
        self.scale = channel_scale
        self.voxel_grid_size = voxel_grid_size

        # data augmentation on point clouds
        if augment == 'NoAugment':
            self.augment = NoAugment()
        elif augment == 'GlobalAugment':
            self.augment = GlobalAugment(yaw=3.1415926, jitter=0.2, scale=1.05, xflip=True, yflip=True)

        # set-up caching
        if os.environ.get('CACHE'):
            self.cache_dir = os.environ['CACHE']
        else:
            self.cache_dir = None

        if self.cache_dir is not None:
            print('=> Setting up DL with caching in {}'.format(self.cache_dir))
            assert os.path.exists(self.cache_dir), 'Cache dir {} does not exist. Please create.'.format(self.cache_dir)

            self.memory = Memory(self.cache_dir, verbose=False, compress=False)
            # ignore=['self'] flag is so that not the whole object (along with the input parameters) is hashed
            self.load_data = self.memory.cache(self.load_data, ignore=['self'])


    def __len__(self) -> int:
        return len(self.db.tokens)


    def __getitem__(self, index: int) -> Dict:
        token = self.db.tokens[index]

        return self.pull_item(token)


    def pull_item(self, token: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        points, points_labels, ring = self.load_data(token)

        points = np.concatenate((points, np.atleast_2d(points_labels).T), axis=1)  # [n, 5]
        points = LidarPointCloud(points=points.T)  # [n, 5]
        pc = self.augment.augment(DataSample(points))
        pc = pc[0]

        return self.process_after_augment(pc)


    def load_data(self, token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.db.load_from_db(token)
    

    def process_after_augment(self, pc: np.ndarray) -> Dict:
        point = pc.points[:4, :].T  # [n, 4]
        # point[:, 3] /= point[:, 3].max()

        point_label = pc.points[-1, :].T  # [n,]

        pc_ = np.round(point[:, :3] / self.voxel_grid_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        fea_ = point.astype(np.float32)

        _, inds, inverse_map = sparse_quantize(
            pc_,
            return_index=True,
            return_inverse=True,
        )

        pc = pc_[inds]
        fea = fea_[inds]
        label = point_label[inds]
        
        point_fea = SparseTensor(fea, pc)
        label = SparseTensor(label, pc)
        label_mapped = SparseTensor(point_label, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'point_fea': point_fea,
            'point_label': label,
            'point_label_mapped': label_mapped,
            'inverse_map': inverse_map,
        }

    @staticmethod
    def collate_batch(data):
        offset = [sample['point_fea'].C.shape[0] for sample in data]
        batch_data = sparse_collate_fn(data)
        batch_data.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))
        return batch_data
    