import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from joblib import Memory

from data.nuscenes.lidar import LidarPointCloud, LidarSegDatabaseInterface
from data.nuscenes.utils import get_range_view_inputs, DataSample
from data.nuscenes.augment import NoAugment, GlobalAugment


class LidarSegRangeViewDataset(data.Dataset):

    def __init__(
        self,
        db: LidarSegDatabaseInterface,
        channel_mean: Optional[Tuple[float, ...]] = None,
        channel_scale: Optional[Tuple[float, ...]] = None,
        log_intensity: bool = False,
        augment: str = "NoAugment",
        horiz_angular_res: float = 0.1875,
    ):
        self.db = db
        self.tokens = self.db.tokens
        self._min_distance = self.db.min_distance  # 0.9
        self.log_intensity = log_intensity  # False
        self.mean = channel_mean
        self.scale = channel_scale

        # data augmentation on point clouds
        if augment == 'NoAugment':
            self.augment = NoAugment()
        elif augment == 'GlobalAugment':
            self.augment = GlobalAugment(yaw=3.1415926, jitter=0.2, scale=1.05, xflip=True, yflip=True)

        # horizontal angular resolution, set 0.1875 for W=1920 or 0.375 for W=960
        self.horiz_angular_res = horiz_angular_res
        if self.horiz_angular_res == 0.1875:
            self.range_w = 1920
        elif self.horiz_angular_res == 0.375:
            self.range_w = 960
        print("Horizontal angular resolution: '{}'.".format(self.range_w))

        self.ch2idx = {'x': 0, 'y': 1, 'z': 2, 'intensity': 3, 'depth': -2, 'binary_mask': -1}

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


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, str]:
        token = self.db.tokens[index]
        rv, label_rv, idx_rv = self.pull_item(token)

        return {
            'scan': rv,
            'label': label_rv,
            'name': idx_rv,
        }


    def pull_item(self, token: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        rv, label_rv, idx_rv = self.pull_item_core(token)
        rv = torch.from_numpy(rv).permute(2, 0, 1).float()  # [6, H, W]
        label_rv = torch.from_numpy(label_rv.astype(int))   # [H, W]

        return rv, label_rv, idx_rv
    

    def load_data(self, token: str) -> Dict:
        return self.db.load_from_db(token)


    def pull_item_core(self, token: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points, points_labels, ring = self.load_data(token)

        points = np.concatenate((points, np.atleast_2d(points_labels).T), axis=1)  # [n, 5]
        points = LidarPointCloud(points=points.T)  # [n, 5]
        pc = self.augment.augment(DataSample(points))
        pc = pc[0]

        return self.process_after_augment(pc, ring, horiz_angular_res=self.horiz_angular_res)


    def process_after_augment(self, pc: np.ndarray, ring: np.ndarray, horiz_angular_res: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        proj_y, proj_x, ht, wt = self.db.load_range_view_coordinates(pc.points.T, ring=ring, horiz_angular_res=horiz_angular_res)

        rv, label_rv, idx_rv = get_range_view_inputs(
            points=pc.points[:-1].T, 
            points_label=pc.points[-1, :].T, 
            proj_y=proj_y, proj_x=proj_x, ht=ht, wt=wt, 
            add_binary_mask=True,
            ignore_label=self.db.local2id['ignore_label']
        )

        lidar_mask = rv[:, :, -1] == 1  # [H, W]
        if self.log_intensity:
            rv[lidar_mask, 3] = np.log(1e-5 + rv[lidar_mask, 3])
        if self.mean is not None:
            rv[lidar_mask, :-1] = rv[lidar_mask, :-1] - self.mean
        if self.scale is not None:
            rv[lidar_mask, :-1] = rv[lidar_mask, :-1] / self.scale
        
        return rv, label_rv, idx_rv


    @property
    def label_maps(self) -> Dict[str, Dict]:
        return self.db.labelmap


    @staticmethod
    def collate(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[np.ndarray], List[str]]:
        input_rv = []
        targets_rv = []
        index_rv = []
        tokens = []

        for sample in batch:
            input_rv.append(sample[0])
            targets_rv.append(sample[1])
            index_rv.append(sample[2])
            tokens.append(sample[3])
        
        return torch.stack(input_rv, 0), torch.stack(targets_rv, 0), index_rv, tokens


    @property
    def min_distance(self):
        return self._min_distance
