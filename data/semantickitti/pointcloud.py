import os
import random
import numpy as np
from torch.utils import data


LEARNING_MAP = {
    0:   0,   # "unlabeled"
    1:   0,   # "outlier" mapped to "unlabeled" --------------------------mapped
    10:  1,   # "car"
    11:  2,   # "bicycle"
    13:  5,   # "bus" mapped to "other-vehicle" --------------------------mapped
    15:  3,   # "motorcycle"
    16:  5,   # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18:  4,   # "truck"
    20:  5,   # "other-vehicle"
    30:  6,   # "person"
    31:  7,   # "bicyclist"
    32:  8,   # "motorcyclist"
    40:  9,   # "road"
    44:  10,  # "parking"
    48:  11,  # "sidewalk"
    49:  12,  # "other-ground"
    50:  13,  # "building"
    51:  14,  # "fence"
    52:  0,   # "other-structure" mapped to "unlabeled" ------------------mapped
    60:  9,   # "lane-marking" to "road" ---------------------------------mapped
    70:  15,  # "vegetation"
    71:  16,  # "trunk"
    72:  17,  # "terrain"
    80:  18,  # "pole"
    81:  19,  # "traffic-sign"
    99:  0,   # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,   # "moving-car" to "car" ------------------------------------mapped
    253: 7,   # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,   # "moving-person" to "person" ------------------------------mapped
    255: 8,   # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,   # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,   # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,   # "moving-truck" to "truck" --------------------------------mapped
    259: 5,   # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class SemkittiDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        data_split: str = None,
        if_scribble: bool = False,
        if_sup_only: bool = False,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.if_scribble = if_scribble
        self.if_sup_only = if_sup_only

        if data_split == 'full':
            self.data_split_list_path = None
        elif data_split == '1pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_1pct.txt'
        elif data_split == '10pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_10pct.txt'
        elif data_split == '20pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_20pct.txt'
        elif data_split == '50pct':
            self.data_split_list_path = 'script/split/semantickitti/semantickitti_50pct.txt'
        else:
            raise NotImplementedError

        if self.data_split_list_path:
            with open(self.data_split_list_path, "r") as f:
                data_split_list = f.read().splitlines()
            data_split_list = [i.split('train/')[-1] for i in data_split_list]
        
        if self.split == 'train':
            folders = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            folders = ['08']
        elif self.split == 'test':
            folders = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise NotImplementedError
        
        self.lidar_list = []
        for folder in folders:
            self.lidar_list += absoluteFilePaths('/'.join([self.root, str(folder).zfill(2), 'velodyne']))
        self.lidar_list.sort()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sample_idx)
    
    def get_kitti_points_ringID(self, points):
        scan_x = points[:, 0]
        scan_y = points[:, 1]

        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        ringID = np.cumsum(proj_y)
        ringID = np.clip(ringID, 0, 63)
        return ringID

    def __getitem__(self, index):
        raw_data = np.fromfile(self.lidar_list[index], dtype=np.float32).reshape((-1, 4))

        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:

            annotated_data = np.fromfile(
                self.lidar_list[index].replace('velodyne', 'labels')[:-3] + 'label',
                    dtype=np.uint32).reshape((-1, 1)
                )
            annotated_data = annotated_data & 0xFFFF
            annotated_data = np.vectorize(LEARNING_MAP.__getitem__)(annotated_data)
        
        ringID = self.get_kitti_points_ringID(raw_data).reshape((-1,1))
        raw_data= np.concatenate([raw_data, ringID.reshape(-1, 1)], axis=1).astype(np.float32)
        pc_data = {
            'xyzret': raw_data,
            'labels': annotated_data.astype(np.uint8),
            'path': self.lidar_list[index],
        }

        return pc_data
